import sys
import os
sys.path.insert(1, 'Utils')

if len(sys.argv) != 4:
    print("Parameters example: AWSCTD_train_test_split.py train_data.csv test_data.csv CNN")
    quit()

import tensorflow as tf
import numpy as np
import gc
from configparser import ConfigParser

# Configure TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.random.set_seed(0)

m_sWorkingDir = os.path.dirname(os.path.abspath(__file__))
m_sWorkingDir = m_sWorkingDir + '/'

# Read device configuration first
config = ConfigParser()
config.read(m_sWorkingDir + 'config.ini')
sDevice = config.get('MAIN', 'sDevice', fallback='auto')

# Configure device based on user preference
if sDevice.lower() == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print("🖥️  Примусово використовуємо CPU")
elif sDevice.lower() == 'gpu':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if not gpus:
        print("❌ GPU не знайдено, але в конфігурації вказано GPU")
        quit()
    print("🚀 Примусово використовуємо GPU")
else:
    print("🔄 Автоматичний вибір пристрою")

# Configure GPU if available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ Знайдено {len(gpus)} GPU пристроїв")
    except RuntimeError as e:
        print(e)
else:
    print("💻 Використовуємо CPU")

m_sTrainFile = sys.argv[1]
m_sTestFile = sys.argv[2]
m_sModel = sys.argv[3]

np.random.seed(0)

import AWSCTDReadDataWindowed
import AWSCTDCreateModel
import AWSCTDClearSesion

print(m_sWorkingDir)

config = ConfigParser()
config.read(m_sWorkingDir + 'config.ini')
nEpochs = config.getint('MAIN', 'nEpochs')
nBatchSize = config.getint('MAIN', 'nBatchSize')
nPatience = config.getint('MAIN', 'nPatience')
bCategorical = config.getboolean('MAIN', 'bCategorical')
fLearningRate = config.getfloat('MAIN', 'fLearningRate', fallback=0.001)
bGradientClipping = config.getboolean('MAIN', 'bGradientClipping', fallback=True)

with open(m_sWorkingDir + 'config.ini', "r") as fIniFile:
    sConfig = fIniFile.read()
print("Config file:")
print(sConfig)

# Читаємо тренувальні дані
print("\n📖 Читання тренувального датасету...")
Xtr_train, Ytr_train, trace_ids_train, m_nParametersCount, m_nClassCount, m_nWordCount = AWSCTDReadDataWindowed.ReadDataWindowedImpl(m_sTrainFile, bCategorical)
print(f"✅ Train data: X shape: {Xtr_train.shape}, Y shape: {Ytr_train.shape}")
gc.collect()

# Читаємо тестові дані
print("\n📖 Читання тестового датасету...")
Xtr_test, Ytr_test, trace_ids_test, _, _, _ = AWSCTDReadDataWindowed.ReadDataWindowedImpl(m_sTestFile, bCategorical)
print(f"✅ Test data: X shape: {Xtr_test.shape}, Y shape: {Ytr_test.shape}")
gc.collect()

from tensorflow.keras.callbacks import EarlyStopping
sMonitor = 'accuracy'

if bCategorical:
    sMonitor = 'categorical_accuracy'

callbacks_list = [EarlyStopping(monitor=sMonitor, patience=nPatience, verbose=1)]

from time import gmtime, strftime
import time

sTime = strftime("%Y-%m-%d %H:%M:%S", gmtime())
start = time.time()

print("\n🎯 Тренування моделі\n")

# Create model
model = AWSCTDCreateModel.CreateModelImpl(m_sModel, m_nWordCount, m_nClassCount, m_nParametersCount, bCategorical, fLearningRate, bGradientClipping)

startFit = time.time()
# Train на всіх тренувальних даних
history = model.fit([Xtr_train], Ytr_train, epochs=nEpochs, batch_size=nBatchSize, callbacks=callbacks_list, verbose=1)
endFit = time.time()
tmExecFit = endFit - startFit

startTest = time.time()
scores = model.evaluate([Xtr_test], Ytr_test, verbose=0)
endTest = time.time()
tmExecTest = endTest - startTest

# Predictions на тестових даних
y_pred_windows = model.predict([Xtr_test], verbose=0)

# Window-based accuracy
if bCategorical:
    y_pred_class = np.argmax(y_pred_windows, axis=1)
    y_true_class = np.argmax(Ytr_test, axis=1)
else:
    y_pred_class = (y_pred_windows > 0.5).astype(int).flatten()
    y_true_class = Ytr_test.flatten()

window_acc = np.mean(y_pred_class == y_true_class) * 100

# Window-based метрики
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

tn_window, fp_window, fn_window, tp_window = confusion_matrix(y_true_class, y_pred_class).ravel()

precision_window = precision_score(y_true_class, y_pred_class, zero_division=0)
recall_window = recall_score(y_true_class, y_pred_class, zero_division=0)
f1_window = f1_score(y_true_class, y_pred_class, zero_division=0)
fpr_window = fp_window / (fp_window + tn_window) if (fp_window + tn_window) > 0 else 0
fnr_window = fn_window / (fn_window + tp_window) if (fn_window + tp_window) > 0 else 0

print(f"\n📊 Window-based metrics:")
print(f"   Accuracy:  {window_acc:.2f}%")
print(f"   Precision: {precision_window:.4f}")
print(f"   Recall:    {recall_window:.4f}")
print(f"   F1-Score:  {f1_window:.4f}")
print(f"   FPR:       {fpr_window:.4f}")
print(f"   FNR:       {fnr_window:.4f}")
print(f"   TP: {tp_window}, TN: {tn_window}, FP: {fp_window}, FN: {fn_window}")

# Trace-based accuracy з агрегацією MEAN та MAX
unique_traces_test = np.unique(trace_ids_test)

trace_predictions = {}
trace_true_labels = {}

# Агрегуємо predictions по трейсам
for i, trace_id in enumerate(trace_ids_test):
    if trace_id not in trace_predictions:
        trace_predictions[trace_id] = []
        trace_true_labels[trace_id] = y_true_class[i]
    
    # Додаємо prediction для цього вікна
    if bCategorical:
        trace_predictions[trace_id].append(y_pred_windows[i])
    else:
        trace_predictions[trace_id].append(y_pred_windows[i][0])

# Обчислюємо predictions для обох методів агрегації
trace_total = len(unique_traces_test)

# Для MEAN агрегації
trace_pred_mean = []
trace_true = []

# Для MAX агрегації
trace_pred_max = []

for trace_id in unique_traces_test:
    true_label = trace_true_labels[trace_id]
    trace_true.append(true_label)
    
    # MEAN агрегація
    mean_pred = np.mean(trace_predictions[trace_id], axis=0)
    if bCategorical:
        final_pred_mean = np.argmax(mean_pred)
    else:
        final_pred_mean = 1 if mean_pred > 0.5 else 0
    trace_pred_mean.append(final_pred_mean)
    
    # MAX агрегація (максимальна ймовірність malware)
    max_pred = np.max(trace_predictions[trace_id], axis=0)
    if bCategorical:
        final_pred_max = np.argmax(max_pred)
    else:
        final_pred_max = 1 if max_pred > 0.5 else 0
    trace_pred_max.append(final_pred_max)

trace_pred_mean = np.array(trace_pred_mean)
trace_pred_max = np.array(trace_pred_max)
trace_true = np.array(trace_true)

# Обчислюємо метрики для MEAN агрегації
print(f"\n🎯 Trace-based metrics (MEAN aggregation):")

# Confusion matrix
tn_mean, fp_mean, fn_mean, tp_mean = confusion_matrix(trace_true, trace_pred_mean).ravel()

# Метрики
trace_acc_mean = np.mean(trace_pred_mean == trace_true) * 100
precision_mean = precision_score(trace_true, trace_pred_mean, zero_division=0)
recall_mean = recall_score(trace_true, trace_pred_mean, zero_division=0)
f1_mean = f1_score(trace_true, trace_pred_mean, zero_division=0)
fpr_mean = fp_mean / (fp_mean + tn_mean) if (fp_mean + tn_mean) > 0 else 0
fnr_mean = fn_mean / (fn_mean + tp_mean) if (fn_mean + tp_mean) > 0 else 0

print(f"   Accuracy:  {trace_acc_mean:.2f}%")
print(f"   Precision: {precision_mean:.4f}")
print(f"   Recall:    {recall_mean:.4f}")
print(f"   F1-Score:  {f1_mean:.4f}")
print(f"   FPR:       {fpr_mean:.4f}")
print(f"   FNR:       {fnr_mean:.4f}")
print(f"   TP: {tp_mean}, TN: {tn_mean}, FP: {fp_mean}, FN: {fn_mean}")

# Обчислюємо метрики для MAX агрегації
print(f"\n🎯 Trace-based metrics (MAX aggregation):")

# Confusion matrix
tn_max, fp_max, fn_max, tp_max = confusion_matrix(trace_true, trace_pred_max).ravel()

# Метрики
trace_acc_max = np.mean(trace_pred_max == trace_true) * 100
precision_max = precision_score(trace_true, trace_pred_max, zero_division=0)
recall_max = recall_score(trace_true, trace_pred_max, zero_division=0)
f1_max = f1_score(trace_true, trace_pred_max, zero_division=0)
fpr_max = fp_max / (fp_max + tn_max) if (fp_max + tn_max) > 0 else 0
fnr_max = fn_max / (fn_max + tp_max) if (fn_max + tp_max) > 0 else 0

print(f"   Accuracy:  {trace_acc_max:.2f}%")
print(f"   Precision: {precision_max:.4f}")
print(f"   Recall:    {recall_max:.4f}")
print(f"   F1-Score:  {f1_max:.4f}")
print(f"   FPR:       {fpr_max:.4f}")
print(f"   FNR:       {fnr_max:.4f}")
print(f"   TP: {tp_max}, TN: {tn_max}, FP: {fp_max}, FN: {fn_max}")

end = time.time()
tmExec = end - start

print("\n" + "="*70)
print("📈 ФІНАЛЬНІ РЕЗУЛЬТАТИ")
print("="*70)
print(f"All time            : {tmExec:.2f}s")
print(f"Training time       : {tmExecFit:.2f}s")
print(f"Testing time        : {tmExecTest:.2f}s")
print()
print("Window-based metrics:")
print(f"  Accuracy:  {window_acc:.2f}%")
print(f"  Precision: {precision_window:.4f}")
print(f"  Recall:    {recall_window:.4f}")
print(f"  F1-Score:  {f1_window:.4f}")
print(f"  FPR:       {fpr_window:.4f}")
print(f"  FNR:       {fnr_window:.4f}")
print()
print("MEAN Aggregation:")
print(f"  Accuracy:  {trace_acc_mean:.2f}%")
print(f"  Precision: {precision_mean:.4f}")
print(f"  Recall:    {recall_mean:.4f}")
print(f"  F1-Score:  {f1_mean:.4f}")
print(f"  FPR:       {fpr_mean:.4f}")
print(f"  FNR:       {fnr_mean:.4f}")
print()
print("MAX Aggregation:")
print(f"  Accuracy:  {trace_acc_max:.2f}%")
print(f"  Precision: {precision_max:.4f}")
print(f"  Recall:    {recall_max:.4f}")
print(f"  F1-Score:  {f1_max:.4f}")
print(f"  FPR:       {fpr_max:.4f}")
print(f"  FNR:       {fnr_max:.4f}")
print()
print(f"Loss: {scores[0]:.4f}")
print("="*70)

print(f"\n✅ Train: {m_sTrainFile}")
print(f"✅ Test:  {m_sTestFile}")
print("\n✅ Тренування завершено!")
