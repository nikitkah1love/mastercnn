import sys
import os
sys.path.insert(1, 'Utils')

if len(sys.argv) != 3:
    print("Parameters example: AWSCTD_windowed.py file_to_data.csv CNN")
    quit()

import tensorflow as tf
import numpy as np
import gc
from configparser import ConfigParser
from collections import defaultdict

# Configure TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.random.set_seed(0)

m_sWorkingDir = os.getcwd()
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

m_sDataFile = sys.argv[1]
m_sModel = sys.argv[2]

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
nKFolds = config.getint('MAIN', 'nKFolds')
bCategorical = config.getboolean('MAIN', 'bCategorical')
fLearningRate = config.getfloat('MAIN', 'fLearningRate', fallback=0.001)
bGradientClipping = config.getboolean('MAIN', 'bGradientClipping', fallback=True)

with open(m_sWorkingDir + 'config.ini', "r") as fIniFile:
    sConfig = fIniFile.read()
print("Config file:")
print(sConfig)

# Читаємо дані з trace_ids
m_nParametersCount = 0
m_nClassCount = 0
m_nWordCount = 0
Xtr, Ytr, trace_ids, m_nParametersCount, m_nClassCount, m_nWordCount = AWSCTDReadDataWindowed.ReadDataWindowedImpl(m_sDataFile, bCategorical)
print(f"Trace IDs shape: {trace_ids.shape}")
gc.collect()

import math

def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.8
    epochs_drop = 100.0
    lrate = initial_lrate * 1.0 / (1.0 + (0.0000005 * epoch * 100))
    if lrate < 0.0001:
        return 0.0001
    return lrate

from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
sMonitor = 'accuracy'

if bCategorical:
    sMonitor = 'categorical_accuracy'

lrate = LearningRateScheduler(step_decay, verbose=1)
callbacks_list = [EarlyStopping(monitor=sMonitor, patience=nPatience, verbose=1)]

arrWindowAcc = []
arrTraceAcc = []
arrLoss = []
arrTimeFit = []
arrTimeTest = []

# Ініціалізуємо списки для window-based метрик
arrWindowPrecision = []
arrWindowRecall = []
arrWindowF1 = []
arrWindowFPR = []
arrWindowFNR = []

# Ініціалізуємо списки для метрик
arrTracePrecisionMean = []
arrTraceRecallMean = []
arrTraceF1Mean = []
arrTraceFPRMean = []
arrTraceFNRMean = []

arrTraceAccMax = []
arrTracePrecisionMax = []
arrTraceRecallMax = []
arrTraceF1Max = []
arrTraceFPRMax = []
arrTraceFNRMax = []

# ВАЖЛИВО: Робимо split на рівні ТРЕЙСІВ, а не вікон!
# Інакше вікна одного трейсу потраплять і в train, і в test
from sklearn.model_selection import StratifiedKFold

# Отримуємо унікальні трейси
unique_traces = np.unique(trace_ids)
print(f"\n📊 Підготовка до StratifiedKFold split:")
print(f"   Всього унікальних трейсів: {len(unique_traces)}")
print(f"   Всього вікон: {len(Xtr)}")

# Створюємо mapping: trace_id -> індекси вікон
trace_to_windows = {}
for i, trace_id in enumerate(trace_ids):
    if trace_id not in trace_to_windows:
        trace_to_windows[trace_id] = []
    trace_to_windows[trace_id].append(i)

# Отримуємо лейбл для кожного трейсу (беремо лейбл першого вікна)
trace_labels = []
for trace_id in unique_traces:
    first_window_idx = trace_to_windows[trace_id][0]
    trace_labels.append(Ytr[first_window_idx])

trace_labels = np.array(trace_labels).flatten()

# Перевіряємо розподіл класів
unique_labels, label_counts = np.unique(trace_labels, return_counts=True)
print(f"\n   Розподіл класів по трейсам:")
for label, count in zip(unique_labels, label_counts):
    percentage = (count / len(trace_labels)) * 100
    class_name = "Malware" if label == 1 else "Benign"
    print(f"      {class_name}: {count} трейсів ({percentage:.1f}%)")

print(f"   Середня кількість вікон на трейс: {len(Xtr) / len(unique_traces):.1f}")

# StratifiedKFold на рівні трейсів - зберігає пропорції класів
kfold = StratifiedKFold(n_splits=nKFolds, shuffle=True, random_state=0)

from time import gmtime, strftime
sTime = strftime("%Y-%m-%d %H:%M:%S", gmtime())
import time
start = time.time()
nFoldNumber = 1

print("\n🎯 Тренування з агрегацією по трейсам (StratifiedKFold split)\n")

for train_traces_idx, test_traces_idx in kfold.split(unique_traces, trace_labels):
    # Отримуємо trace_ids для train та test
    train_trace_ids = unique_traces[train_traces_idx]
    test_trace_ids = unique_traces[test_traces_idx]
    
    # Конвертуємо trace_ids в індекси вікон
    train = []
    for trace_id in train_trace_ids:
        train.extend(trace_to_windows[trace_id])
    
    test = []
    for trace_id in test_trace_ids:
        test.extend(trace_to_windows[trace_id])
    
    train = np.array(train)
    test = np.array(test)
    
    # Перевіряємо розподіл класів у train та test
    train_trace_labels = trace_labels[train_traces_idx]
    test_trace_labels = trace_labels[test_traces_idx]
    
    train_unique, train_counts = np.unique(train_trace_labels, return_counts=True)
    test_unique, test_counts = np.unique(test_trace_labels, return_counts=True)
    
    print(f"KFold number: {nFoldNumber}")
    print(f"   Train: {len(train_trace_ids)} трейсів ({len(train)} вікон)")
    for label, count in zip(train_unique, train_counts):
        percentage = (count / len(train_trace_labels)) * 100
        class_name = "Malware" if label == 1 else "Benign"
        print(f"      {class_name}: {count} трейсів ({percentage:.1f}%)")
    
    print(f"   Test:  {len(test_trace_ids)} трейсів ({len(test)} вікон)")
    for label, count in zip(test_unique, test_counts):
        percentage = (count / len(test_trace_labels)) * 100
        class_name = "Malware" if label == 1 else "Benign"
        print(f"      {class_name}: {count} трейсів ({percentage:.1f}%)")
    
    # Перевірка що немає перетину
    train_set = set(train_trace_ids)
    test_set = set(test_trace_ids)
    overlap = train_set.intersection(test_set)
    if len(overlap) > 0:
        print(f"   ⚠️  УВАГА: Знайдено перетин {len(overlap)} трейсів!")
    else:
        print(f"   ✅ Немає перетину між train та test трейсами")
    overlap = train_set.intersection(test_set)
    if len(overlap) > 0:
        print(f"   ⚠️  УВАГА: Знайдено перетин {len(overlap)} трейсів!")
    else:
        print(f"   ✅ Немає перетину між train та test трейсами")
    print(f"KFold number: {nFoldNumber}")
    nFoldNumber += 1
    
    # Create model
    model = AWSCTDCreateModel.CreateModelImpl(m_sModel, m_nWordCount, m_nClassCount, m_nParametersCount, bCategorical, fLearningRate, bGradientClipping)
    
    startFit = time.time()
    # Train на вікнах
    history = model.fit([Xtr[train]], Ytr[train], epochs=nEpochs, batch_size=nBatchSize, callbacks=callbacks_list, verbose=1)
    endFit = time.time()
    tmExecFit = endFit - startFit
    
    startTest = time.time()
    scores = model.evaluate([Xtr[test]], Ytr[test], verbose=0)
    endTest = time.time()
    tmExecTest = endTest - startTest
    
    # Predictions на вікнах
    y_pred_windows = model.predict([Xtr[test]], verbose=0)
    
    # Window-based accuracy
    if bCategorical:
        y_pred_class = np.argmax(y_pred_windows, axis=1)
        y_true_class = np.argmax(Ytr[test], axis=1)
    else:
        y_pred_class = (y_pred_windows > 0.5).astype(int).flatten()
        y_true_class = Ytr[test].flatten()
    
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
    trace_ids_test = trace_ids[test]
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
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    
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
    
    arrWindowAcc.append(window_acc)
    arrWindowPrecision.append(precision_window)
    arrWindowRecall.append(recall_window)
    arrWindowF1.append(f1_window)
    arrWindowFPR.append(fpr_window)
    arrWindowFNR.append(fnr_window)
    
    arrTraceAcc.append(trace_acc_mean)
    arrLoss.append(scores[0])
    arrTimeFit.append(tmExecFit)
    arrTimeTest.append(tmExecTest)
    
    # Зберігаємо метрики для обох методів агрегації
    arrTracePrecisionMean.append(precision_mean)
    arrTraceRecallMean.append(recall_mean)
    arrTraceF1Mean.append(f1_mean)
    arrTraceFPRMean.append(fpr_mean)
    arrTraceFNRMean.append(fnr_mean)
    
    arrTraceAccMax.append(trace_acc_max)
    arrTracePrecisionMax.append(precision_max)
    arrTraceRecallMax.append(recall_max)
    arrTraceF1Max.append(f1_max)
    arrTraceFPRMax.append(fpr_max)
    arrTraceFNRMax.append(fnr_max)
    arrTimeTest.append(tmExecTest)
    
    try:
        del model
    except:
        pass
    AWSCTDClearSesion.reset_keras()

end = time.time()

tmExec = end - start
dTimeTrain = np.mean(arrTimeFit)
dTimeTest = np.mean(arrTimeTest)

print("\n" + "="*70)
print("📈 ФІНАЛЬНІ РЕЗУЛЬТАТИ")
print("="*70)
print(f"All time            : {tmExec:.2f}s")
print(f"Training time       : {dTimeTrain:.2f}s")
print(f"Testing time        : {dTimeTest:.2f}s")
print()
print("Window-based metrics:")
print(f"  Accuracy:  {np.mean(arrWindowAcc):.2f}% (+/- {np.std(arrWindowAcc):.2f})")
print(f"  Precision: {np.mean(arrWindowPrecision):.4f} (+/- {np.std(arrWindowPrecision):.4f})")
print(f"  Recall:    {np.mean(arrWindowRecall):.4f} (+/- {np.std(arrWindowRecall):.4f})")
print(f"  F1-Score:  {np.mean(arrWindowF1):.4f} (+/- {np.std(arrWindowF1):.4f})")
print(f"  FPR:       {np.mean(arrWindowFPR):.4f} (+/- {np.std(arrWindowFPR):.4f})")
print(f"  FNR:       {np.mean(arrWindowFNR):.4f} (+/- {np.std(arrWindowFNR):.4f})")
print()
print("MEAN Aggregation:")
print(f"  Accuracy:  {np.mean(arrTraceAcc):.2f}% (+/- {np.std(arrTraceAcc):.2f})")
print(f"  Precision: {np.mean(arrTracePrecisionMean):.4f} (+/- {np.std(arrTracePrecisionMean):.4f})")
print(f"  Recall:    {np.mean(arrTraceRecallMean):.4f} (+/- {np.std(arrTraceRecallMean):.4f})")
print(f"  F1-Score:  {np.mean(arrTraceF1Mean):.4f} (+/- {np.std(arrTraceF1Mean):.4f})")
print(f"  FPR:       {np.mean(arrTraceFPRMean):.4f} (+/- {np.std(arrTraceFPRMean):.4f})")
print(f"  FNR:       {np.mean(arrTraceFNRMean):.4f} (+/- {np.std(arrTraceFNRMean):.4f})")
print()
print("MAX Aggregation:")
print(f"  Accuracy:  {np.mean(arrTraceAccMax):.2f}% (+/- {np.std(arrTraceAccMax):.2f})")
print(f"  Precision: {np.mean(arrTracePrecisionMax):.4f} (+/- {np.std(arrTracePrecisionMax):.4f})")
print(f"  Recall:    {np.mean(arrTraceRecallMax):.4f} (+/- {np.std(arrTraceRecallMax):.4f})")
print(f"  F1-Score:  {np.mean(arrTraceF1Max):.4f} (+/- {np.std(arrTraceF1Max):.4f})")
print(f"  FPR:       {np.mean(arrTraceFPRMax):.4f} (+/- {np.std(arrTraceFPRMax):.4f})")
print(f"  FNR:       {np.mean(arrTraceFNRMax):.4f} (+/- {np.std(arrTraceFNRMax):.4f})")
print()
print(f"Loss: {np.mean(arrLoss):.4f} (+/- {np.std(arrLoss):.4f})")
print("="*70)

# Зберігаємо результати
import sqlite3
con = sqlite3.connect('results_updated.db')
sTestTag = m_sModel + "_windowed_stratified"

dWindowAcc = np.mean(arrWindowAcc)
dTraceAccMean = np.mean(arrTraceAcc)
dTraceAccMax = np.mean(arrTraceAccMax)
dLoss = np.mean(arrLoss)

dWindowAccStd = np.std(arrWindowAcc)
dTraceAccMeanStd = np.std(arrTraceAcc)
dTraceAccMaxStd = np.std(arrTraceAccMax)
dLossStd = np.std(arrLoss)

# Window-based метрики
dWindowPrecision = np.mean(arrWindowPrecision)
dWindowRecall = np.mean(arrWindowRecall)
dWindowF1 = np.mean(arrWindowF1)
dWindowFPR = np.mean(arrWindowFPR)
dWindowFNR = np.mean(arrWindowFNR)

dWindowPrecisionStd = np.std(arrWindowPrecision)
dWindowRecallStd = np.std(arrWindowRecall)
dWindowF1Std = np.std(arrWindowF1)
dWindowFPRStd = np.std(arrWindowFPR)
dWindowFNRStd = np.std(arrWindowFNR)

# Trace MEAN метрики
dPrecisionMean = np.mean(arrTracePrecisionMean)
dRecallMean = np.mean(arrTraceRecallMean)
dF1Mean = np.mean(arrTraceF1Mean)
dFPRMean = np.mean(arrTraceFPRMean)
dFNRMean = np.mean(arrTraceFNRMean)

dPrecisionMeanStd = np.std(arrTracePrecisionMean)
dRecallMeanStd = np.std(arrTraceRecallMean)
dF1MeanStd = np.std(arrTraceF1Mean)
dFPRMeanStd = np.std(arrTraceFPRMean)
dFNRMeanStd = np.std(arrTraceFNRMean)

# Trace MAX метрики
dPrecisionMax = np.mean(arrTracePrecisionMax)
dRecallMax = np.mean(arrTraceRecallMax)
dF1Max = np.mean(arrTraceF1Max)
dFPRMax = np.mean(arrTraceFPRMax)
dFNRMax = np.mean(arrTraceFNRMax)

dPrecisionMaxStd = np.std(arrTracePrecisionMax)
dRecallMaxStd = np.std(arrTraceRecallMax)
dF1MaxStd = np.std(arrTraceF1Max)
dFPRMaxStd = np.std(arrTraceFPRMax)
dFNRMaxStd = np.std(arrTraceFNRMax)

result = (m_sDataFile, m_nParametersCount, m_nClassCount, nEpochs, nBatchSize, 
          tmExec, dWindowAcc, dTraceAccMean, dTraceAccMax, dLoss, dTimeTrain, dTimeTest, 
          sTestTag, dWindowAccStd, dTraceAccMeanStd, dTraceAccMaxStd, dLossStd, 
          dWindowPrecision, dWindowRecall, dWindowF1, dWindowFPR, dWindowFNR,
          dWindowPrecisionStd, dWindowRecallStd, dWindowF1Std, dWindowFPRStd, dWindowFNRStd,
          dPrecisionMean, dRecallMean, dF1Mean, dFPRMean, dFNRMean,
          dPrecisionMeanStd, dRecallMeanStd, dF1MeanStd, dFPRMeanStd, dFNRMeanStd,
          dPrecisionMax, dRecallMax, dF1Max, dFPRMax, dFNRMax,
          dPrecisionMaxStd, dRecallMaxStd, dF1MaxStd, dFPRMaxStd, dFNRMaxStd,
          sTime, sConfig)

sql = """INSERT INTO results_windowed_v2
         (File, ParamCount, ClassCount, Epochs, BatchSize, Time, 
          WindowAcc, TraceAccMean, TraceAccMax, Loss, TimeTrain, TimeTest, Comment, 
          WindowAccStd, TraceAccMeanStd, TraceAccMaxStd, LossStd,
          WindowPrecision, WindowRecall, WindowF1, WindowFPR, WindowFNR,
          WindowPrecisionStd, WindowRecallStd, WindowF1Std, WindowFPRStd, WindowFNRStd,
          PrecisionMean, RecallMean, F1Mean, FPRMean, FNRMean,
          PrecisionMeanStd, RecallMeanStd, F1MeanStd, FPRMeanStd, FNRMeanStd,
          PrecisionMax, RecallMax, F1Max, FPRMax, FNRMax,
          PrecisionMaxStd, RecallMaxStd, F1MaxStd, FPRMaxStd, FNRMaxStd,
          ExecutionTime, Config)
         VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
      """

try:
    cur = con.cursor()
    cur.execute(sql, result)
    con.commit()
    print("✅ Результати збережено в базу даних")
except Exception as e:
    print(f"⚠️  Помилка збереження в БД: {e}")
    print("   Можливо потрібно створити таблицю results_windowed_v2")

print("\n✅ Тренування завершено!")
