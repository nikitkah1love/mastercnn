#!/usr/bin/env python3
"""
AWSCTD Embedding з StratifiedKFold крос-валідацією.
Приймає один CSV файл, ділить на фолди зі збереженням балансу класів.
Usage: AWSCTD_embedding_kfold.py data.csv AWSCTD-CNN-S-EMBEDDING [--embedding-dim=16]
"""
import sys
import os
sys.path.insert(1, 'Utils')

if len(sys.argv) < 3 or len(sys.argv) > 6:
    print("Usage: AWSCTD_embedding_kfold.py data.csv AWSCTD-CNN-S-EMBEDDING [--trace-aggregation] [--embedding-dim=16]")
    quit()

nEmbeddingDim = 16
bTraceAggregation = False
for arg in sys.argv[3:]:
    if arg == '--trace-aggregation':
        bTraceAggregation = True
        print("📊 Trace aggregation увімкнена")
    elif arg.startswith('--embedding-dim='):
        nEmbeddingDim = int(arg.split('=')[1])
        print(f"📐 Embedding dimension: {nEmbeddingDim}")

import tensorflow as tf
import numpy as np
import gc
from configparser import ConfigParser

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.random.set_seed(0)
np.random.seed(0)

m_sWorkingDir = os.path.dirname(os.path.abspath(__file__)) + '/'

config = ConfigParser()
config.read(m_sWorkingDir + 'config.ini')
sDevice = config.get('MAIN', 'sDevice', fallback='auto')

if sDevice.lower() == 'cpu':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    print("🖥️  Примусово використовуємо CPU")
elif sDevice.lower() == 'gpu':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if not gpus:
        print("❌ GPU не знайдено")
        quit()
    print("🚀 Примусово використовуємо GPU")

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ Знайдено {len(gpus)} GPU")
    except RuntimeError as e:
        print(e)
else:
    print("💻 Використовуємо CPU")

m_sDataFile = sys.argv[1]
m_sModel = sys.argv[2]

import AWSCTDReadDataEmbedding
import AWSCTDCreateModel
import AWSCTDClearSesion

nEpochs = config.getint('MAIN', 'nEpochs')
nBatchSize = config.getint('MAIN', 'nBatchSize')
nPatience = config.getint('MAIN', 'nPatience')
nKFolds = config.getint('MAIN', 'nKFolds')
bCategorical = config.getboolean('MAIN', 'bCategorical')
fLearningRate = config.getfloat('MAIN', 'fLearningRate', fallback=0.001)
bGradientClipping = config.getboolean('MAIN', 'bGradientClipping', fallback=True)

with open(m_sWorkingDir + 'config.ini', "r") as f:
    print("Config file:")
    print(f.read())

# Читаємо дані
print("\n📖 Читання датасету...")
Xtr, Ytr, trace_ids, m_nParametersCount, m_nClassCount, m_nWordCount = AWSCTDReadDataEmbedding.ReadDataEmbeddingImpl(m_sDataFile, bCategorical)
print(f"✅ Data: X shape: {Xtr.shape}, Y shape: {Ytr.shape}")
gc.collect()

from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from time import gmtime, strftime
import time

sMonitor = 'categorical_accuracy' if bCategorical else 'accuracy'

# Лейбли для стратифікації
if bCategorical:
    y_stratify = np.argmax(Ytr, axis=1)
else:
    y_stratify = Ytr.ravel()

kfold = StratifiedKFold(n_splits=nKFolds, shuffle=True, random_state=0)

start = time.time()
nFoldNumber = 1

arrAcc = []
arrLoss = []
arrTimeFit = []
arrTimeTest = []
cm = np.zeros((m_nClassCount, m_nClassCount), dtype=int)

# Trace aggregation accumulation
if bTraceAggregation:
    cm_mean_total = np.zeros((m_nClassCount, m_nClassCount), dtype=int)
    cm_max_total = np.zeros((m_nClassCount, m_nClassCount), dtype=int)
    arrAccMean = []
    arrAccMax = []

print(f"\n🎯 StratifiedKFold крос-валідація ({nKFolds} фолдів) з Embedding\n")

for train_idx, test_idx in kfold.split(Xtr, y_stratify):
    print(f"\n{'='*70}")
    print(f"📊 KFold {nFoldNumber}/{nKFolds}")
    print(f"{'='*70}")
    nFoldNumber += 1

    model = AWSCTDCreateModel.CreateModelImpl(m_sModel, m_nWordCount, m_nClassCount, m_nParametersCount, bCategorical, fLearningRate, bGradientClipping, nEmbeddingDim)
    callbacks_list = [EarlyStopping(monitor=sMonitor, patience=nPatience, verbose=1)]

    startFit = time.time()
    history = model.fit(Xtr[train_idx], Ytr[train_idx], epochs=nEpochs, batch_size=nBatchSize, callbacks=callbacks_list, verbose=1)
    endFit = time.time()
    tmFit = endFit - startFit

    startTest = time.time()
    scores = model.evaluate(Xtr[test_idx], Ytr[test_idx], verbose=0)
    endTest = time.time()
    tmTest = endTest - startTest

    y_pred = model.predict(Xtr[test_idx], verbose=0)

    if bCategorical:
        y_pred_class = np.argmax(y_pred, axis=1)
        y_true_class = np.argmax(Ytr[test_idx], axis=1)
    else:
        y_pred_class = (y_pred > 0.5).astype(int).flatten()
        y_true_class = Ytr[test_idx].flatten()

    fold_acc = np.mean(y_pred_class == y_true_class) * 100
    arrAcc.append(fold_acc)
    arrLoss.append(scores[0])
    arrTimeFit.append(tmFit)
    arrTimeTest.append(tmTest)

    cm += confusion_matrix(y_true_class, y_pred_class, labels=list(range(m_nClassCount)))

    print(f"  Window accuracy: {fold_acc:.2f}%")
    print(f"  Fold loss: {scores[0]:.4f}")
    print(f"  Train time: {tmFit:.2f}s, Test time: {tmTest:.2f}s")

    # Trace aggregation per fold
    if bTraceAggregation:
        fold_trace_ids = trace_ids[test_idx]
        unique_fold_traces = np.unique(fold_trace_ids)

        trace_predictions = {}
        trace_true_labels = {}
        for i, idx in enumerate(test_idx):
            tid = trace_ids[idx]
            if tid not in trace_predictions:
                trace_predictions[tid] = []
                trace_true_labels[tid] = y_true_class[i]
            if bCategorical:
                trace_predictions[tid].append(y_pred[i])
            else:
                trace_predictions[tid].append(y_pred[i][0])

        trace_true = []
        trace_pred_mean = []
        trace_pred_max = []
        for tid in unique_fold_traces:
            trace_true.append(trace_true_labels[tid])
            mean_p = np.mean(trace_predictions[tid], axis=0)
            max_p = np.max(trace_predictions[tid], axis=0)
            if bCategorical:
                trace_pred_mean.append(np.argmax(mean_p))
                trace_pred_max.append(np.argmax(max_p))
            else:
                trace_pred_mean.append(1 if mean_p > 0.5 else 0)
                trace_pred_max.append(1 if max_p > 0.5 else 0)

        trace_true = np.array(trace_true)
        trace_pred_mean = np.array(trace_pred_mean)
        trace_pred_max = np.array(trace_pred_max)

        fold_acc_mean = np.mean(trace_pred_mean == trace_true) * 100
        fold_acc_max = np.mean(trace_pred_max == trace_true) * 100
        arrAccMean.append(fold_acc_mean)
        arrAccMax.append(fold_acc_max)

        cm_mean_total += confusion_matrix(trace_true, trace_pred_mean, labels=list(range(m_nClassCount)))
        cm_max_total += confusion_matrix(trace_true, trace_pred_max, labels=list(range(m_nClassCount)))

        print(f"  MEAN trace accuracy: {fold_acc_mean:.2f}%")
        print(f"  MAX trace accuracy:  {fold_acc_max:.2f}%")

    try:
        del model
    except:
        pass
    AWSCTDClearSesion.reset_keras()

end = time.time()
tmExec = end - start

# Фінальні метрики
dAcc = np.mean(arrAcc)
dAccStd = np.std(arrAcc)
dLoss = np.mean(arrLoss)
dTimeTrain = np.mean(arrTimeFit)
dTimeTest = np.mean(arrTimeTest)

# Метрики з confusion matrix
cm_true = []
cm_pred = []
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        cm_true.extend([i] * cm[i, j])
        cm_pred.extend([j] * cm[i, j])

precision_macro = precision_score(cm_true, cm_pred, average='macro', zero_division=0)
recall_macro = recall_score(cm_true, cm_pred, average='macro', zero_division=0)
f1_macro = f1_score(cm_true, cm_pred, average='macro', zero_division=0)

# Heatmap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def save_confusion_matrix_heatmap(cm, title, filename, class_names=None):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"   💾 Heatmap збережено: {filename}")

class_names = None
try:
    from sklearn.preprocessing import LabelEncoder
    import pandas as pd
    df_tmp = pd.read_csv(m_sDataFile)
    labels_tmp = df_tmp.iloc[:, -1].values
    if not str(labels_tmp[0]).isdigit():
        le = LabelEncoder()
        le.fit(labels_tmp)
        class_names = list(le.classes_)
    del df_tmp, labels_tmp
except:
    pass

cm_base = os.path.splitext(os.path.basename(m_sDataFile))[0]
os.makedirs('Python/CM', exist_ok=True)
save_confusion_matrix_heatmap(cm, f'Window-based CM - {m_sModel} (KFold)', f'Python/CM/cm_kfold_window_{cm_base}.png', class_names)

# Trace aggregation heatmaps та метрики
if bTraceAggregation:
    # MEAN
    cm_true_mean = []
    cm_pred_mean = []
    for i in range(cm_mean_total.shape[0]):
        for j in range(cm_mean_total.shape[1]):
            cm_true_mean.extend([i] * cm_mean_total[i, j])
            cm_pred_mean.extend([j] * cm_mean_total[i, j])
    precision_mean = precision_score(cm_true_mean, cm_pred_mean, average='macro', zero_division=0)
    recall_mean = recall_score(cm_true_mean, cm_pred_mean, average='macro', zero_division=0)
    f1_mean = f1_score(cm_true_mean, cm_pred_mean, average='macro', zero_division=0)

    save_confusion_matrix_heatmap(cm_mean_total, f'MEAN Aggregation CM - {m_sModel} (KFold)', f'Python/CM/cm_kfold_mean_{cm_base}.png', class_names)

    # MAX
    cm_true_max = []
    cm_pred_max = []
    for i in range(cm_max_total.shape[0]):
        for j in range(cm_max_total.shape[1]):
            cm_true_max.extend([i] * cm_max_total[i, j])
            cm_pred_max.extend([j] * cm_max_total[i, j])
    precision_max = precision_score(cm_true_max, cm_pred_max, average='macro', zero_division=0)
    recall_max = recall_score(cm_true_max, cm_pred_max, average='macro', zero_division=0)
    f1_max = f1_score(cm_true_max, cm_pred_max, average='macro', zero_division=0)

    save_confusion_matrix_heatmap(cm_max_total, f'MAX Aggregation CM - {m_sModel} (KFold)', f'Python/CM/cm_kfold_max_{cm_base}.png', class_names)

# Збереження метрик у CSV
import csv as csv_module
os.makedirs('Python/Metrics', exist_ok=True)
metrics_csv = f'Python/Metrics/{cm_base}_kfold_metrics.csv'

fieldnames = ['Dataset', 'Type', 'Accuracy', 'AccStd', 'Precision', 'Recall', 'F1', 'FPR', 'FNR', 'Loss', 'Train_Time', 'Test_Time']
with open(metrics_csv, 'w', newline='') as f:
    writer = csv_module.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    # Window-based
    writer.writerow({
        'Dataset': cm_base, 'Type': 'Window',
        'Accuracy': round(dAcc, 2), 'AccStd': round(dAccStd, 2),
        'Precision': round(precision_macro, 4), 'Recall': round(recall_macro, 4),
        'F1': round(f1_macro, 4), 'FPR': round(1.0 - precision_macro, 4),
        'FNR': round(1.0 - recall_macro, 4), 'Loss': round(dLoss, 4),
        'Train_Time': round(dTimeTrain, 2), 'Test_Time': round(dTimeTest, 2)
    })
    if bTraceAggregation:
        dAccMean = np.mean(arrAccMean)
        dAccMeanStd = np.std(arrAccMean)
        dAccMax = np.mean(arrAccMax)
        dAccMaxStd = np.std(arrAccMax)
        writer.writerow({
            'Dataset': cm_base, 'Type': 'MEAN',
            'Accuracy': round(dAccMean, 2), 'AccStd': round(dAccMeanStd, 2),
            'Precision': round(precision_mean, 4), 'Recall': round(recall_mean, 4),
            'F1': round(f1_mean, 4), 'FPR': round(1.0 - precision_mean, 4),
            'FNR': round(1.0 - recall_mean, 4), 'Loss': round(dLoss, 4),
            'Train_Time': round(dTimeTrain, 2), 'Test_Time': round(dTimeTest, 2)
        })
        writer.writerow({
            'Dataset': cm_base, 'Type': 'MAX',
            'Accuracy': round(dAccMax, 2), 'AccStd': round(dAccMaxStd, 2),
            'Precision': round(precision_max, 4), 'Recall': round(recall_max, 4),
            'F1': round(f1_max, 4), 'FPR': round(1.0 - precision_max, 4),
            'FNR': round(1.0 - recall_max, 4), 'Loss': round(dLoss, 4),
            'Train_Time': round(dTimeTrain, 2), 'Test_Time': round(dTimeTest, 2)
        })

# Фінальний вивід
print("\n" + "="*70)
print("📈 ФІНАЛЬНІ РЕЗУЛЬТАТИ (EMBEDDING + StratifiedKFold)")
print("="*70)
print(f"All time       : {tmExec:.2f}s")
print(f"Training time  : {dTimeTrain:.2f}s")
print(f"Testing time   : {dTimeTest:.2f}s")
print(f"\nWindow-based:")
print(f"  Accuracy:  {dAcc:.2f}% (+/- {dAccStd:.2f})")
print(f"  Precision: {precision_macro:.4f} (macro)")
print(f"  Recall:    {recall_macro:.4f} (macro)")
print(f"  F1-Score:  {f1_macro:.4f} (macro)")
if bTraceAggregation:
    print(f"\nMEAN Aggregation:")
    print(f"  Accuracy:  {dAccMean:.2f}% (+/- {dAccMeanStd:.2f})")
    print(f"  Precision: {precision_mean:.4f} (macro)")
    print(f"  Recall:    {recall_mean:.4f} (macro)")
    print(f"  F1-Score:  {f1_mean:.4f} (macro)")
    print(f"\nMAX Aggregation:")
    print(f"  Accuracy:  {dAccMax:.2f}% (+/- {dAccMaxStd:.2f})")
    print(f"  Precision: {precision_max:.4f} (macro)")
    print(f"  Recall:    {recall_max:.4f} (macro)")
    print(f"  F1-Score:  {f1_max:.4f} (macro)")
print(f"\nLoss: {dLoss:.4f}")
print("="*70)
print(f"\n💾 Метрики збережено у {metrics_csv}")
print("✅ Тренування завершено!")
