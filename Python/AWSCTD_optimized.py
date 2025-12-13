import sys
import os
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, 'Utils')

if len(sys.argv) != 3:
    print("Parameters example: AWSCTD_optimized.py file_to_data.csv CNN")
    quit()

import tensorflow as tf
import numpy as np
import gc
from configparser import ConfigParser
from memory_monitor import MemoryMonitor

# Configure TensorFlow for memory optimization
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.random.set_seed(0)

# Configure GPU if available
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Ініціалізація моніторингу пам'яті
memory_monitor = MemoryMonitor("training_memory.log")
memory_monitor.log_memory("Script started")

m_sDataFile = sys.argv[1]
m_sModel = sys.argv[2]

# fix random seed for reproducibility
np.random.seed(0)

# Оцінка пам'яті датасету
print("🔍 Аналіз датасету...")
from memory_monitor import estimate_dataset_memory
estimated_memory = estimate_dataset_memory(m_sDataFile)

# Вибір стратегії обробки даних
if estimated_memory > 4000:  # Більше 4GB
    print("⚡ Використовуємо оптимізовану обробку даних...")
    import AWSCTDReadDataOptimized as AWSCTDReadData
    use_optimized = True
else:
    print("✅ Використовуємо стандартну обробку даних...")
    import AWSCTDReadData
    use_optimized = False

import AWSCTDCreateModel
import AWSCTDClearSesion

m_sWorkingDir = os.getcwd()
m_sWorkingDir = m_sWorkingDir + '/'
print(m_sWorkingDir)

config = ConfigParser()
config.read(m_sWorkingDir + 'config.ini')
nEpochs = config.getint('MAIN', 'nEpochs')
nBatchSize = config.getint('MAIN', 'nBatchSize')
nPatience = config.getint('MAIN', 'nPatience')
nKFolds = config.getint('MAIN', 'nKFolds')
bCategorical = config.getboolean('MAIN', 'bCategorical')

# Адаптивні параметри для великих датасетів
if estimated_memory > 4000:
    print("🔧 Адаптація параметрів для великого датасету...")
    nBatchSize = max(nBatchSize * 2, 64)  # Збільшуємо batch size
    nEpochs = min(nEpochs, 50)  # Зменшуємо кількість епох
    print(f"   Новий batch size: {nBatchSize}")
    print(f"   Нова кількість епох: {nEpochs}")

with open(m_sWorkingDir + 'config.ini', "r") as fIniFile:
    sConfig = fIniFile.read()
print("Config file:")
print(sConfig)

memory_monitor.log_memory("Config loaded")

# Читання даних з моніторингом пам'яті
print("📖 Читання та обробка даних...")
memory_monitor.start_monitoring(interval=10)

try:
    if use_optimized:
        # Використовуємо оптимізовану версію з контролем пам'яті
        Xtr, Ytr, m_nParametersCount, m_nClassCount, m_nWordCount = AWSCTDReadData.ReadDataImpl(
            m_sDataFile, bCategorical, batch_processing=True, max_memory_mb=6000
        )
    else:
        Xtr, Ytr, m_nParametersCount, m_nClassCount, m_nWordCount = AWSCTDReadData.ReadDataImpl(
            m_sDataFile, bCategorical
        )
    
    memory_monitor.log_memory("Data loaded successfully")
    print(f"📊 Розмір даних: X={Xtr.shape}, Y={Ytr.shape}")
    
except MemoryError as e:
    memory_monitor.log_memory(f"Memory error during data loading: {e}")
    print("❌ Помилка пам'яті при завантаженні даних!")
    print("💡 Спробуйте:")
    print("   1. Зменшити розмір датасету")
    print("   2. Збільшити swap пам'ять")
    print("   3. Використати машину з більшою RAM")
    sys.exit(1)

print(Ytr)
gc.collect()
memory_monitor.log_memory("After initial GC")

import math

def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.8
    epochs_drop = 100.0
    lrate = initial_lrate * 1.0 / (1.0 + (0.0000005 * epoch * 100))
    if lrate < 0.0001:
        return 0.0001
    return lrate

# Callback function to achieve early stopping
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
sMonitor = 'accuracy'

if bCategorical:
    sMonitor = 'categorical_accuracy'

# Додаткові callbacks для оптимізації
callbacks_list = [
    EarlyStopping(monitor=sMonitor, patience=nPatience, verbose=1, restore_best_weights=True),
    ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=0.0001, verbose=1)
]

arrAcc = []
arrLoss = []
arrMae = []
arrTimeFit = []
arrTimeTest = []
arrTimePredict = []

from sklearn.model_selection import KFold
kfold = KFold(n_splits=nKFolds, shuffle=True)

from time import gmtime, strftime
sTime = strftime("%Y-%m-%d %H:%M:%S", gmtime())
import time
start = time.time()
nFoldNumber = 1

nAllSize = 0

import AWSCTDPlotAcc
model_history = []

# For Confusion Matrix
import AWSCTDPlotCM
from sklearn.metrics import confusion_matrix
cm = np.zeros((m_nClassCount, m_nClassCount), dtype=int)

# For ROC
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from collections import defaultdict

plt.rcParams['svg.fonttype'] = 'none'

tprs = {}
aucs = {}
EER = {}

if m_nClassCount == 5:
    tprs = {0: [], 1: [], 2: [], 3: [], 4: []}
    aucs = {0: [], 1: [], 2: [], 3: [], 4: []}
    EER = {0: [], 1: [], 2: [], 3: [], 4: []}
elif m_nClassCount == 6:
    tprs = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
    aucs = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
    EER = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
elif m_nClassCount == 2:
    tprs = {0: [], 1: []}
    aucs = {0: [], 1: []}
    EER = {0: [], 1: []}

mean_fpr = np.linspace(0, 1, 100)

memory_monitor.log_memory("Starting K-Fold training")

for train, test in kfold.split(Xtr, Ytr):
    print("KFold number: " + str(nFoldNumber))
    nFoldNumber += 1
    
    memory_monitor.log_memory(f"Starting fold {nFoldNumber-1}")
    
    # Перевірка пам'яті перед створенням моделі
    if not memory_monitor.check_memory_limit(7000):  # 7GB ліміт
        print("⚠️  Досягнуто ліміт пам'яті, зупиняємо тренування")
        break
    
    # Create model
    model = AWSCTDCreateModel.CreateModelImpl(m_sModel, m_nWordCount, m_nClassCount, m_nParametersCount, bCategorical)
    
    startFit = time.time()
    # Train
    try:
        history = model.fit([Xtr[train]], Ytr[train], epochs=nEpochs, batch_size=nBatchSize, 
                          callbacks=callbacks_list, verbose=1, validation_split=0.1)
        memory_monitor.log_memory(f"Fold {nFoldNumber-1} training completed")
    except Exception as e:
        memory_monitor.log_memory(f"Training error in fold {nFoldNumber-1}: {e}")
        print(f"❌ Помилка тренування: {e}")
        break
    
    endFit = time.time()
    model_history.append(history)
    tmExecFit = endFit - startFit
    
    startTest = time.time()
    scores = model.evaluate([Xtr[test]], Ytr[test], verbose=0)
    endTest = time.time()
    
    startPredict = time.time()
    y_pred = model.predict([Xtr[test]], verbose=0)
    endPredict = time.time()
    
    nAllSize = nAllSize + len(Xtr[test])
    
    tmExecTest = endTest - startTest
    arrTimeFit.append(tmExecFit)
    arrTimeTest.append(tmExecTest)
    
    tmExecPredict = endPredict - startPredict
    arrTimePredict.append(tmExecPredict)
    print(model.metrics_names)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    arrAcc.append(scores[1] * 100)
    arrLoss.append(scores[0])

    # To get accuracy comparison with metrics
    nPredCorr = 0
    dAccuracy = 0.0
    
    if bCategorical:
        all_count = len(y_pred)
        for i in range(all_count):
            if np.argmax(Ytr[test][i]) == np.argmax(y_pred[i]):
                nPredCorr += 1
        dAccuracy = float(nPredCorr) / float(all_count)
    else:
        y_pred_class = (y_pred > 0.5)
        y_pred_class = y_pred_class.astype(int)
        all_count = len(y_pred_class)
        for i in range(all_count):
            if Ytr[test][i] == y_pred_class[i]:
                nPredCorr += 1
        dAccuracy = float(nPredCorr) / float(all_count)

    dAccuracy = dAccuracy * 100
    print("Accuracy (Sanity Check): %.2f%%" % dAccuracy)

    # ROC calculations (тільки якщо не занадто великий датасет)
    if len(Ytr[test]) < 50000 and bCategorical:
        for x in range(m_nClassCount):
            plt.figure(x)
            fpr, tpr, thresholds = roc_curve(Ytr[test][:, x], y_pred[:, x])
            fnr = 1 - tpr
            eer_ = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
            EER[x].append(eer_)
            tprs[x].append(np.interp(mean_fpr, fpr, tpr))
            tprs[x][-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs[x].append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.5, label='ROC fold %d (AUC = %0.2f)' % (nFoldNumber, roc_auc))

    # Confusion Matrix calculations
    if m_nClassCount != 2:
        y_pred = np.argmax(y_pred, axis=1)
        cm += confusion_matrix(Ytr[test].argmax(axis=1), y_pred)
    else:
        if bCategorical:
            y_pred = np.argmax(y_pred, axis=1)
            cm += confusion_matrix(Ytr[test].argmax(axis=1), y_pred)
        else:
            y_pred = (y_pred > 0.5)
            y_pred = y_pred.astype(int)
            cm += confusion_matrix(Ytr[test], y_pred)

    try:
        del model
    except:
        pass
    
    AWSCTDClearSesion.reset_keras()
    gc.collect()
    memory_monitor.log_memory(f"Fold {nFoldNumber-1} cleanup completed")

memory_monitor.stop_monitoring()

end = time.time()

tmExec = end - start
dTimeTrain = np.mean(arrTimeFit)
dTimeTest = np.mean(arrTimeTest)

dTimePredSum = np.sum(arrTimePredict)
dTimePredForOneSample = dTimePredSum / nAllSize if nAllSize > 0 else 0
dTimeTest = np.mean(arrTimeTest)

print("All time            : %.7f" % tmExec)
print("Training time       : %.7f" % dTimeTrain)
print("Testing time        : %.7f" % dTimeTest)
print("Predicting time All : %.7f" % dTimePredSum)
print("Predicting time One : %.7f" % dTimePredForOneSample)
print(" Acc: %.2f%% (+/- %.2f)" % (np.mean(arrAcc), np.std(arrAcc)))

# Збереження результатів (тільки якщо тренування завершилось успішно)
if len(arrAcc) > 0:
    model = AWSCTDCreateModel.CreateModelImpl(m_sModel, m_nWordCount, m_nClassCount, m_nParametersCount, bCategorical)

    sModel = str(model.to_json())
    dAcc = np.mean(arrAcc)
    dAccStd = np.std(arrAcc)
    dLoss = np.mean(arrLoss)
    dLossStd = np.std(arrLoss)

    dAcc1 = arrAcc[0] if len(arrAcc) > 0 else 0
    dAcc2 = arrAcc[1] if len(arrAcc) > 1 else 0
    dAcc3 = arrAcc[2] if len(arrAcc) > 2 else 0
    dAcc4 = arrAcc[3] if len(arrAcc) > 3 else 0
    dAcc5 = arrAcc[4] if len(arrAcc) > 4 else 0

    import sqlite3
    con = sqlite3.connect('results.db')
    sTestTag = m_sModel
    result = (m_sDataFile, m_nParametersCount, m_nClassCount, nEpochs, nBatchSize, sModel, tmExec, dAcc, dLoss, dTimeTrain, dTimeTest, sTestTag, dAccStd, dLossStd, sTime, dTimePredForOneSample, dAcc1, dAcc2, dAcc3, dAcc4, dAcc5, sConfig)
    sql = """INSERT INTO results 
             (File, ParamCount, ClassCount, Epochs, BatchSize, Model, Time, Acc, Loss, TimeTrain, TimeTest, Comment, AccStd, LossStd, ExecutionTime, PredictingOneTime, Acc1, Acc2, Acc3, Acc4, Acc5, Config)
             VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
          """
    cur = con.cursor()
    cur.execute(sql, result)
    con.commit()

    # Генерація графіків (тільки якщо не занадто великий датасет)
    if len(model_history) > 0:
        try:
            AWSCTDPlotAcc.plot_acc_loss(model_history, m_sModel, m_sDataFile, bCategorical, m_sWorkingDir)
            AWSCTDPlotCM.plot_cm(cm, m_sModel, m_nClassCount, m_sDataFile, m_sWorkingDir)
        except Exception as e:
            print(f"⚠️  Помилка генерації графіків: {e}")

print("✅ Тренування завершено!")
memory_monitor.log_memory("Script completed")