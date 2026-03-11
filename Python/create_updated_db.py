import sqlite3

# Створюємо нову БД
con = sqlite3.connect('results_updated.db')
cur = con.cursor()

# Створюємо таблицю з повною структурою
sql = """
CREATE TABLE IF NOT EXISTS results_windowed_v2 (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    File TEXT,
    ParamCount INTEGER,
    ClassCount INTEGER,
    Epochs INTEGER,
    BatchSize INTEGER,
    Time REAL,
    
    -- Window-based метрики
    WindowAcc REAL,
    WindowAccStd REAL,
    WindowPrecision REAL,
    WindowPrecisionStd REAL,
    WindowRecall REAL,
    WindowRecallStd REAL,
    WindowF1 REAL,
    WindowF1Std REAL,
    WindowFPR REAL,
    WindowFPRStd REAL,
    WindowFNR REAL,
    WindowFNRStd REAL,
    
    -- MEAN Aggregation метрики
    TraceAccMean REAL,
    TraceAccMeanStd REAL,
    PrecisionMean REAL,
    PrecisionMeanStd REAL,
    RecallMean REAL,
    RecallMeanStd REAL,
    F1Mean REAL,
    F1MeanStd REAL,
    FPRMean REAL,
    FPRMeanStd REAL,
    FNRMean REAL,
    FNRMeanStd REAL,
    
    -- MAX Aggregation метрики
    TraceAccMax REAL,
    TraceAccMaxStd REAL,
    PrecisionMax REAL,
    PrecisionMaxStd REAL,
    RecallMax REAL,
    RecallMaxStd REAL,
    F1Max REAL,
    F1MaxStd REAL,
    FPRMax REAL,
    FPRMaxStd REAL,
    FNRMax REAL,
    FNRMaxStd REAL,
    
    -- Loss
    Loss REAL,
    LossStd REAL,
    
    -- Timing
    TimeTrain REAL,
    TimeTest REAL,
    
    -- Metadata
    Comment TEXT,
    ExecutionTime TEXT,
    Config TEXT
)
"""

cur.execute(sql)
con.commit()
con.close()

print("✅ Створено нову БД: results_updated.db")
print("\n📊 Структура таблиці:")
print("   - Window-based: Accuracy, Precision, Recall, F1, FPR, FNR (+ Std)")
print("   - MEAN Aggregation: Accuracy, Precision, Recall, F1, FPR, FNR (+ Std)")
print("   - MAX Aggregation: Accuracy, Precision, Recall, F1, FPR, FNR (+ Std)")
print("   - Loss (+ Std)")
print("   - Timing: All time, Training time, Testing time")
print("   - Config: Epochs, BatchSize + повний config.ini")
