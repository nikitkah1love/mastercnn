import sqlite3

con = sqlite3.connect('results.db')
cur = con.cursor()

# Додаємо нові колонки для window-based метрик
columns_to_add = [
    "WindowPrecision REAL",
    "WindowRecall REAL",
    "WindowF1 REAL",
    "WindowFPR REAL",
    "WindowFNR REAL",
    "WindowPrecisionStd REAL",
    "WindowRecallStd REAL",
    "WindowF1Std REAL",
    "WindowFPRStd REAL",
    "WindowFNRStd REAL",
    "PrecisionMeanStd REAL",
    "RecallMeanStd REAL",
    "F1MeanStd REAL",
    "FPRMeanStd REAL",
    "FNRMeanStd REAL",
    "PrecisionMaxStd REAL",
    "RecallMaxStd REAL",
    "F1MaxStd REAL",
    "FPRMaxStd REAL",
    "FNRMaxStd REAL"
]

for column in columns_to_add:
    col_name = column.split()[0]
    try:
        cur.execute(f"ALTER TABLE results_windowed_v2 ADD COLUMN {column}")
        print(f"✅ Додано колонку {col_name}")
    except Exception as e:
        print(f"⚠️  {col_name}: {e}")

con.commit()
con.close()

print("\n✅ Міграція завершена!")
print("\nТепер БД містить всі метрики:")
print("- Window-based: Accuracy, Precision, Recall, F1, FPR, FNR (+ Std)")
print("- MEAN Aggregation: Accuracy, Precision, Recall, F1, FPR, FNR (+ Std)")
print("- MAX Aggregation: Accuracy, Precision, Recall, F1, FPR, FNR (+ Std)")
print("- Timing: All time, Training time, Testing time")
print("- Config: Epochs, BatchSize, Patience, KFolds, Categorical, Device (в Config)")

