#!/usr/bin/env python3
"""
Ініціалізація таблиці для windowed результатів v2 (з детальними метриками)
"""

import sqlite3

def init_db():
    """Створює таблицю results_windowed_v2 якщо її немає"""
    
    print("🗄️  Ініціалізація бази даних для windowed результатів v2...")
    
    con = sqlite3.connect('results.db')
    cur = con.cursor()
    
    # Створюємо таблицю
    cur.execute("""
        CREATE TABLE IF NOT EXISTS results_windowed_v2 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            File TEXT,
            ParamCount INTEGER,
            ClassCount INTEGER,
            Epochs INTEGER,
            BatchSize INTEGER,
            Time REAL,
            WindowAcc REAL,
            TraceAccMean REAL,
            TraceAccMax REAL,
            Loss REAL,
            TimeTrain REAL,
            TimeTest REAL,
            Comment TEXT,
            WindowAccStd REAL,
            TraceAccMeanStd REAL,
            TraceAccMaxStd REAL,
            LossStd REAL,
            PrecisionMean REAL,
            RecallMean REAL,
            F1Mean REAL,
            FPRMean REAL,
            FNRMean REAL,
            PrecisionMax REAL,
            RecallMax REAL,
            F1Max REAL,
            FPRMax REAL,
            FNRMax REAL,
            ExecutionTime TEXT,
            Config TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    con.commit()
    
    # Перевіряємо чи таблиця створена
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='results_windowed_v2'")
    result = cur.fetchone()
    
    if result:
        print("✅ Таблиця results_windowed_v2 створена успішно")
        
        # Показуємо кількість записів
        cur.execute("SELECT COUNT(*) FROM results_windowed_v2")
        count = cur.fetchone()[0]
        print(f"   Поточна кількість записів: {count}")
    else:
        print("❌ Помилка створення таблиці")
    
    con.close()

if __name__ == "__main__":
    init_db()
