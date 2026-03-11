#!/usr/bin/env python3
"""
Ініціалізація таблиці для windowed результатів
"""

import sqlite3

def init_db():
    """Створює таблицю results_windowed якщо її немає"""
    
    print("🗄️  Ініціалізація бази даних для windowed результатів...")
    
    con = sqlite3.connect('results.db')
    cur = con.cursor()
    
    # Створюємо таблицю
    cur.execute("""
        CREATE TABLE IF NOT EXISTS results_windowed (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            File TEXT,
            ParamCount INTEGER,
            ClassCount INTEGER,
            Epochs INTEGER,
            BatchSize INTEGER,
            Time REAL,
            WindowAcc REAL,
            TraceAcc REAL,
            Loss REAL,
            TimeTrain REAL,
            TimeTest REAL,
            Comment TEXT,
            WindowAccStd REAL,
            TraceAccStd REAL,
            LossStd REAL,
            ExecutionTime TEXT,
            Config TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    con.commit()
    
    # Перевіряємо чи таблиця створена
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='results_windowed'")
    result = cur.fetchone()
    
    if result:
        print("✅ Таблиця results_windowed створена успішно")
        
        # Показуємо кількість записів
        cur.execute("SELECT COUNT(*) FROM results_windowed")
        count = cur.fetchone()[0]
        print(f"   Поточна кількість записів: {count}")
    else:
        print("❌ Помилка створення таблиці")
    
    con.close()

if __name__ == "__main__":
    init_db()
