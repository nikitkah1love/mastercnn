-- Таблиця для збереження результатів windowed тренування
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
);