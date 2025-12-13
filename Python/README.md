# AWSCTD - Refactored for Modern TensorFlow

Цей проект було рефакторено для роботи з сучасними версіями TensorFlow 2.x та Python 3.

## Встановлення залежностей

```bash
pip install -r requirements.txt
```

## Основні зміни

### 1. Оновлення імпортів
- `keras` → `tensorflow.keras`
- `ConfigParser` → `configparser`
- `CuDNNLSTM/CuDNNGRU` → `LSTM/GRU` (сумісність з CPU/GPU)

### 2. Оновлення метрик
- `acc` → `accuracy`
- Оновлено callback функції для TensorFlow 2.x

### 3. Оновлення синтаксису Python
- `print` statements → `print()` functions
- Оновлено string formatting
- Покращено обробку файлів з `with` statements

### 4. GPU конфігурація
- Автоматичне налаштування GPU memory growth
- Сумісність з CPU та GPU

## Використання

```bash
# Основний скрипт
python AWSCTD.py data.csv CNN

# Генерація моделей
python AWSCTDModel.py data.csv
```

## Підтримувані моделі
- FCN
- LSTM-FCN
- GRU-FCN
- AWSCTD-CNN-S
- AWSCTD-CNN-LSTM
- AWSCTD-CNN-GRU
- AWSCTD-CNN-D

## Структура проекту
```
Python/
├── AWSCTD.py              # Основний скрипт
├── AWSCTDCreateModel.py   # Створення моделей
├── AWSCTDReadData.py      # Читання даних
├── AWSCTDModel.py         # Генерація схем моделей
├── AWSCTDClearSesion.py   # Очищення сесій
├── config.ini             # Конфігурація
├── Utils/                 # Утиліти
│   ├── AWSCTDGenerateImg.py
│   ├── AWSCTDPlotAcc.py
│   └── AWSCTDPlotCM.py
└── requirements.txt       # Залежності
```