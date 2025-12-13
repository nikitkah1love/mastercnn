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

### Перевірка пам'яті (рекомендовано)
```bash
# Перевірити чи потягне система ваш датасет
python test_memory.py ../CSV/AllMalware/1000_5.csv
```

### Для невеликих датасетів (< 4GB в пам'яті)
```bash
python AWSCTD.py ../CSV/MalwarePlusClean/010_2.csv FCN
```

### Для великих датасетів (> 4GB в пам'яті)
```bash
# Використовуйте оптимізовану версію
python AWSCTD_optimized.py ../CSV/AllMalware/1000_5.csv FCN

# Або скопіюйте конфіг для великих датасетів
cp config_large.ini config.ini
python AWSCTD.py ../CSV/AllMalware/1000_5.csv FCN
```

### Моніторинг пам'яті
```bash
# Запустити моніторинг в фоні
python memory_monitor.py &

# Дивитися логи в реальному часі
tail -f training_memory.log
```

### Генерація моделей
```bash
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

## 🚀 Оптимізації для великих датасетів

### Проблема "Killed"
Якщо процес завершується з "Killed", це означає нестачу пам'яті (OOM). Рішення:

1. **Використовуйте оптимізовану версію**:
   ```bash
   python AWSCTD_optimized.py dataset.csv MODEL
   ```

2. **Збільште swap пам'ять** (Linux/WSL):
   ```bash
   sudo fallocate -l 8G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

3. **Налаштуйте конфігурацію**:
   ```bash
   cp config_large.ini config.ini
   ```

### Оптимізації в коді

- **Пакетна обробка**: Великі датасети обробляються по частинах
- **Sparse encoding**: Економія пам'яті для великих словників
- **Адаптивні параметри**: Автоматичне налаштування batch_size та epochs
- **Моніторинг пам'яті**: Відстеження використання RAM в реальному часі
- **Early stopping**: Зупинка при досягненні ліміту пам'яті

### Системні вимоги для різних датасетів

| Датасет | Розмір файлу | RAM потрібно | Рекомендації |
|---------|-------------|--------------|--------------|
| 010_2.csv | ~1MB | 2GB | Стандартний AWSCTD.py |
| 100_2.csv | ~10MB | 4GB | Стандартний або оптимізований |
| 1000_5.csv | ~100MB | 8GB+ | Тільки AWSCTD_optimized.py |

### Налаштування TensorFlow для економії пам'яті

```python
# Автоматично включено в оптимізованій версії
tf.config.experimental.set_memory_growth(gpu, True)
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
```