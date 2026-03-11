# Windowed Training - Quick Start

## 🚀 Швидкий старт

### 1. Тестування датасету
```bash
cd Python
python test_windowed_data.py
```

### 2. Ініціалізація БД
```bash
python init_windowed_db.py
```

### 3. Тренування
```bash
python AWSCTD_windowed.py ../CSV/all_syscalls_w1000_s250.csv AWSCTD-CNN-S
```

## 📊 Що відбувається

1. **Читання даних**: Завантажується датасет з trace_id у першій колонці
2. **Тренування**: Модель навчається на окремих вікнах
3. **Агрегація**: Predictions по вікнам одного трейсу усереднюються (MEAN)
4. **Метрики**: Виводяться Window-based та Trace-based accuracy

## 🎯 Ключові відмінності

| Аспект | Стандартний AWSCTD.py | Windowed AWSCTD_windowed.py |
|--------|----------------------|----------------------------|
| Вхідні дані | Один трейс = один зразок | Один трейс = багато вікон |
| Тренування | На цілих трейсах | На окремих вікнах |
| Prediction | Один prediction на трейс | Багато predictions → агрегація |
| Метрики | Accuracy | Window Acc + Trace Acc |

## 📈 Очікувані результати

```
Window-based Accuracy: ~85%  (точність на вікнах)
Trace-based Accuracy:  ~87%  (точність на трейсах після агрегації)
```

Trace-based accuracy зазвичай вища, оскільки агрегація зменшує шум.

## 🔧 Налаштування

Використовує стандартний `config.ini`. Рекомендовані параметри:

```ini
nEpochs = 20
nBatchSize = 64
nKFolds = 5
fLearningRate = 0.001
bGradientClipping = true
```

## 📁 Нові файли

- `AWSCTD_windowed.py` - тренінг скрипт
- `AWSCTDReadDataWindowed.py` - читання даних
- `init_windowed_db.py` - ініціалізація БД
- `test_windowed_data.py` - тестування датасету
- `WINDOWED_README.md` - повна документація
