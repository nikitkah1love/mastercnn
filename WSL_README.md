# AWSCTD для WSL Ubuntu

Інструкції для налаштування ідентичного віртуального середовища в WSL.

## 🚀 Швидке встановлення

### Варіант 1: Повне встановлення (рекомендовано)
```bash
# Клонувати репозиторій
git clone https://github.com/your-username/awsctd-private.git
cd awsctd-private

# Запустити скрипт встановлення
./setup_wsl.sh
```

### Варіант 2: Швидке встановлення
```bash
# Для автоматичного встановлення без запитань
./quick_setup_wsl.sh
```

## 📋 Системні вимоги

- **OS**: WSL Ubuntu 20.04+ або нативний Ubuntu
- **RAM**: Мінімум 4GB, рекомендовано 8GB+
- **Диск**: 5GB вільного місця
- **Python**: 3.11 (встановлюється автоматично)

## 🔧 Ручне встановлення

Якщо автоматичні скрипти не працюють:

```bash
# 1. Оновити систему
sudo apt update && sudo apt upgrade -y

# 2. Встановити Python 3.11
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update -y
sudo apt install python3.11 python3.11-venv python3.11-dev -y

# 3. Встановити системні залежності
sudo apt install -y \
    build-essential \
    pkg-config \
    libhdf5-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    libffi-dev \
    libssl-dev \
    zlib1g-dev

# 4. Створити віртуальне середовище
cd Python
python3.11 -m venv venv
source venv/bin/activate

# 5. Встановити залежності
pip install --upgrade pip
pip install -r requirements.txt
```

## 🎯 Використання

### Активація середовища
```bash
cd Python
source activate_wsl.sh
```

### Запуск моделей
```bash
# Базова модель FCN
python AWSCTD.py ../CSV/MalwarePlusClean/010_2.csv FCN

# CNN модель
python AWSCTD.py ../CSV/MalwarePlusClean/010_2.csv AWSCTD-CNN-S

# LSTM модель
python AWSCTD.py ../CSV/MalwarePlusClean/010_2.csv LSTM-FCN
```

### Доступні моделі
- `FCN` - Fully Convolutional Network
- `LSTM-FCN` - LSTM + FCN
- `GRU-FCN` - GRU + FCN  
- `AWSCTD-CNN-S` - CNN Static
- `AWSCTD-CNN-LSTM` - CNN + LSTM
- `AWSCTD-CNN-GRU` - CNN + GRU
- `AWSCTD-CNN-D` - CNN Dynamic

## 🧪 Тестування

```bash
# Перевірка встановлення
cd Python
source venv/bin/activate
python test_installation.py
```

## ⚡ Оптимізація для WSL

### GPU підтримка (NVIDIA)
```bash
# Встановити CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-0 -y

# Встановити TensorFlow з GPU
pip install tensorflow[and-cuda]
```

### Налаштування пам'ті
```bash
# Додати до ~/.bashrc для оптимізації пам'яті
echo 'export TF_CPP_MIN_LOG_LEVEL=2' >> ~/.bashrc
echo 'export TF_FORCE_GPU_ALLOW_GROWTH=true' >> ~/.bashrc
```

## 🔍 Діагностика проблем

### Перевірка Python
```bash
python3.11 --version
which python3.11
```

### Перевірка TensorFlow
```bash
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices())"
```

### Логи помилок
```bash
# Запуск з детальними логами
TF_CPP_MIN_LOG_LEVEL=0 python AWSCTD.py data.csv FCN
```

## 📊 Очікувані результати

На WSL Ubuntu з нашими оптимізаціями:
- **Швидкість навчання**: ~10-15 сек/епоха (CPU)
- **Точність**: 87-89% на тестових даних
- **Пам'ять**: ~2-4GB RAM
- **GPU прискорення**: 3-5x швидше (якщо доступно)

## 🆘 Підтримка

При проблемах:
1. Запустіть `python test_installation.py`
2. Перевірте логи помилок
3. Переконайтеся що WSL оновлено до останньої версії

## 📝 Зміни від macOS версії

- Використовується `apt` замість `brew`
- Python 3.11 встановлюється через PPA
- Додано оптимізації для WSL
- Покращено GPU підтримку для NVIDIA