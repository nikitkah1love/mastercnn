#!/bin/bash

# AWSCTD WSL Setup Script
# Створює ідентичне віртуальне середовище для WSL Ubuntu

set -e  # Зупинити при помилці

echo "🚀 AWSCTD WSL Setup - Створення віртуального середовища"
echo "=================================================="

# Кольори для виводу
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Перевірка WSL
if ! grep -q Microsoft /proc/version 2>/dev/null; then
    print_warning "Схоже, що це не WSL. Скрипт оптимізований для WSL Ubuntu."
    read -p "Продовжити? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Оновлення системи
print_status "Оновлення системних пакетів..."
sudo apt update && sudo apt upgrade -y

# Встановлення Python та залежностей
print_status "Встановлення Python 3.11 та системних залежностей..."
sudo apt install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    build-essential \
    pkg-config \
    libhdf5-dev \
    libhdf5-serial-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    libatlas-base-dev \
    libopenblas-dev \
    libffi-dev \
    libssl-dev \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    libfreetype6-dev

# Перевірка версії Python
PYTHON_VERSION=$(python3.11 --version 2>&1 | cut -d' ' -f2)
print_success "Python встановлено: $PYTHON_VERSION"

# Перехід до каталогу Python
if [ ! -d "Python" ]; then
    print_error "Каталог Python не знайдено. Запустіть скрипт з кореневого каталогу проекту."
    exit 1
fi

cd Python

# Видалення старого venv якщо існує
if [ -d "venv" ]; then
    print_warning "Видалення існуючого віртуального середовища..."
    rm -rf venv
fi

# Створення віртуального середовища з Python 3.11
print_status "Створення віртуального середовища з Python 3.11..."
python3.11 -m venv venv

# Активація віртуального середовища
print_status "Активація віртуального середовища..."
source venv/bin/activate

# Оновлення pip
print_status "Оновлення pip до останньої версії..."
pip install --upgrade pip setuptools wheel

# Встановлення залежностей
print_status "Встановлення Python залежностей..."
pip install -r requirements.txt

# Перевірка встановлення TensorFlow
print_status "Перевірка встановлення TensorFlow..."
python -c "
import tensorflow as tf
import numpy as np
import matplotlib
import sklearn
print('✅ TensorFlow version:', tf.__version__)
print('✅ NumPy version:', np.__version__)
print('✅ Matplotlib version:', matplotlib.__version__)
print('✅ Scikit-learn version:', sklearn.__version__)
print('✅ GPU доступність:', tf.config.list_physical_devices('GPU'))
"

# Створення скрипту активації для WSL
cat > activate_wsl.sh << 'EOF'
#!/bin/bash
# Скрипт активації віртуального середовища для WSL

cd "$(dirname "$0")"
source venv/bin/activate

echo "🐧 WSL Віртуальне середовище активовано!"
echo "Python version: $(python --version)"
echo "TensorFlow version: $(python -c 'import tensorflow as tf; print(tf.__version__)')"
echo ""
echo "Для запуску моделі використовуйте:"
echo "python AWSCTD.py ../CSV/MalwarePlusClean/010_2.csv MODEL_NAME"
echo ""
echo "Доступні моделі:"
echo "- FCN"
echo "- LSTM-FCN" 
echo "- GRU-FCN"
echo "- AWSCTD-CNN-S"
echo "- AWSCTD-CNN-LSTM"
echo "- AWSCTD-CNN-GRU"
echo "- AWSCTD-CNN-D"
echo ""
echo "Для деактивації: deactivate"
EOF

chmod +x activate_wsl.sh

# Створення тестового скрипту
cat > test_installation.py << 'EOF'
#!/usr/bin/env python3
"""
Тестовий скрипт для перевірки правильності встановлення
"""

import sys
import subprocess

def test_imports():
    """Тестування імпортів"""
    print("🧪 Тестування імпортів...")
    
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__}")
        
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
        
        import matplotlib
        print(f"✅ Matplotlib {matplotlib.__version__}")
        
        import sklearn
        print(f"✅ Scikit-learn {sklearn.__version__}")
        
        # Тест TensorFlow
        print("\n🔧 Тестування TensorFlow...")
        
        # Перевірка GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ Знайдено GPU: {len(gpus)} пристроїв")
            for i, gpu in enumerate(gpus):
                print(f"   GPU {i}: {gpu}")
        else:
            print("ℹ️  GPU не знайдено, використовується CPU")
        
        # Простий тест обчислень
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
        print(f"✅ TensorFlow обчислення працюють: {c.numpy().tolist()}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Помилка імпорту: {e}")
        return False
    except Exception as e:
        print(f"❌ Помилка тестування: {e}")
        return False

def test_model_creation():
    """Тест створення простої моделі"""
    print("\n🏗️  Тестування створення моделі...")
    
    try:
        import tensorflow as tf
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        print("✅ Модель створена та скомпільована успішно")
        
        # Тест з фіктивними даними
        import numpy as np
        X = np.random.random((100, 5))
        y = np.random.randint(2, size=(100, 1))
        
        model.fit(X, y, epochs=1, verbose=0)
        print("✅ Модель навчена на тестових даних")
        
        return True
        
    except Exception as e:
        print(f"❌ Помилка створення моделі: {e}")
        return False

if __name__ == "__main__":
    print("🧪 AWSCTD Installation Test")
    print("=" * 40)
    
    success = True
    
    # Тест імпортів
    if not test_imports():
        success = False
    
    # Тест створення моделі
    if not test_model_creation():
        success = False
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 Всі тести пройдені успішно!")
        print("Середовище готове для роботи з AWSCTD")
    else:
        print("❌ Деякі тести не пройдені")
        print("Перевірте встановлення залежностей")
        sys.exit(1)
EOF

chmod +x test_installation.py

# Запуск тестів
print_status "Запуск тестів встановлення..."
python test_installation.py

print_success "Віртуальне середовище створено успішно!"
print_status "Файли створено:"
echo "  - venv/ (віртуальне середовище)"
echo "  - activate_wsl.sh (скрипт активації)"
echo "  - test_installation.py (тести)"

print_status "Для використання:"
echo "  cd Python"
echo "  source activate_wsl.sh"
echo "  python AWSCTD.py ../CSV/MalwarePlusClean/010_2.csv FCN"

print_success "Готово! 🎉"