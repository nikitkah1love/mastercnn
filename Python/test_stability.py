#!/usr/bin/env python3
"""
Тестовий скрипт для діагностики проблем з NaN loss
"""

import os
import sys
import numpy as np
import tensorflow as tf
from configparser import ConfigParser

def test_model_stability():
    """Тестує стабільність моделі на синтетичних даних"""
    
    print("🔍 Діагностика стабільності моделі\n")
    
    # Читаємо конфігурацію
    config = ConfigParser()
    config.read('config.ini')
    
    sDevice = config.get('MAIN', 'sDevice', fallback='auto')
    fLearningRate = config.getfloat('MAIN', 'fLearningRate', fallback=0.001)
    bGradientClipping = config.getboolean('MAIN', 'bGradientClipping', fallback=True)
    
    print(f"📋 Конфігурація:")
    print(f"   Пристрій: {sDevice}")
    print(f"   Learning Rate: {fLearningRate}")
    print(f"   Gradient Clipping: {bGradientClipping}")
    
    # Налаштовуємо пристрій
    if sDevice.lower() == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print("🖥️  Використовуємо CPU")
    elif sDevice.lower() == 'gpu':
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if not gpus:
            print("❌ GPU не знайдено!")
            return False
        print("🚀 Використовуємо GPU")
    
    # Створюємо синтетичні дані (як у реальному датасеті)
    print("\n🧪 Створення тестових даних...")
    nSamples = 1000
    nFeatures = 100  # Як у 100_2.csv
    nClasses = 2
    
    # Генеруємо дані схожі на one-hot encoded системні виклики
    X = np.random.randint(0, 2, size=(nSamples, nFeatures, 1)).astype(np.float32)
    y = np.random.randint(0, nClasses, size=(nSamples,)).astype(np.float32)
    
    print(f"   Форма X: {X.shape}")
    print(f"   Форма y: {y.shape}")
    print(f"   Діапазон X: [{X.min():.3f}, {X.max():.3f}]")
    print(f"   Діапазон y: [{y.min():.0f}, {y.max():.0f}]")
    
    # Перевіряємо на NaN/Inf
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        print("❌ Знайдено NaN/Inf в X!")
        return False
    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        print("❌ Знайдено NaN/Inf в y!")
        return False
    
    print("✅ Дані валідні")
    
    # Створюємо просту модель для тестування
    print("\n🏗️  Створення тестової моделі...")
    
    sys.path.insert(1, 'Utils')
    import AWSCTDCreateModel
    
    try:
        model = AWSCTDCreateModel.CreateModelImpl(
            "AWSCTD-CNN-S", 1, nClasses, nFeatures, False, fLearningRate, bGradientClipping
        )
        print("✅ Модель створено успішно")
    except Exception as e:
        print(f"❌ Помилка створення моделі: {e}")
        return False
    
    # Тестуємо forward pass
    print("\n🔄 Тестування forward pass...")
    try:
        predictions = model.predict(X[:10], verbose=0)
        print(f"   Форма predictions: {predictions.shape}")
        print(f"   Діапазон predictions: [{predictions.min():.6f}, {predictions.max():.6f}]")
        
        if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
            print("❌ NaN/Inf в predictions!")
            return False
        print("✅ Forward pass успішний")
    except Exception as e:
        print(f"❌ Помилка forward pass: {e}")
        return False
    
    # Тестуємо один крок навчання
    print("\n📚 Тестування одного кроку навчання...")
    try:
        history = model.fit(X[:100], y[:100], epochs=1, batch_size=32, verbose=0)
        loss = history.history['loss'][0]
        accuracy = history.history['accuracy'][0]
        
        print(f"   Loss: {loss:.6f}")
        print(f"   Accuracy: {accuracy:.6f}")
        
        if np.isnan(loss) or np.isinf(loss):
            print("❌ NaN/Inf в loss!")
            print("💡 Рекомендації:")
            print("   - Зменшіть learning rate (fLearningRate = 0.0001)")
            print("   - Увімкніть gradient clipping (bGradientClipping = true)")
            print("   - Спробуйте CPU режим (sDevice = cpu)")
            return False
        
        print("✅ Навчання стабільне")
    except Exception as e:
        print(f"❌ Помилка навчання: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("🧪 Тест стабільності моделі для діагностики NaN loss\n")
    
    if test_model_stability():
        print("\n✅ Всі тести пройшли успішно! Модель стабільна.")
    else:
        print("\n❌ Виявлено проблеми зі стабільністю!")
        print("\n🔧 Спробуйте:")
        print("   cp config_gpu_stable.ini config.ini")
        print("   python test_stability.py")
        sys.exit(1)