#!/usr/bin/env python3
"""
Тестовий скрипт для перевірки перемикача CPU/GPU
"""

import os
import sys
import tensorflow as tf
from configparser import ConfigParser

def test_device_config():
    """Тестує конфігурацію пристрою"""
    
    # Читаємо конфігурацію
    config = ConfigParser()
    config.read('config.ini')
    sDevice = config.get('MAIN', 'sDevice', fallback='auto')
    
    print(f"🔧 Конфігурація пристрою: {sDevice}")
    
    # Налаштовуємо пристрій
    if sDevice.lower() == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print("🖥️  Примусово використовуємо CPU")
    elif sDevice.lower() == 'gpu':
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if not gpus:
            print("❌ GPU не знайдено, але в конфігурації вказано GPU")
            return False
        print("🚀 Примусово використовуємо GPU")
    else:
        print("🔄 Автоматичний вибір пристрою")
    
    # Перевіряємо доступні пристрої
    gpus = tf.config.experimental.list_physical_devices('GPU')
    cpus = tf.config.experimental.list_physical_devices('CPU')
    
    print(f"\n📊 Доступні пристрої:")
    print(f"   CPU: {len(cpus)} пристроїв")
    print(f"   GPU: {len(gpus)} пристроїв")
    
    # Тестуємо TensorFlow
    print(f"\n🧪 Тестування TensorFlow:")
    with tf.device('/CPU:0' if sDevice.lower() == 'cpu' else tf.test.gpu_device_name() if gpus else '/CPU:0'):
        # Простий тест
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
        
        device_name = c.device
        print(f"   Операція виконана на: {device_name}")
        print(f"   Результат: {c.numpy()}")
    
    return True

if __name__ == "__main__":
    print("🔍 Тестування конфігурації пристрою\n")
    
    if test_device_config():
        print("\n✅ Тест пройшов успішно!")
    else:
        print("\n❌ Тест не пройшов!")
        sys.exit(1)