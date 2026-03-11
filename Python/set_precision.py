#!/usr/bin/env python3
"""
Налаштування точності обчислень для TensorFlow
"""

import tensorflow as tf
import os

def set_mixed_precision():
    """Увімкнути mixed precision для прискорення на GPU"""
    
    print("🚀 Налаштування Mixed Precision (Float16/Float32)")
    
    # Перевіряємо чи є GPU з підтримкою Tensor Cores
    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    if not gpus:
        print("❌ GPU не знайдено. Mixed precision працює тільки на GPU.")
        return False
    
    # Увімкнути mixed precision
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    
    print(f"✅ Mixed precision увімкнено:")
    print(f"   Compute dtype: {policy.compute_dtype}")
    print(f"   Variable dtype: {policy.variable_dtype}")
    
    return True

def set_float32():
    """Встановити стандартну точність Float32"""
    
    print("🖥️  Налаштування стандартної точності (Float32)")
    
    policy = tf.keras.mixed_precision.Policy('float32')
    tf.keras.mixed_precision.set_global_policy(policy)
    
    print(f"✅ Float32 встановлено:")
    print(f"   Compute dtype: {policy.compute_dtype}")
    print(f"   Variable dtype: {policy.variable_dtype}")
    
    return True

def set_float64():
    """Встановити високу точність Float64"""
    
    print("🔬 Налаштування високої точності (Float64)")
    
    # TensorFlow за замовчуванням використовує float32
    # Для float64 потрібно явно вказувати dtype в шарах
    tf.keras.backend.set_floatx('float64')
    
    print(f"✅ Float64 встановлено:")
    print(f"   Backend floatx: {tf.keras.backend.floatx()}")
    
    return True

def test_precision_performance():
    """Тестує продуктивність різних точностей"""
    
    print("\n⏱️  Тестування продуктивності різних точностей:")
    
    import time
    import numpy as np
    
    # Тестові дані
    size = 1000
    iterations = 100
    
    precisions = [
        ('Float16', tf.float16),
        ('Float32', tf.float32), 
        ('Float64', tf.float64)
    ]
    
    for name, dtype in precisions:
        print(f"\n   {name}:")
        
        try:
            # Створюємо тестові тензори
            a = tf.random.normal([size, size], dtype=dtype)
            b = tf.random.normal([size, size], dtype=dtype)
            
            # Вимірюємо час
            start_time = time.time()
            
            for _ in range(iterations):
                c = tf.matmul(a, b)
            
            end_time = time.time()
            elapsed = end_time - start_time
            
            print(f"     Час: {elapsed:.3f}s ({elapsed/iterations*1000:.1f}ms на операцію)")
            print(f"     Пам'ять: ~{size*size*dtype.size*3/1024/1024:.1f}MB")
            
        except Exception as e:
            print(f"     ❌ Помилка: {e}")

def create_precision_configs():
    """Створює конфігураційні файли для різних точностей"""
    
    configs = {
        'config_float16.ini': {
            'comment': '# Конфігурація для Mixed Precision (Float16/Float32) - найшвидша на сучасних GPU',
            'learning_rate': '0.001',
            'device': 'gpu'
        },
        'config_float32.ini': {
            'comment': '# Стандартна конфігурація (Float32) - баланс швидкості та точності',
            'learning_rate': '0.001', 
            'device': 'auto'
        },
        'config_float64.ini': {
            'comment': '# Висока точність (Float64) - найточніша, але повільна',
            'learning_rate': '0.0001',  # Менший LR для стабільності
            'device': 'cpu'  # CPU краще для float64
        }
    }
    
    print("\n📝 Створення конфігураційних файлів:")
    
    for filename, config in configs.items():
        content = f"""[MAIN]
{config['comment']}
nEpochs = 50
nBatchSize = 32
nPatience = 5
nKFolds = 5
bCategorical = false
sDevice = {config['device']}
fLearningRate = {config['learning_rate']}
bGradientClipping = true
"""
        
        with open(filename, 'w') as f:
            f.write(content)
        
        print(f"   ✅ {filename}")

if __name__ == "__main__":
    print("🔧 Налаштування точності TensorFlow\n")
    
    # Показуємо поточні налаштування
    print("📊 Поточні налаштування:")
    policy = tf.keras.mixed_precision.global_policy()
    print(f"   Policy: {policy.name}")
    print(f"   Compute dtype: {policy.compute_dtype}")
    print(f"   Variable dtype: {policy.variable_dtype}")
    print(f"   Backend floatx: {tf.keras.backend.floatx()}")
    
    # Тестуємо продуктивність
    test_precision_performance()
    
    # Створюємо конфіги
    create_precision_configs()
    
    print(f"\n💡 Рекомендації:")
    print(f"   • Для GPU з Tensor Cores: використовуйте Mixed Precision (Float16)")
    print(f"   • Для стандартного навчання: Float32 (за замовчуванням)")
    print(f"   • Для наукових обчислень: Float64")
    print(f"   • При проблемах з NaN: Float32 з меншим learning rate")