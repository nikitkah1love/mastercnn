#!/usr/bin/env python3
"""
Перевірка точності зберігання ваг моделі
"""

import sys
import os
import tensorflow as tf
import numpy as np

sys.path.insert(1, 'Utils')

def check_model_precision():
    """Перевіряє точність ваг моделі"""
    
    print("🔍 Перевірка точності ваг моделі\n")
    
    # Створюємо просту модель
    import AWSCTDCreateModel
    
    model = AWSCTDCreateModel.CreateModelImpl(
        "AWSCTD-CNN-S", 1, 2, 100, False, 0.001, True
    )
    
    print("📊 Інформація про модель:")
    print(f"   Кількість шарів: {len(model.layers)}")
    
    # Перевіряємо точність кожного шару
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'weights') and layer.weights:
            for j, weight in enumerate(layer.weights):
                dtype = weight.dtype
                shape = weight.shape
                print(f"   Шар {i} ({layer.name}) - Ваги {j}: {dtype} {shape}")
                
                # Отримуємо значення ваг
                weight_values = weight.numpy()
                print(f"     Діапазон: [{weight_values.min():.8f}, {weight_values.max():.8f}]")
                print(f"     Середнє: {weight_values.mean():.8f}")
                print(f"     Std: {weight_values.std():.8f}")
    
    # Перевіряємо глобальні налаштування TensorFlow
    print(f"\n🔧 Налаштування TensorFlow:")
    print(f"   Версія TensorFlow: {tf.__version__}")
    print(f"   Стандартний dtype: {tf.keras.backend.floatx()}")
    
    # Перевіряємо mixed precision
    policy = tf.keras.mixed_precision.global_policy()
    print(f"   Mixed precision policy: {policy.name}")
    print(f"   Compute dtype: {policy.compute_dtype}")
    print(f"   Variable dtype: {policy.variable_dtype}")
    
    # Перевіряємо GPU precision
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(f"\n🚀 GPU інформація:")
        for gpu in gpus:
            print(f"   {gpu}")
            
        # Перевіряємо чи увімкнено tensor cores
        try:
            # Створюємо тензор на GPU
            with tf.device('/GPU:0'):
                a = tf.random.normal([1000, 1000], dtype=tf.float32)
                b = tf.random.normal([1000, 1000], dtype=tf.float32)
                c = tf.matmul(a, b)
                print(f"   GPU обчислення: {c.dtype}")
        except:
            print("   GPU недоступний для тестування")
    
    # Тестуємо різні типи даних
    print(f"\n🧪 Тестування різних типів даних:")
    
    dtypes_to_test = [tf.float16, tf.float32, tf.float64]
    
    for dtype in dtypes_to_test:
        try:
            # Створюємо простий шар з певним dtype
            layer = tf.keras.layers.Dense(10, dtype=dtype)
            x = tf.random.normal([1, 100], dtype=dtype)
            y = layer(x)
            
            print(f"   {dtype.name}: ✅")
            print(f"     Input: {x.dtype}, Output: {y.dtype}")
            print(f"     Weights: {layer.weights[0].dtype}")
            
        except Exception as e:
            print(f"   {dtype.name}: ❌ {e}")
    
    # Перевіряємо точність обчислень
    print(f"\n🔬 Тестування точності обчислень:")
    
    # Створюємо тест на втрату точності
    x_f32 = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
    x_f16 = tf.cast(x_f32, tf.float16)
    x_f64 = tf.cast(x_f32, tf.float64)
    
    # Виконуємо операції
    result_f32 = tf.reduce_sum(x_f32 * x_f32)
    result_f16 = tf.reduce_sum(x_f16 * x_f16)
    result_f64 = tf.reduce_sum(x_f64 * x_f64)
    
    print(f"   Float32: {result_f32.numpy():.10f}")
    print(f"   Float16: {result_f16.numpy():.10f}")
    print(f"   Float64: {result_f64.numpy():.10f}")
    
    # Перевіряємо різницю
    diff_16_32 = abs(float(result_f16) - float(result_f32))
    diff_32_64 = abs(float(result_f32) - float(result_f64))
    
    print(f"   Різниця F16-F32: {diff_16_32:.2e}")
    print(f"   Різниця F32-F64: {diff_32_64:.2e}")
    
    return model

def check_saved_model_precision():
    """Перевіряє точність збереженої моделі"""
    
    print(f"\n💾 Перевірка збереження моделі:")
    
    # Створюємо модель
    model = check_model_precision()
    
    # Зберігаємо модель
    model.save('test_model.h5')
    print("   Модель збережено в test_model.h5")
    
    # Завантажуємо модель
    loaded_model = tf.keras.models.load_model('test_model.h5')
    print("   Модель завантажено")
    
    # Порівнюємо ваги
    print("   Порівняння ваг:")
    
    for i, (orig_layer, loaded_layer) in enumerate(zip(model.layers, loaded_model.layers)):
        if hasattr(orig_layer, 'weights') and orig_layer.weights:
            for j, (orig_weight, loaded_weight) in enumerate(zip(orig_layer.weights, loaded_layer.weights)):
                
                orig_values = orig_weight.numpy()
                loaded_values = loaded_weight.numpy()
                
                # Перевіряємо чи ідентичні
                are_identical = np.array_equal(orig_values, loaded_values)
                max_diff = np.max(np.abs(orig_values - loaded_values))
                
                print(f"     Шар {i}, Ваги {j}: {'✅' if are_identical else '⚠️'} (max diff: {max_diff:.2e})")
    
    # Видаляємо тестовий файл
    os.remove('test_model.h5')
    print("   Тестовий файл видалено")

if __name__ == "__main__":
    check_saved_model_precision()