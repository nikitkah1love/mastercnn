#!/usr/bin/env python3
"""
Візуалізація архітектури моделі AWSCTD-CNN-S
"""

import sys
import os
sys.path.insert(1, 'Utils')

import tensorflow as tf
import AWSCTDCreateModel

def visualize_awsctd_cnn_s():
    """Візуалізує архітектуру моделі AWSCTD-CNN-S"""
    
    print("🏗️  Архітектура моделі AWSCTD-CNN-S\n")
    
    # Параметри для прикладу (як у 100_2.csv)
    nWordCount = 1          # One-hot encoding dimension
    nClassCount = 2         # Binary classification  
    nParametersCount = 100  # Number of system calls (features)
    bCategorical = False    # Binary classification
    
    # Створюємо модель
    model = AWSCTDCreateModel.CreateCNNS(
        nWordCount, nClassCount, nParametersCount, bCategorical
    )
    
    print("📊 Детальна архітектура:")
    print("=" * 60)
    
    # Детальний опис кожного шару
    for i, layer in enumerate(model.layers):
        print(f"Шар {i+1}: {layer.__class__.__name__}")
        print(f"   Назва: {layer.name}")
        
        if hasattr(layer, 'input_shape'):
            print(f"   Вхід: {layer.input_shape}")
        if hasattr(layer, 'output_shape'):
            print(f"   Вихід: {layer.output_shape}")
            
        # Специфічні параметри для різних типів шарів
        if isinstance(layer, tf.keras.layers.Conv1D):
            print(f"   Фільтри: {layer.filters}")
            print(f"   Розмір ядра: {layer.kernel_size}")
            print(f"   Padding: {layer.padding}")
            print(f"   Активація: {layer.activation.__name__}")
            
        elif isinstance(layer, tf.keras.layers.Dense):
            print(f"   Нейрони: {layer.units}")
            print(f"   Активація: {layer.activation.__name__}")
            
        elif isinstance(layer, tf.keras.layers.GlobalMaxPooling1D):
            print(f"   Тип: Global Max Pooling")
            
        # Кількість параметрів
        if hasattr(layer, 'count_params'):
            params = layer.count_params()
            print(f"   Параметри: {params:,}")
            
        print()
    
    print("📈 Загальна інформація:")
    model.summary()
    
    print(f"\n🔢 Розрахунок параметрів:")
    print(f"Conv1D шар:")
    print(f"   Ваги: kernel_size × input_channels × filters = 6 × 1 × 256 = 1,536")
    print(f"   Bias: filters = 256")
    print(f"   Всього Conv1D: 1,536 + 256 = 1,792")
    
    print(f"\nDense шар:")
    print(f"   Ваги: input_features × output_units = 256 × 1 = 256")
    print(f"   Bias: output_units = 1")
    print(f"   Всього Dense: 256 + 1 = 257")
    
    print(f"\nЗагальна кількість параметрів: 1,792 + 257 = 2,049")
    
    return model

def explain_architecture():
    """Пояснює архітектуру моделі"""
    
    print("\n🧠 Пояснення архітектури AWSCTD-CNN-S:")
    print("=" * 50)
    
    print("1️⃣  INPUT LAYER")
    print("   📥 Вхідні дані: (batch_size, 100, 1)")
    print("   📝 100 системних викликів, кожен як one-hot вектор розміру 1")
    print()
    
    print("2️⃣  CONV1D LAYER")
    print("   🔍 Фільтри: 256")
    print("   📏 Розмір ядра: 6 (sliding window)")
    print("   🔄 Padding: 'same' (зберігає розмір)")
    print("   ⚡ Активація: tanh")
    print("   📤 Вихід: (batch_size, 100, 256)")
    print("   💡 Призначення: Виявляє локальні патерни в послідовностях системних викликів")
    print()
    
    print("3️⃣  GLOBAL MAX POOLING 1D")
    print("   📉 Зменшує розмірність: (100, 256) → (256,)")
    print("   🎯 Вибирає максимальне значення по кожному фільтру")
    print("   💡 Призначення: Виділяє найважливіші ознаки незалежно від позиції")
    print()
    
    print("4️⃣  DENSE OUTPUT LAYER")
    print("   🧮 Нейрони: 1 (binary classification)")
    print("   ⚡ Активація: sigmoid")
    print("   📤 Вихід: (batch_size, 1)")
    print("   💡 Призначення: Фінальна класифікація (malware vs benign)")
    print()
    
    print("🎯 ОСОБЛИВОСТІ АРХІТЕКТУРИ:")
    print("   • Проста та ефективна для послідовностей")
    print("   • Sliding window розміру 6 для локальних патернів")
    print("   • Global Max Pooling для інваріантності до позиції")
    print("   • Мінімальна кількість параметрів (~2K)")
    print("   • Швидке навчання та інференс")

if __name__ == "__main__":
    model = visualize_awsctd_cnn_s()
    explain_architecture()