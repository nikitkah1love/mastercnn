#!/usr/bin/env python3
"""
Швидкий тест для перевірки чи потягне система великий датасет
"""

import sys
import os
import psutil
from memory_monitor import estimate_dataset_memory, MemoryMonitor

def check_system_resources():
    """Перевірити системні ресурси"""
    
    # Пам'ять
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    
    print("🖥️  Системні ресурси:")
    print(f"   RAM: {memory.total/1024/1024/1024:.1f} GB (доступно: {memory.available/1024/1024/1024:.1f} GB)")
    print(f"   Swap: {swap.total/1024/1024/1024:.1f} GB (вільно: {swap.free/1024/1024/1024:.1f} GB)")
    print(f"   Використання RAM: {memory.percent:.1f}%")
    
    # CPU
    cpu_count = psutil.cpu_count()
    print(f"   CPU cores: {cpu_count}")
    
    return memory, swap

def recommend_settings(estimated_memory_mb, available_memory_mb):
    """Рекомендувати налаштування на основі доступної пам'яті"""
    
    print("\n💡 Рекомендації:")
    
    if estimated_memory_mb > available_memory_mb * 0.8:
        print("❌ Датасет занадто великий для доступної пам'яті!")
        print("   Рекомендації:")
        print("   1. Використовуйте AWSCTD_optimized.py замість AWSCTD.py")
        print("   2. Збільште swap пам'ять:")
        print("      sudo fallocate -l 8G /swapfile")
        print("      sudo chmod 600 /swapfile")
        print("      sudo mkswap /swapfile")
        print("      sudo swapon /swapfile")
        print("   3. Зменште розмір датасету")
        print("   4. Використайте машину з більшою RAM")
        return False
    
    elif estimated_memory_mb > available_memory_mb * 0.6:
        print("⚠️  Датасет великий, але можливо обробити з оптимізаціями:")
        print("   1. Використовуйте AWSCTD_optimized.py")
        print("   2. Використайте config_large.ini")
        print("   3. Закрийте інші програми")
        print("   4. Моніторьте пам'ять під час тренування")
        return True
    
    else:
        print("✅ Датасет має нормальний розмір для вашої системи")
        print("   Можете використовувати звичайний AWSCTD.py")
        return True

def main():
    if len(sys.argv) != 2:
        print("Використання: python test_memory.py path/to/dataset.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    if not os.path.exists(csv_file):
        print(f"❌ Файл не знайдено: {csv_file}")
        sys.exit(1)
    
    print("🧪 Тест пам'яті для AWSCTD")
    print("=" * 40)
    
    # Перевірити системні ресурси
    memory, swap = check_system_resources()
    available_mb = memory.available / 1024 / 1024
    
    # Оцінити пам'ять датасету
    print("\n📊 Аналіз датасету:")
    estimated_mb = estimate_dataset_memory(csv_file)
    
    # Рекомендації
    can_process = recommend_settings(estimated_mb, available_mb)
    
    print("\n🚀 Команди для запуску:")
    if estimated_mb > 4000:
        print("   # Для великих датасетів:")
        print(f"   python AWSCTD_optimized.py {csv_file} FCN")
        print("   # Або скопіюйте config_large.ini в config.ini:")
        print("   cp config_large.ini config.ini")
    else:
        print("   # Стандартний запуск:")
        print(f"   python AWSCTD.py {csv_file} FCN")
    
    print("\n📈 Моніторинг пам'яті:")
    print("   python memory_monitor.py &  # Запустити моніторинг")
    print("   tail -f training_memory.log  # Дивитися логи")
    
    return can_process

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)