#!/usr/bin/env python3
"""
Скрипт для моніторингу використання пам'яті під час тренування
"""

import psutil
import os
import time
import sys
import threading
from datetime import datetime

class MemoryMonitor:
    def __init__(self, log_file="memory_usage.log"):
        self.process = psutil.Process(os.getpid())
        self.log_file = log_file
        self.monitoring = False
        self.max_memory = 0
        
    def get_memory_info(self):
        """Отримати детальну інформацію про пам'ять"""
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()
        
        # Системна пам'ять
        system_memory = psutil.virtual_memory()
        
        return {
            'rss': memory_info.rss / 1024 / 1024,  # MB
            'vms': memory_info.vms / 1024 / 1024,  # MB
            'percent': memory_percent,
            'available_system': system_memory.available / 1024 / 1024,  # MB
            'total_system': system_memory.total / 1024 / 1024,  # MB
            'system_percent': system_memory.percent
        }
    
    def log_memory(self, message=""):
        """Записати використання пам'яті в лог"""
        info = self.get_memory_info()
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        log_entry = f"[{timestamp}] RSS: {info['rss']:.1f}MB, VMS: {info['vms']:.1f}MB, " \
                   f"Process: {info['percent']:.1f}%, System: {info['system_percent']:.1f}% - {message}"
        
        print(log_entry)
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry + "\n")
        
        # Оновити максимум
        if info['rss'] > self.max_memory:
            self.max_memory = info['rss']
    
    def start_monitoring(self, interval=5):
        """Почати моніторинг в окремому потоці"""
        self.monitoring = True
        
        def monitor():
            while self.monitoring:
                self.log_memory("Periodic check")
                time.sleep(interval)
        
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Зупинити моніторинг"""
        self.monitoring = False
        self.log_memory(f"Monitoring stopped. Max memory used: {self.max_memory:.1f}MB")
    
    def check_memory_limit(self, limit_mb=6000):
        """Перевірити чи не перевищено ліміт пам'яті"""
        info = self.get_memory_info()
        if info['rss'] > limit_mb:
            print(f"⚠️  WARNING: Memory usage ({info['rss']:.1f}MB) exceeds limit ({limit_mb}MB)")
            return False
        return True

def estimate_dataset_memory(csv_file):
    """Оцінити скільки пам'яті потребує датасет"""
    import pandas as pd
    
    # Читаємо перші 100 рядків для аналізу
    sample = pd.read_csv(csv_file, nrows=100)
    
    # Оцінюємо загальну кількість рядків
    file_size = os.path.getsize(csv_file)
    sample_size = len(sample.to_csv(index=False).encode())
    estimated_rows = (file_size / sample_size) * 100
    
    # Параметри
    n_features = sample.shape[1] - 1
    max_value = sample.iloc[:, :-1].max().max() if sample.shape[1] > 1 else 100
    
    # Оцінка пам'яті для one-hot encoding
    # Кожен зразок стає матрицею n_features x max_value
    memory_per_sample = n_features * max_value * 1  # 1 byte для int8
    total_memory_mb = (estimated_rows * memory_per_sample) / 1024 / 1024
    
    print(f"📊 Аналіз датасету:")
    print(f"   Файл: {csv_file}")
    print(f"   Розмір файлу: {file_size/1024/1024:.1f} MB")
    print(f"   Оцінка рядків: {estimated_rows:.0f}")
    print(f"   Ознак на зразок: {n_features}")
    print(f"   Максимальне значення: {max_value}")
    print(f"   Оцінка пам'яті після one-hot: {total_memory_mb:.1f} MB")
    
    return total_memory_mb

if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
        estimate_dataset_memory(csv_file)
    else:
        # Демонстрація моніторингу
        monitor = MemoryMonitor()
        monitor.start_monitoring(interval=2)
        
        print("Моніторинг пам'яті запущено. Натисніть Ctrl+C для зупинки.")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            monitor.stop_monitoring()
            print("Моніторинг зупинено.")