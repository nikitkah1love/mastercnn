#!/usr/bin/env python3
"""
Оцінка пам'яті для датасету 2020
"""

# Параметри датасету 2020
n_windows = 267204
n_syscalls = 1000
vocab_size = 54

# One-hot encoding: (n_windows, n_syscalls, vocab_size)
# Тип даних: int8 (1 байт)
memory_X = n_windows * n_syscalls * vocab_size * 1  # bytes

# Labels: (n_windows, 1)
# Тип даних: int16 (2 байти) або float32 (4 байти)
memory_Y = n_windows * 1 * 4  # bytes

total_memory = memory_X + memory_Y

print(f"📊 Оцінка пам'яті для датасету 2020:")
print(f"   Вікон: {n_windows:,}")
print(f"   Syscalls на вікно: {n_syscalls}")
print(f"   Розмір словника: {vocab_size}")
print(f"")
print(f"   X (one-hot): {memory_X / 1024 / 1024 / 1024:.2f} GB")
print(f"   Y (labels): {memory_Y / 1024 / 1024:.2f} MB")
print(f"   Разом: {total_memory / 1024 / 1024 / 1024:.2f} GB")
print(f"")
print(f"💡 Порівняння з попереднім датасетом:")
old_memory = 162107 * 1000 * 32 * 1 / 1024 / 1024 / 1024
print(f"   Старий датасет: {old_memory:.2f} GB")
print(f"   Новий датасет: {memory_X / 1024 / 1024 / 1024:.2f} GB")
print(f"   Збільшення: {(memory_X / 1024 / 1024 / 1024) / old_memory:.1f}x")
