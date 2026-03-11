import tensorflow as tf
import numpy as np
import pandas as pd
import gc

def ReadDataEmbeddingImpl(sDataFile, bCategorical):
    """
    Читає датасет для використання з Embedding layer
    Повертає syscall IDs (не one-hot encoded) для економії пам'яті
    """
    
    print("📖 Читання датасету для Embedding...")
    
    # Читаємо CSV з pandas
    df = pd.read_csv(sDataFile)
    
    print(f"   Загальна кількість вікон: {len(df):,}")
    print(f"   Колонок: {len(df.columns)}")
    
    # Перша колонка - trace_id
    trace_ids = df.iloc[:, 0].values
    
    # Остання колонка - label
    labels = df.iloc[:, -1].values
    
    # Середні колонки - системні виклики (пропускаємо trace_id та label)
    syscalls = df.iloc[:, 1:-1].values
    
    nParametersCount = syscalls.shape[1]
    print(f"   Кількість системних викликів у вікні: {nParametersCount}")
    
    # Конвертуємо в int (залишаємо як syscall IDs, БЕЗ one-hot encoding)
    Xtr = syscalls.astype(dtype=np.int32)
    
    # Обробка лейблів
    if labels[0] in ['True', 'False', 'Malware', 'Benign']:
        # Текстові лейбли
        ytr = np.where((labels == 'True') | (labels == 'Malware'), 1, 0)
    else:
        # Спробуємо конвертувати в int, якщо не вийде - це текстові класи
        try:
            ytr = labels.astype(int)
        except (ValueError, TypeError):
            # Текстові класи (Trojan, Worm, тощо) - конвертуємо в числа
            from sklearn.preprocessing import LabelEncoder
            label_encoder = LabelEncoder()
            ytr = label_encoder.fit_transform(labels)
            print(f"   Знайдено текстові класи: {list(label_encoder.classes_)}")
    
    # Підрахунок унікальних класів та trace_id
    unique_traces = np.unique(trace_ids)
    nClassCount = len(np.unique(ytr))
    
    print(f"   Унікальних трейсів: {len(unique_traces):,}")
    print(f"   Класів: {nClassCount}")
    
    # Підрахунок розподілу по трейсам
    trace_labels = {}
    for trace_id, label in zip(trace_ids, ytr):
        if trace_id not in trace_labels:
            trace_labels[trace_id] = label
    
    trace_label_counts = np.bincount([trace_labels[tid] for tid in unique_traces])
    for i, count in enumerate(trace_label_counts):
        print(f"   Трейсів класу {i}: {count:,}")
    
    # Визначаємо розмір словника (максимальний syscall ID + 1)
    nMaxSysCallValue = int(np.amax(Xtr))
    nWordCount = nMaxSysCallValue + 1
    print(f"   Максимальне значення syscall: {nMaxSysCallValue}")
    print(f"   Розмір словника: {nWordCount}")
    
    # НЕ робимо one-hot encoding! Залишаємо як є
    print("✅ Дані готові (без one-hot encoding, для Embedding layer)")
    
    # Обробка лейблів
    if bCategorical:
        # Для мультикласової класифікації просто робимо to_categorical
        Ytr = tf.keras.utils.to_categorical(ytr, num_classes=nClassCount).astype(np.int16)
    else:
        # Для бінарної класифікації
        from sklearn.preprocessing import LabelBinarizer
        encoder = LabelBinarizer()
        Ytr = encoder.fit_transform(ytr)
    
    print(f"   X shape: {Xtr.shape}")
    print(f"   Y shape: {Ytr.shape}")
    
    # Очищення пам'яті
    del df
    del syscalls
    del labels
    gc.collect()
    
    return Xtr, Ytr, trace_ids, nParametersCount, nClassCount, nWordCount
