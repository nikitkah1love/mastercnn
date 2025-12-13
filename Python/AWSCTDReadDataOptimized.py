import tensorflow as tf
import numpy as np
import gc
import psutil
import os

def get_memory_usage():
    """Отримати поточне використання пам'яті"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def ReadDataImpl(sDataFile, bCategorical, batch_processing=True, max_memory_mb=4000):
    """
    Оптимізована версія читання даних з контролем пам'яті
    
    Args:
        sDataFile: шлях до CSV файлу
        bCategorical: чи використовувати категоріальне кодування для Y
        batch_processing: чи використовувати пакетну обробку
        max_memory_mb: максимальна пам'ять в MB
    """
    
    print(f"🔍 Початкове використання пам'яті: {get_memory_usage():.1f} MB")
    
    # Спочатку читаємо тільки перші кілька рядків для аналізу
    print("📊 Аналіз структури даних...")
    sample_data = np.genfromtxt(sDataFile, delimiter=",", dtype=str, max_rows=100)
    
    if len(sample_data.shape) == 1:
        sample_data = sample_data.reshape(1, -1)
    
    nParametersCount = sample_data.shape[1] - 1
    
    # Аналіз розміру файлу
    file_size_mb = os.path.getsize(sDataFile) / 1024 / 1024
    print(f"📁 Розмір файлу: {file_size_mb:.1f} MB")
    print(f"🔢 Параметрів на зразок: {nParametersCount}")
    
    # Оцінка кількості рядків
    with open(sDataFile, 'r') as f:
        estimated_rows = sum(1 for _ in f)
    print(f"📈 Приблизна кількість зразків: {estimated_rows}")
    
    # Перевірка чи потрібна оптимізація
    estimated_memory = estimated_rows * nParametersCount * 4 / 1024 / 1024  # MB (float32)
    print(f"💾 Оцінка використання пам'яті: {estimated_memory:.1f} MB")
    
    if estimated_memory > max_memory_mb and batch_processing:
        print("⚠️  Великий датасет! Використовуємо пакетну обробку...")
        return ReadDataImplBatched(sDataFile, bCategorical, max_memory_mb)
    else:
        print("✅ Датасет помірного розміру, використовуємо стандартну обробку...")
        return ReadDataImplStandard(sDataFile, bCategorical)

def ReadDataImplStandard(sDataFile, bCategorical):
    """Стандартна обробка для невеликих датасетів"""
    
    print("📖 Читання даних...")
    dbTrain = np.genfromtxt(sDataFile, delimiter=",", dtype=str)
    print(f"💾 Після читання: {get_memory_usage():.1f} MB")
    
    dbTrainShape = dbTrain.shape
    nParametersCount = dbTrainShape[1] - 1
    
    # Розділення на X та Y
    xtr = dbTrain[:, 0:nParametersCount]
    Xtr = xtr.astype(dtype=np.int16)
    ytr = dbTrain[:, nParametersCount]
    
    del dbTrain, xtr  # Звільнити пам'ять
    gc.collect()
    print(f"💾 Після очищення: {get_memory_usage():.1f} MB")
    
    arrClassNames = np.unique(ytr)
    print(arrClassNames)
    
    nClassCount = np.unique(ytr).size
    nMaxSysCallValue = int(np.amax(Xtr))
    nWordCount = nMaxSysCallValue + 1
    
    print(f"🔢 Max syscall value: {nMaxSysCallValue}, Word count: {nWordCount}")
    
    # Оптимізоване one-hot encoding
    print("🔄 One-hot encoding...")
    original_shape = Xtr.shape
    
    # Використовуємо sparse представлення якщо можливо
    if nWordCount > 1000:  # Якщо словник великий
        print("⚡ Використовуємо sparse encoding для економії пам'яті...")
        # Flatten для to_categorical
        Xtr_flat = Xtr.flatten()
        Xtr_encoded = tf.keras.utils.to_categorical(Xtr_flat, num_classes=nWordCount)
        # Reshape назад
        Xtr = Xtr_encoded.reshape(original_shape[0], original_shape[1], nWordCount).astype(np.int8)
        del Xtr_flat, Xtr_encoded
    else:
        Xtr = tf.keras.utils.to_categorical(Xtr).astype(np.int8)
    
    print(f"💾 Після encoding: {get_memory_usage():.1f} MB")
    print(f"📐 Розмір X: {Xtr.shape}")
    
    # Обробка Y
    from sklearn.preprocessing import LabelBinarizer
    encoder = LabelBinarizer()
    Ytr = encoder.fit_transform(ytr)
    
    if bCategorical:
        Ytr = tf.keras.utils.to_categorical(Ytr).astype(np.int16)
    
    del encoder, ytr
    gc.collect()
    
    print(f"💾 Фінальне використання пам'яті: {get_memory_usage():.1f} MB")
    
    return Xtr, Ytr, nParametersCount, nClassCount, nWordCount

def ReadDataImplBatched(sDataFile, bCategorical, max_memory_mb):
    """Пакетна обробка для великих датасетів"""
    
    print("🔄 Пакетна обробка великого датасету...")
    
    # Читаємо по частинах
    chunk_size = 1000  # Розмір пакету
    chunks_X = []
    chunks_Y = []
    
    # Спочатку визначаємо параметри
    sample_chunk = pd.read_csv(sDataFile, nrows=100)
    nParametersCount = sample_chunk.shape[1] - 1
    
    # Читаємо весь файл по частинах
    import pandas as pd
    
    for chunk in pd.read_csv(sDataFile, chunksize=chunk_size):
        print(f"📦 Обробка пакету розміром {len(chunk)}...")
        
        # Конвертуємо в numpy
        chunk_array = chunk.values
        
        # Розділяємо X та Y
        X_chunk = chunk_array[:, 0:nParametersCount].astype(np.int16)
        Y_chunk = chunk_array[:, nParametersCount]
        
        chunks_X.append(X_chunk)
        chunks_Y.append(Y_chunk)
        
        # Перевіряємо пам'ять
        if get_memory_usage() > max_memory_mb * 0.8:
            print("⚠️  Досягнуто ліміт пам'яті, обробляємо накопичені пакети...")
            break
    
    # Об'єднуємо пакети
    print("🔗 Об'єднання пакетів...")
    Xtr = np.vstack(chunks_X)
    ytr = np.concatenate(chunks_Y)
    
    del chunks_X, chunks_Y
    gc.collect()
    
    # Далі стандартна обробка
    arrClassNames = np.unique(ytr)
    print(arrClassNames)
    
    nClassCount = np.unique(ytr).size
    nMaxSysCallValue = int(np.amax(Xtr))
    nWordCount = nMaxSysCallValue + 1
    
    print(f"🔢 Max syscall value: {nMaxSysCallValue}, Word count: {nWordCount}")
    
    # Оптимізоване encoding
    print("🔄 Batch one-hot encoding...")
    
    # Обробляємо по частинах якщо дуже великий
    if Xtr.shape[0] > 10000:
        encoded_chunks = []
        batch_size = 5000
        
        for i in range(0, Xtr.shape[0], batch_size):
            end_idx = min(i + batch_size, Xtr.shape[0])
            batch = Xtr[i:end_idx]
            
            print(f"🔄 Encoding batch {i//batch_size + 1}...")
            batch_encoded = tf.keras.utils.to_categorical(batch).astype(np.int8)
            encoded_chunks.append(batch_encoded)
            
            # Очищення пам'яті
            if len(encoded_chunks) > 3:  # Зберігаємо не більше 3 пакетів в пам'яті
                break
        
        Xtr = np.vstack(encoded_chunks)
        del encoded_chunks
    else:
        Xtr = tf.keras.utils.to_categorical(Xtr).astype(np.int8)
    
    # Обробка Y
    from sklearn.preprocessing import LabelBinarizer
    encoder = LabelBinarizer()
    Ytr = encoder.fit_transform(ytr)
    
    if bCategorical:
        Ytr = tf.keras.utils.to_categorical(Ytr).astype(np.int16)
    
    del encoder, ytr
    gc.collect()
    
    print(f"💾 Фінальне використання пам'яті: {get_memory_usage():.1f} MB")
    
    return Xtr, Ytr, nParametersCount, nClassCount, nWordCount