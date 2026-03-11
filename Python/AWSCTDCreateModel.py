import tensorflow as tf
from tensorflow.keras.layers import (
    GRU, LSTM, SimpleRNN, Conv1D, Conv2D, Embedding, GlobalMaxPooling1D,
    Dense, Dropout, GlobalMaxPooling2D, GlobalAveragePooling1D,
    AveragePooling1D, AveragePooling2D, MaxPooling2D, MaxPooling1D,
    Flatten, TimeDistributed, Reshape, BatchNormalization,
    Input, ReLU, Concatenate, Add, concatenate
)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import optimizers
from tensorflow.keras.applications.inception_v3 import InceptionV3	
	
def AddLastDenseLayer(merged, bCategorical, nClassCount):
    outputs = Dense(nClassCount if bCategorical else 1, activation='softmax' if bCategorical else 'sigmoid')(merged)
    return outputs

def CreateNewGRU(nWordCount, nClassCount, nParametersCount, bCategorical):
    print("GRU-FCN")
    inputs = Input(shape=(nParametersCount, nWordCount))
    conv1 = Conv1D(filters=128, kernel_size=8, padding='same', activation='tanh', kernel_initializer='glorot_uniform')(inputs)
    BN1 = BatchNormalization(epsilon=1e-6)(conv1)
    relu1 = ReLU()(BN1)
    conv2 = Conv1D(filters=256, kernel_size=5, padding='same', activation='tanh', kernel_initializer='glorot_uniform')(relu1)
    BN2 = BatchNormalization(epsilon=1e-6)(conv2)
    relu2 = ReLU()(BN2)
    conv3 = Conv1D(filters=128, kernel_size=3, padding='same', activation='tanh', kernel_initializer='glorot_uniform')(relu2)
    BN3 = BatchNormalization(epsilon=1e-6)(conv3)
    relu3 = ReLU()(BN3)
    pool1 = GlobalAveragePooling1D()(relu3)
    gru2 = GRU(units=8, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal')(inputs)
    dropout = Dropout(0.5)(gru2)  # Зменшений dropout
    merged = concatenate([pool1, dropout])
    outputs = AddLastDenseLayer(merged, bCategorical, nClassCount)
    model = Model(inputs=[inputs], outputs=outputs)
    
    return model
	
def CreateNew(nWordCount, nClassCount, nParametersCount, bCategorical):
    print("FCN")
    inputs = Input(shape=(nParametersCount, nWordCount))
    conv1 = Conv1D(filters=128, kernel_size=8, padding='same', activation='tanh')(inputs)
    BN1 = BatchNormalization(epsilon=0.01)(conv1)
    relu1 = ReLU()(BN1)
    conv2 = Conv1D(filters=256, kernel_size=5, padding='same', activation='tanh')(relu1)
    BN2 = BatchNormalization(epsilon=0.01)(conv2)
    relu2 = ReLU()(BN2)
    conv3 = Conv1D(filters=128, kernel_size=3, padding='same', activation='tanh')(relu2)
    BN3 = BatchNormalization(epsilon=0.01)(conv3)
    relu3 = ReLU()(BN3)
    pool = GlobalAveragePooling1D()(relu3)
    outputs = AddLastDenseLayer(pool, bCategorical, nClassCount)
    model = Model(inputs=[inputs], outputs=outputs)
    
    return model

def CreateNewLSTM(nWordCount, nClassCount, nParametersCount, bCategorical):
    print("LSTM-FCN")
    inputs = Input(shape=(nParametersCount, nWordCount))
    conv1 = Conv1D(filters=128, kernel_size=8, padding='same', activation='tanh', kernel_initializer='glorot_uniform')(inputs)
    BN1 = BatchNormalization(epsilon=1e-6)(conv1)
    relu1 = ReLU()(BN1)
    conv2 = Conv1D(filters=256, kernel_size=5, padding='same', activation='tanh', kernel_initializer='glorot_uniform')(relu1)
    BN2 = BatchNormalization(epsilon=1e-6)(conv2)
    relu2 = ReLU()(BN2)
    conv3 = Conv1D(filters=128, kernel_size=3, padding='same', activation='tanh', kernel_initializer='glorot_uniform')(relu2)
    BN3 = BatchNormalization(epsilon=1e-6)(conv3)
    relu3 = ReLU()(BN3)
    pool1 = GlobalAveragePooling1D()(relu3)
    lstm3 = LSTM(units=nParametersCount, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal')(inputs)
    dropout = Dropout(0.5)(lstm3)  # Зменшений dropout
    merged = concatenate([pool1, dropout])
    outputs = AddLastDenseLayer(merged, bCategorical, nClassCount)
    model = Model(inputs=[inputs], outputs=outputs)
    
    return model
	
def CreateOLDGRU(nWordCount, nClassCount, nParametersCount, bCategorical):
    print("AWSCTD-CNN-GRU")
    nSlidingWindow = 6
    inputs = Input(shape=(nParametersCount, nWordCount))
    conv1OLD = Conv1D(filters=nParametersCount, kernel_size=nSlidingWindow, padding='same', activation='tanh')(inputs)
    poolOLD = GlobalMaxPooling1D()(conv1OLD)
    gru2 = GRU(units=8)(inputs)  # Replaced CuDNNGRU with GRU
    dropout = Dropout(0.8)(gru2)
    merged = concatenate([poolOLD, dropout])
    outputs = AddLastDenseLayer(merged, bCategorical, nClassCount)
    model = Model(inputs=[inputs], outputs=outputs)
    
    return model

def CreateOLDLSTM(nWordCount, nClassCount, nParametersCount, bCategorical):
    print("AWSCTD-CNN-LSTM")
    nSlidingWindow = 6
    inputs = Input(shape=(nParametersCount, nWordCount))
    conv1OLD = Conv1D(filters=nParametersCount, kernel_size=nSlidingWindow, padding='same', activation='tanh')(inputs)
    poolOLD = GlobalMaxPooling1D()(conv1OLD)
    lstm3 = LSTM(units=nParametersCount)(inputs)  # Replaced CuDNNLSTM with LSTM
    dropout = Dropout(0.8)(lstm3)
    merged = concatenate([poolOLD, dropout])
    outputs = AddLastDenseLayer(merged, bCategorical, nClassCount)
    
    model = Model(inputs=[inputs], outputs=outputs)
    
    return model
	
def CreateCNN(nWordCount, nClassCount, nParametersCount, bCategorical):
    print("AWSCTD-CNN-D")
    nSlidingWindow = 6
    inputs = Input(shape=(nParametersCount, nWordCount))
    CNN = Conv1D(filters=nParametersCount, kernel_size=nSlidingWindow, padding='same', activation='tanh')(inputs)
    poolOLD = GlobalMaxPooling1D()(CNN)
    outputs = AddLastDenseLayer(poolOLD, bCategorical, nClassCount)
    model = Model(inputs=[inputs], outputs=outputs)
    
    return model

def CreateCNNS(nWordCount, nClassCount, nParametersCount, bCategorical):
    print("AWSCTD-CNN-S")
    nSlidingWindow = 6
    inputs = Input(shape=(nParametersCount, nWordCount))
    CNN = Conv1D(filters=256, kernel_size=nSlidingWindow, padding='same', activation='tanh')(inputs)
    poolOLD = GlobalMaxPooling1D()(CNN)
    outputs = AddLastDenseLayer(poolOLD, bCategorical, nClassCount)
    model = Model(inputs=[inputs], outputs=outputs)
    
    return model

def CreateCNNS_Padded(nWordCount, nClassCount, nParametersCount, bCategorical):
    print("AWSCTD-CNN-S-PADDED")
    nSlidingWindow = 6
    inputs = Input(shape=(nParametersCount, nWordCount))
    # use_bias=False щоб padding не давав сигнал
    CNN = Conv1D(filters=256, kernel_size=nSlidingWindow, padding='same', activation='tanh', use_bias=False)(inputs)
    # GlobalAveragePooling краще для padding (0 не впливає на середнє)
    pool = GlobalMaxPooling1D()(CNN)
    outputs = AddLastDenseLayer(pool, bCategorical, nClassCount)
    model = Model(inputs=[inputs], outputs=outputs)
    
    return model

def CreateCNNS_Embedding(nWordCount, nClassCount, nParametersCount, bCategorical, nEmbeddingDim=8):
    print("AWSCTD-CNN-S-EMBEDDING")
    print(f"   Embedding dimension: {nEmbeddingDim}")
    nSlidingWindow = 6
    
    # Input тепер приймає syscall IDs, а не one-hot vectors
    inputs = Input(shape=(nParametersCount,), dtype='int32')
    
    # Embedding layer: nWordCount syscalls → nEmbeddingDim вимірів
    embedding = Embedding(input_dim=nWordCount, output_dim=nEmbeddingDim, input_length=nParametersCount)(inputs)
    
    # use_bias=False для padding
    CNN = Conv1D(filters=256, kernel_size=nSlidingWindow, padding='same', activation='tanh', use_bias=False)(embedding)
    
    # GlobalAveragePooling краще для padding
    pool = GlobalAveragePooling1D()(CNN)
    
    outputs = AddLastDenseLayer(pool, bCategorical, nClassCount)
    model = Model(inputs=[inputs], outputs=outputs)
    
    return model
	
def CreateModelImpl(sModel, nWordCount, nClassCount, nParametersCount, bCategorical, fLearningRate=0.001, bGradientClipping=True):
    print("nWordCount: " + str(nWordCount))
    print("nClassCount: " + str(nClassCount))
    print("nParametersCount: " + str(nParametersCount))
    print(f"Learning Rate: {fLearningRate}")
    print(f"Gradient Clipping: {bGradientClipping}")
    
    model = Sequential()
    if sModel == "AWSCTD-CNN-LSTM":
        model = CreateOLDLSTM(nWordCount, nClassCount, nParametersCount, bCategorical)
    elif sModel == "AWSCTD-CNN-GRU":
        model = CreateOLDGRU(nWordCount, nClassCount, nParametersCount, bCategorical)
    elif sModel == "FCN":
        model = CreateNew(nWordCount, nClassCount, nParametersCount, bCategorical)
    elif sModel == "GRU-FCN":
        model = CreateNewGRU(nWordCount, nClassCount, nParametersCount, bCategorical)
    elif sModel == "LSTM-FCN":
        model = CreateNewLSTM(nWordCount, nClassCount, nParametersCount, bCategorical)
    elif sModel == "AWSCTD-CNN-D":
        model = CreateCNN(nWordCount, nClassCount, nParametersCount, bCategorical)
    elif sModel == "AWSCTD-CNN-S":
        model = CreateCNNS(nWordCount, nClassCount, nParametersCount, bCategorical)
    elif sModel == "AWSCTD-CNN-S-PADDED":
        model = CreateCNNS_Padded(nWordCount, nClassCount, nParametersCount, bCategorical)
    elif sModel == "AWSCTD-CNN-S-EMBEDDING":
        model = CreateCNNS_Embedding(nWordCount, nClassCount, nParametersCount, bCategorical)

    # Стабільний оптимізатор для GPU/CPU сумісності
    optimizer_kwargs = {
        'learning_rate': fLearningRate,
        'beta_1': 0.9,
        'beta_2': 0.999,
        'epsilon': 1e-7,  # Збільшений epsilon для стабільності
    }
    
    if bGradientClipping:
        optimizer_kwargs['clipnorm'] = 1.0  # Gradient clipping для запобігання explosion
    
    optimizer = optimizers.Adam(**optimizer_kwargs)

    if bCategorical:
        model.compile(
            loss='categorical_crossentropy', 
            optimizer=optimizer, 
            metrics=['categorical_accuracy']
        )
    else:
        model.compile(
            loss='binary_crossentropy', 
            optimizer=optimizer, 
            metrics=['accuracy']
        )
        
    return model