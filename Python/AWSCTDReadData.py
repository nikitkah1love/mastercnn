import tensorflow as tf
import numpy as np
import gc

def ReadDataImpl(sDataFile, bCategorical):
    # Used when all data is in numbers
    # dbTrain = np.loadtxt(sDataFile, delimiter=",", dtype=np.int16)
    dbTrain = np.genfromtxt(sDataFile, delimiter=",", dtype=str)
    # print(dbTrain)
    dbTrainShape = dbTrain.shape
    # Used on index data
    # nParametersCount = dbTrainShape[1] - 1
    nParametersCount = dbTrainShape[1] - 1
    # print('nParametersCount = ' + str(nParametersCount) + '\n')
    
    # split into input (X) and output (Y) variables
    xtr = dbTrain[:, 0:nParametersCount]
    # xtr = dbTrain[:,600:nParametersCount] <-jei darom nuo kazkur
    Xtr = xtr.astype(dtype=np.int16)
    ytr = dbTrain[:, nParametersCount]
    arrClassNames = np.unique(ytr)
    print(arrClassNames)
    # arrClassNames = ["AdWare", "Trojan", "WebToolbar", "Downloader", "DangerousObject"]
    
    nClassCount = np.unique(ytr).size
    # print("nClassCount = " + str(nClassCount) + "\n")
    
    nMaxSysCallValue = int(np.amax(Xtr))
    nWordCount = nMaxSysCallValue + 1
    
    # Hot One encoding
    Xtr = tf.keras.utils.to_categorical(Xtr).astype(np.int8)
    
    # from sklearn.preprocessing import LabelEncoder
    # encoder = LabelEncoder()
    # Ytr = encoder.fit_transform(ytr)
    
    from sklearn.preprocessing import LabelBinarizer
    encoder = LabelBinarizer()
    Ytr = encoder.fit_transform(ytr).astype(np.int16)
    
    del dbTrain
    del encoder
    del xtr
    del ytr
    gc.collect()
    
    return Xtr, Ytr, nParametersCount, nClassCount, nWordCount