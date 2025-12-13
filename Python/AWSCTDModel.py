import sys
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import ntpath

if len(sys.argv) != 2:
    print("Parameters example: AWSCTDModel.py file_to_data.csv")
    quit()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

m_sDataFile = sys.argv[1]

# fix random seed
np.random.seed(0)
tf.random.set_seed(0)

import AWSCTDReadData
import AWSCTDCreateModel

m_nParametersCount = 0
m_nClassCount = 0
m_nWordCount = 0
bCategorical = False
Xtr, Ytr, m_nParametersCount, m_nClassCount, m_nWordCount = AWSCTDReadData.ReadDataImpl(m_sDataFile, bCategorical)

m_sWorkingDir = os.getcwd()
m_sWorkingDir = m_sWorkingDir + '/'
print(m_sWorkingDir)

def RunModel(sModel):
    fileModel = m_sWorkingDir + 'Model/MODEL_' + sModel + '_'
    fileModel += ntpath.basename(m_sDataFile)
    fileModel += '.svg'

    model = AWSCTDCreateModel.CreateModelImpl(sModel, m_nWordCount, m_nClassCount, m_nParametersCount, bCategorical)
    tf.keras.utils.plot_model(model, to_file=fileModel, show_shapes=False, show_layer_names=False)

RunModel("FCN")
RunModel("LSTM-FCN")
RunModel("GRU-FCN")
RunModel("AWSCTD-CNN-S")
RunModel("AWSCTD-CNN-LSTM")
RunModel("AWSCTD-CNN-GRU")
RunModel("AWSCTD-CNN-D")