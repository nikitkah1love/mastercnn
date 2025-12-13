#!/bin/bash
# Скрипт для активації віртуального середовища
source venv/bin/activate
echo "Віртуальне середовище активовано!"
echo "Python version: $(python --version)"
echo "TensorFlow version: $(python -c 'import tensorflow as tf; print(tf.__version__)')"
echo ""
echo "Для запуску моделі використовуйте:"
echo "python AWSCTD.py data.csv MODEL_NAME"
echo ""
echo "Доступні моделі:"
echo "- FCN"
echo "- LSTM-FCN" 
echo "- GRU-FCN"
echo "- AWSCTD-CNN-S"
echo "- AWSCTD-CNN-LSTM"
echo "- AWSCTD-CNN-GRU"
echo "- AWSCTD-CNN-D"