#!/bin/bash

# AWSCTD WSL Quick Setup - Без інтерактивних запитань
# Для автоматичного встановлення в CI/CD або скриптах

set -e

echo "🚀 AWSCTD WSL Quick Setup"
echo "========================"

# Встановлення без запитань
export DEBIAN_FRONTEND=noninteractive

# Оновлення системи
sudo apt update -y
sudo apt upgrade -y

# Встановлення Python 3.11 та залежностей
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update -y

sudo apt install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    build-essential \
    pkg-config \
    libhdf5-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    libffi-dev \
    libssl-dev \
    zlib1g-dev

# Перехід до каталогу Python
cd Python

# Створення venv
rm -rf venv
python3.11 -m venv venv
source venv/bin/activate

# Встановлення залежностей
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo "✅ Швидке встановлення завершено!"
echo "Для активації: cd Python && source venv/bin/activate"