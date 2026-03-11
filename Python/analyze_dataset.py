#!/usr/bin/env python3
"""
Аналіз розподілу лейблів у всіх датасетах
"""

import numpy as np
import os
import glob

def analyze_labels(dataset_path):
    """Аналізує розподіл лейблів у датасеті"""
    
    try:
        # Читаємо датасет
        data = np.genfromtxt(dataset_path, delimiter=",", dtype=str)
        
        # Останній стовпець - це лейбли
        labels_raw = data[:, -1]
        
        # Конвертуємо текстові лейбли в числові
        if labels_raw[0] in ['Malware', 'Benign']:
            # Бінарна класифікація
            labels = np.where(labels_raw == 'Malware', 1, 0)
        elif labels_raw[0] in ['AdWare', 'Backdoor', 'Trojan', 'Virus', 'Worm']:
            # Багатокласова класифікація malware
            unique_classes = np.unique(labels_raw)
            labels = labels_raw  # Залишаємо як текст для багатокласової
        else:
            # Числові лейбли
            labels = labels_raw.astype(int)
        
        # Підраховуємо розподіл
        if isinstance(labels[0], str):
            # Багатокласова класифікація
            unique_labels, counts = np.unique(labels, return_counts=True)
            total_samples = len(labels)
            
            result = {
                'total': total_samples,
                'classes': len(unique_labels),
                'distribution': {},
                'type': 'multiclass'
            }
            
            for label, count in zip(unique_labels, counts):
                percentage = (count / total_samples) * 100
                result['distribution'][label] = {
                    'count': count,
                    'percentage': percentage
                }
        else:
            # Бінарна класифікація
            unique_labels, counts = np.unique(labels, return_counts=True)
            total_samples = len(labels)
            
            result = {
                'total': total_samples,
                'classes': len(unique_labels),
                'distribution': {},
                'type': 'binary'
            }
            
            for label, count in zip(unique_labels, counts):
                percentage = (count / total_samples) * 100
                class_name = "Malware" if label == 1 else "Benign"
                result['distribution'][class_name] = {
                    'count': count,
                    'percentage': percentage
                }
        
        return result
        
    except Exception as e:
        return {'error': str(e)}

def analyze_all_datasets():
    """Аналізує всі датасети"""
    
    print("📊 Аналіз всіх датасетів AWSCTD\n")
    
    # Шукаємо всі CSV файли
    dataset_dirs = [
        "../CSV/MalwarePlusClean/*.csv",
        "../CSV/AllMalware/*.csv", 
        "../CSV/AllMalwarePlusClean/*.csv",
        "../CSV/MalwarePlusClean2/*.csv",
        "../CSV/AllMalware2/*.csv",
        "../CSV/AllMalwarePlusClean2/*.csv"
    ]
    
    all_results = {}
    
    for pattern in dataset_dirs:
        files = glob.glob(pattern)
        if files:
            dir_name = pattern.split('/')[-2]
            all_results[dir_name] = {}
            
            print(f"📁 {dir_name}:")
            
            for file_path in sorted(files):
                filename = os.path.basename(file_path)
                result = analyze_labels(file_path)
                
                if 'error' in result:
                    print(f"   ❌ {filename}: {result['error']}")
                else:
                    all_results[dir_name][filename] = result
                    
                    # Форматуємо вивід
                    total = result['total']
                    dist = result['distribution']
                    
                    if result.get('type') == 'multiclass':
                        # Багатокласова класифікація
                        classes_str = " | ".join([f"{cls}: {dist[cls]['percentage']:.1f}%" for cls in sorted(dist.keys())])
                        print(f"   📄 {filename}: {total:,} зразків | {classes_str}")
                    elif 'Malware' in dist and 'Benign' in dist:
                        # Бінарна класифікація
                        mal_pct = dist['Malware']['percentage']
                        ben_pct = dist['Benign']['percentage']
                        print(f"   📄 {filename}: {total:,} зразків | Malware: {mal_pct:.1f}% | Benign: {ben_pct:.1f}%")
                    elif 'Malware' in dist:
                        print(f"   📄 {filename}: {total:,} зразків | Тільки Malware (100%)")
                    else:
                        print(f"   📄 {filename}: {total:,} зразків | Невідомий розподіл")
            
            print()
    
    return all_results

if __name__ == "__main__":
    results = analyze_all_datasets()
    
    print("📋 Підсумок:")
    print("   • MalwarePlusClean - змішані датасети (malware + benign)")
    print("   • AllMalware - тільки malware зразки")
    print("   • Числа в назвах (010, 020, ..., 1000) - кількість системних викликів")
    print("   • Суфікси _2, _5 - різні версії датасетів")