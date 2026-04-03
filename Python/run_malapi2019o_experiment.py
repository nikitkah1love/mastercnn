#!/usr/bin/env python3
"""
Тренінг на всіх датасетах malapi2019_o з embedding
"""
import subprocess
import re
import csv
import os

# Автоматично знаходимо всі пари train/test
base_dir = "CSV/malapi2019_o"
datasets = []

for f in sorted(os.listdir(base_dir)):
    if f.startswith('train_windowed') and f != 'train_windowed.csv':
        suffix = f.replace('train_windowed', '')
        test_name = f'test_windowed{suffix}'
        if os.path.exists(os.path.join(base_dir, test_name)):
            label = suffix.replace('.csv', '').lstrip('_') or 'default'
            datasets.append((label, f, test_name))

results = []

print(f"🚀 Тренінг на {len(datasets)} датасетах malapi2019_o (embedding)\n")

for label, train_name, test_name in datasets:
    train_file = f"{base_dir}/{train_name}"
    test_file = f"{base_dir}/{test_name}"
    
    print(f"{'='*50}")
    print(f"📊 {label} ({train_name})")
    
    cmd = [
        'python', 'Python/AWSCTD_embedding.py',
        train_file,
        test_file,
        'AWSCTD-CNN-S-EMBEDDING',
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=36000)
        output = result.stdout + result.stderr
        
        if 'killed' in output.lower() or result.returncode == 137:
            print(f"   ❌ Killed (OOM)")
            results.append({'Dataset': label, 'Accuracy': 'OOM', 'Precision': 'OOM', 'Recall': 'OOM', 'F1': 'OOM', 'FPR': 'OOM', 'FNR': 'OOM', 'Loss': 'OOM', 'Train_Time': 'OOM', 'Test_Time': 'OOM'})
            continue
        
        accuracy_match = re.search(r'Accuracy:\s+([\d.]+)%', output)
        precision_match = re.search(r'Precision:\s+([\d.]+)\s+\(macro\)', output)
        recall_match = re.search(r'Recall:\s+([\d.]+)\s+\(macro\)', output)
        f1_match = re.search(r'F1-Score:\s+([\d.]+)\s+\(macro\)', output)
        loss_match = re.search(r'Loss:\s+([\d.]+)', output)
        train_time_match = re.search(r'Training time\s+:\s+([\d.]+)s', output)
        test_time_match = re.search(r'Testing time\s+:\s+([\d.]+)s', output)
        
        if accuracy_match and precision_match and recall_match and f1_match:
            row = {
                'Dataset': label,
                'Accuracy': float(accuracy_match.group(1)),
                'Precision': float(precision_match.group(1)),
                'Recall': float(recall_match.group(1)),
                'F1': float(f1_match.group(1)),
                'FPR': 1.0 - float(precision_match.group(1)),
                'FNR': 1.0 - float(recall_match.group(1)),
                'Loss': float(loss_match.group(1)) if loss_match else 0.0,
                'Train_Time': float(train_time_match.group(1)) if train_time_match else 0.0,
                'Test_Time': float(test_time_match.group(1)) if test_time_match else 0.0
            }
            results.append(row)
            print(f"   ✅ Accuracy={row['Accuracy']:.2f}%, F1={row['F1']:.4f}, Train={row['Train_Time']:.0f}s")
        else:
            print(f"   ❌ Не вдалося розпарсити метрики")
            print(f"   {output[-300:]}")
            
    except subprocess.TimeoutExpired:
        print(f"   ⏱️  Таймаут")
    except Exception as e:
        print(f"   ❌ Помилка: {e}")

# Збереження
output_file = 'Python/malapi2019o_results.csv'
if results:
    with open(output_file, 'w', newline='') as f:
        fieldnames = ['Dataset', 'Accuracy', 'Precision', 'Recall', 'F1', 'FPR', 'FNR', 'Loss', 'Train_Time', 'Test_Time']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    
    print(f"\n{'='*50}")
    print(f"✅ Результати збережено у {output_file}")
