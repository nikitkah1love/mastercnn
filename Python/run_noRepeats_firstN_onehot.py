#!/usr/bin/env python3
"""
Скрипт для запуску тренінгів на всіх no_repeats_first_n датасетах з one-hot encoding
"""
import subprocess
import re
import csv
import os

datasets = ['n10', 'n20', 'n40', 'n80', 'n100', 'n200', 'n400', 'n600', 'n800', 'n1000']
results = []

print("🚀 Запуск експериментів на no_repeats_first_n (one-hot)\n")

for dataset in datasets:
    train_file = f"CSV/output_dataset_no_repeats_first_n/dataset_{dataset}_train.csv"
    test_file = f"CSV/output_dataset_no_repeats_first_n/dataset_{dataset}_test.csv"
    
    print(f"{'='*70}")
    print(f"📊 Датасет: {dataset}")
    
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print(f"⚠️  Файли не знайдено, пропускаємо")
        continue
    
    cmd = [
        'python', 'Python/AWSCTD_train_test_split.py',
        train_file, test_file,
        'AWSCTD-CNN-S-PADDED'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        output = result.stdout + result.stderr
        
        if 'killed' in output.lower() or result.returncode == 137:
            print(f"   ❌ Killed (OOM)")
            results.append({'Dataset': dataset, 'Accuracy': 'OOM', 'Precision': 'OOM', 'Recall': 'OOM', 'F1': 'OOM', 'FPR': 'OOM', 'FNR': 'OOM', 'Loss': 'OOM', 'Train_Time': 'OOM', 'Test_Time': 'OOM'})
            continue
        
        accuracy_match = re.search(r'Accuracy:\s+([\d.]+)%', output)
        precision_match = re.search(r'Precision:\s+([\d.]+)', output)
        recall_match = re.search(r'Recall:\s+([\d.]+)', output)
        f1_match = re.search(r'F1-Score:\s+([\d.]+)', output)
        loss_match = re.search(r'Loss:\s+([\d.]+)', output)
        train_time_match = re.search(r'Training time\s+:\s+([\d.]+)s', output)
        test_time_match = re.search(r'Testing time\s+:\s+([\d.]+)s', output)
        
        if accuracy_match and precision_match and recall_match and f1_match:
            accuracy = float(accuracy_match.group(1))
            precision = float(precision_match.group(1))
            recall = float(recall_match.group(1))
            f1 = float(f1_match.group(1))
            loss = float(loss_match.group(1)) if loss_match else 0.0
            train_time = float(train_time_match.group(1)) if train_time_match else 0.0
            test_time = float(test_time_match.group(1)) if test_time_match else 0.0
            
            results.append({
                'Dataset': dataset,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
                'FPR': 1.0 - precision,
                'FNR': 1.0 - recall,
                'Loss': loss,
                'Train_Time': train_time,
                'Test_Time': test_time
            })
            print(f"✅ Accuracy={accuracy:.2f}%, F1={f1:.4f}")
        else:
            print(f"❌ Не вдалося розпарсити метрики")
            print(f"{output[-500:]}")
    except subprocess.TimeoutExpired:
        print(f"⏱️  Таймаут")
    except Exception as e:
        print(f"❌ Помилка: {e}")

output_file = 'Python/noRepeats_firstN_onehot_results.csv'
if results:
    with open(output_file, 'w', newline='') as f:
        fieldnames = ['Dataset', 'Accuracy', 'Precision', 'Recall', 'F1', 'FPR', 'FNR', 'Loss', 'Train_Time', 'Test_Time']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    
    print(f"\n{'='*120}")
    print(f"{'Dataset':<10} {'Accuracy':<10} {'Precision':<12} {'Recall':<10} {'F1':<10} {'FPR':<10} {'FNR':<10} {'Loss':<10} {'Train(s)':<10} {'Test(s)':<10}")
    print(f"{'='*120}")
    for row in results:
        acc = f"{row['Accuracy']:.2f}" if isinstance(row['Accuracy'], (int, float)) else row['Accuracy']
        prec = f"{row['Precision']:.4f}" if isinstance(row['Precision'], (int, float)) else row['Precision']
        rec = f"{row['Recall']:.4f}" if isinstance(row['Recall'], (int, float)) else row['Recall']
        f1v = f"{row['F1']:.4f}" if isinstance(row['F1'], (int, float)) else row['F1']
        fpr = f"{row['FPR']:.4f}" if isinstance(row['FPR'], (int, float)) else row['FPR']
        fnr = f"{row['FNR']:.4f}" if isinstance(row['FNR'], (int, float)) else row['FNR']
        loss = f"{row['Loss']:.4f}" if isinstance(row['Loss'], (int, float)) else row['Loss']
        tt = f"{row['Train_Time']:.2f}" if isinstance(row['Train_Time'], (int, float)) else row['Train_Time']
        te = f"{row['Test_Time']:.2f}" if isinstance(row['Test_Time'], (int, float)) else row['Test_Time']
        print(f"{row['Dataset']:<10} {acc:<10} {prec:<12} {rec:<10} {f1v:<10} {fpr:<10} {fnr:<10} {loss:<10} {tt:<10} {te:<10}")
    print(f"\n✅ Результати збережено у {output_file}")
