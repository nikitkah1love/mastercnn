#!/usr/bin/env python3
"""
Grid search по embedding dimensions на фіксованому датасеті (n400)
Тестує dimensions: 4, 8, 16, 32, 64
"""
import subprocess
import re
import csv
import os

# Фіксований датасет (n400 показав найкращі результати)
train_file = "CSV/malapi2019_firstN/dataset_n400_train.csv"
test_file = "CSV/malapi2019_firstN/dataset_n400_test.csv"

# Dimensions для тестування
dimensions = [4, 8, 16, 32, 64]

results = []

print("🔍 Grid Search: Embedding Dimensions")
print(f"📁 Датасет: n400\n")

for dim in dimensions:
    print(f"\n{'='*70}")
    print(f"📐 Embedding Dimension: {dim}")
    print(f"{'='*70}")
    
    cmd = [
        'python', 'Python/AWSCTD_embedding.py',
        train_file,
        test_file,
        'AWSCTD-CNN-S-EMBEDDING',
        '--no-trace-aggregation',
        f'--embedding-dim={dim}'
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600
        )
        
        output = result.stdout + result.stderr
        
        accuracy_match = re.search(r'Accuracy:\s+([\d.]+)%', output)
        precision_match = re.search(r'Precision:\s+([\d.]+)\s+\(macro\)', output)
        recall_match = re.search(r'Recall:\s+([\d.]+)\s+\(macro\)', output)
        f1_match = re.search(r'F1-Score:\s+([\d.]+)\s+\(macro\)', output)
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
            fnr = 1.0 - recall
            fpr = 1.0 - precision
            
            results.append({
                'EmbeddingDim': dim,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
                'FPR': fpr,
                'FNR': fnr,
                'Loss': loss,
                'Train_Time': train_time,
                'Test_Time': test_time
            })
            
            print(f"✅ dim={dim}: Accuracy={accuracy:.2f}%, F1={f1:.4f}, Train={train_time:.2f}s")
        else:
            print(f"❌ Не вдалося розпарсити метрики")
            print(f"Output (last 500 chars): {output[-500:]}")
            
    except subprocess.TimeoutExpired:
        print(f"⏱️  Таймаут для dim={dim}")
    except Exception as e:
        print(f"❌ Помилка: {e}")

# Збереження результатів
output_file = 'Python/embedding_dim_search_results.csv'
if results:
    print(f"\n{'='*70}")
    print(f"💾 Збереження результатів у {output_file}")
    print(f"{'='*70}\n")
    
    with open(output_file, 'w', newline='') as f:
        fieldnames = ['EmbeddingDim', 'Accuracy', 'Precision', 'Recall', 'F1', 'FPR', 'FNR', 'Loss', 'Train_Time', 'Test_Time']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    
    print("\n📊 РЕЗУЛЬТАТИ GRID SEARCH:")
    print(f"{'='*120}")
    print(f"{'Dim':<6} {'Accuracy':<10} {'Precision':<12} {'Recall':<10} {'F1':<10} {'FPR':<10} {'FNR':<10} {'Loss':<10} {'Train(s)':<10} {'Test(s)':<10}")
    print(f"{'='*120}")
    
    for row in results:
        print(f"{row['EmbeddingDim']:<6} {row['Accuracy']:<10.2f} {row['Precision']:<12.4f} {row['Recall']:<10.4f} {row['F1']:<10.4f} {row['FPR']:<10.4f} {row['FNR']:<10.4f} {row['Loss']:<10.4f} {row['Train_Time']:<10.2f} {row['Test_Time']:<10.2f}")
    
    # Знаходимо найкращий dimension по F1
    best = max(results, key=lambda x: x['F1'])
    print(f"{'='*120}")
    print(f"\n🏆 Найкращий: dim={best['EmbeddingDim']}, F1={best['F1']:.4f}, Accuracy={best['Accuracy']:.2f}%")
    print(f"\n✅ Результати збережено у {output_file}")
