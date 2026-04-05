# AWSCTD research papers

Attack-Caused Windows OS System Calls Traces Dataset

 - Čeponis, Dainius, and Nikolaj Goranin. "Towards a robust method of dataset generation of malicious activity for anomaly-based HIDS training and presentation of AWSCTD dataset." Baltic Journal of Modern Computing 6, no. 3 (2018): 217-234.

 - Goranin, Nikolaj, and Dainius Čeponis. "Investigation of AWSCTD dataset applicability for malware type classification." Security & Future 2, no. 2 (2018): 83-86.

 - Čeponis, Dainius, and Nikolaj Goranin. "Evaluation of Deep Learning Methods Efficiency for Malicious and Benign System Calls Classification on the AWSCTD." Security and Communication Networks 2019 (2019).
 - Čeponis, Dainius, and Nikolaj Goranin. "Investigation of Dual-Flow Deep Learning Models LSTM-FCN and GRU-FCN Efficiency against Single-Flow CNN Models for the Host-Based Intrusion and Malware Detection Task on Univariate Times Series Data." Applied Sciences 10.7 (2020): 2373.
 - Vyšniūnas, Tolvinas, et al. "Risk-Based System-Call Sequence Grouping Method for Malware Intrusion Detection." Electronics 13.1 (2024): 206.

---

## Скрипти тренування

### 1. AWSCTD.py — One-Hot Encoding + StratifiedKFold

Оригінальний скрипт. Приймає один CSV файл, ділить через StratifiedKFold (баланс класів зберігається в кожному фолді).

```bash
# Запуск з папки Python/
python AWSCTD.py ../CSV/malapi2019_firstN/dataset_n1000_train_noheader.csv AWSCTD-CNN-S
```

- CSV без header (перший рядок — дані, не назви колонок)
- Кількість фолдів, епох, batch size — в `config.ini`
- Результати: heatmap в `Python/CM/`, метрики в `Python/Metrics/`, ROC в `Python/ROC/`

### 2. AWSCTD_embedding.py — Embedding + Train/Test Split

Приймає окремі train та test CSV файли. Один прохід (без крос-валідації).

```bash
python Python/AWSCTD_embedding.py CSV/malapi2019_firstN/dataset_n1000_train.csv CSV/malapi2019_firstN/dataset_n1000_test.csv AWSCTD-CNN-S-EMBEDDING --no-trace-aggregation
```

- `--no-trace-aggregation` — вимикає агрегацію по трейсам
- `--embedding-dim=16` — розмір embedding (за замовчуванням 16)
- CSV з header

### 3. AWSCTD_embedding_kfold.py — Embedding + StratifiedKFold

Як embedding, але з крос-валідацією. Приймає один CSV файл.

```bash
python Python/AWSCTD_embedding_kfold.py CSV/malapi2019_firstN/dataset_n1000_train.csv AWSCTD-CNN-S-EMBEDDING
```

- `--embedding-dim=16` — розмір embedding
- Кількість фолдів з `config.ini`
- Результати: heatmap в `Python/CM/`, метрики в `Python/Metrics/`

---

## Runner-скрипти (batch експерименти)

| Скрипт | Модель | Датасети |
|--------|--------|----------|
| `run_firstN_experiments.py` | Embedding | malapi2019_firstN (n10–n1000) |
| `run_firstN_experiments_onehot.py` | One-Hot | malapi2019_firstN (n10–n1000) |
| `run_noRepeats_firstN_experiments.py` | Embedding | no_repeats_first_n |
| `run_noRepeats_firstN_onehot.py` | One-Hot | no_repeats_first_n |
| `run_lastN_experiments.py` | Embedding | last_n |
| `run_malapi2019o_experiment.py` | Embedding | malapi2019_o (різні вікна) |
| `run_embedding_dim_search.py` | Embedding | grid search по embedding dim |

Результати зберігаються в `Python/*_results*.csv`.

---

## config.ini

```ini
[MAIN]
nEpochs = 100
nBatchSize = 64
nPatience = 3
nKFolds = 5
bCategorical = true
sDevice = cpu        # cpu / gpu / auto
fLearningRate = 0.001
bGradientClipping = false
```

---

## Структура виводу

- `Python/CM/` — confusion matrix heatmaps (PNG)
- `Python/Metrics/` — CSV з метриками (Accuracy, Precision, Recall, F1, FPR, FNR, Loss, Train/Test Time)
- `Python/ROC/` — ROC криві (SVG)
- `Python/*_results*.csv` — зведені результати batch експериментів
