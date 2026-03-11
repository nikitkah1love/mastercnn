import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import sys

# Підключення до БД
con = sqlite3.connect('results_updated.db')

# Параметри: скільки останніх результатів показати (за замовчуванням 5)
n_results = int(sys.argv[1]) if len(sys.argv) > 1 else 5

# Читаємо останні результати
query = f"""
SELECT 
    File,
    Config as 'Parameters',
    ROUND(WindowAcc, 2) as 'Win Acc',
    ROUND(WindowPrecision, 3) as 'Win Prec',
    ROUND(WindowRecall, 3) as 'Win Rec',
    ROUND(WindowF1, 3) as 'Win F1',
    ROUND(WindowFPR, 3) as 'Win FPR',
    ROUND(TraceAccMean, 2) as 'MEAN Acc',
    ROUND(PrecisionMean, 3) as 'MEAN Prec',
    ROUND(RecallMean, 3) as 'MEAN Rec',
    ROUND(F1Mean, 3) as 'MEAN F1',
    ROUND(TraceAccMax, 2) as 'MAX Acc',
    ROUND(PrecisionMax, 3) as 'MAX Prec',
    ROUND(RecallMax, 3) as 'MAX Rec',
    ROUND(F1Max, 3) as 'MAX F1',
    ROUND(Loss, 4) as 'Loss',
    ExecutionTime as 'Time'
FROM results_windowed_v2 
ORDER BY ExecutionTime DESC 
LIMIT {n_results}
"""

df = pd.read_sql_query(query, con)
con.close()

# Скорочуємо шлях до файлу для читабельності
df['File'] = df['File'].str.replace('../CSV/', '').str.replace('.csv', '')

# Форматуємо Parameters - замінюємо довгі рядки на багаторядковий текст
def format_params(config):
    if pd.isna(config):
        return ''
    # Замінюємо коментар на коротший
    config = config.replace('# Стабільність навчання (для вирішення NaN loss на GPU)', '# Stability')
    return config

df['Parameters'] = df['Parameters'].apply(format_params)

# Створюємо фігуру - збільшуємо висоту для багаторядкового тексту
fig, ax = plt.subplots(figsize=(32, max(5, len(df) * 3)))
ax.axis('tight')
ax.axis('off')

# Встановлюємо ширини колонок вручну
col_widths = [0.06]  # File
col_widths.append(0.20)  # Parameters - дуже широка колонка
for _ in range(len(df.columns) - 2):  # Решта колонок
    col_widths.append(0.035)

# Створюємо таблицю
table = ax.table(cellText=df.values, colLabels=df.columns, 
                cellLoc='left', loc='center',
                colWidths=col_widths)

# Стилізація
table.auto_set_font_size(False)
table.set_fontsize(7)
table.scale(1, 6)  # Ще більша висота рядків

# Колір заголовків
for i in range(len(df.columns)):
    table[(0, i)].set_facecolor('#4472C4')
    table[(0, i)].set_text_props(weight='bold', color='white', va='center')

# Чергування кольорів рядків + рожевий для Parameters
for i in range(1, len(df) + 1):
    for j in range(len(df.columns)):
        cell = table[(i, j)]
        cell.set_height(0.3)  # Збільшуємо висоту клітинки
        
        # Колонка 1 - це Parameters - рожевий, вирівнювання зліва і зверху
        if j == 1:
            cell.set_facecolor('#FFB6C1')  # Світло-рожевий
            cell.set_text_props(fontsize=5, va='top', ha='left')  # Ще менший шрифт
        # Інші колонки - по центру
        else:
            cell.set_text_props(va='center', ha='center')
            if i % 2 == 0:
                cell.set_facecolor('#E7E6E6')
            else:
                cell.set_facecolor('#FFFFFF')

plt.title('Training Results - Windowed Dataset', fontsize=14, weight='bold', pad=20)
plt.tight_layout()
plt.savefig('results_table.png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Таблиця збережена в results_table.png")
print(f"   Показано останніх {len(df)} результатів")
