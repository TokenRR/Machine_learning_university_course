'''
Лабораторна робота № 2. Метрики
Підгрупа № 3
Виконав: Романецький Микита, КМ-01
'''


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


print('\n---  1  ---')

df = pd.read_csv('KM-01-3.csv')
print('Дані імпортовано')
# GT - Фактичне значення цільової характеристики. 
# Model_1_0 - Результат передбачення моделі № 1 у вигляді ймовірності приналежності об’єкту до класу 0. 
# Model_2_1 - Результат передбачення моделі № 2 у вигляді ймовірності приналежності об’єкту до класу 1. 

print('---  1  ---\n')

print('\n---  2  ---')
nrows, ncols = df.shape
print(f'Загальна кількість об`єктів = {nrows}')
print(f'Кількість об`єктів класу 1 = {sum(df["GT"])}')
print(f'Кількість об`єктів класу 0 = {nrows - sum(df["GT"])}')
print(f'{100*(sum(df["GT"])/nrows)}% - об`єктів класу 1 у вибірці')
print(f'{100-(100*(sum(df["GT"])/nrows))}% - об`єктів класу 0')
# df.value_counts()
print('---  2  ---\n')

print('\n---  3  ---')
print('---  A  ---')
step = 0.1
my_range = np.arange(step, 1, step)

accuracies_model_1_0 = []
accuracies_model_2_1 = []

precision_model_1_0 = []
precision_model_2_1 = []

recall_model_1_0 = []
recall_model_2_1 = []

f1_score_model_1_0 = []
f1_score_model_2_1 = []

metthews_c_c_model_1_0 = []
metthews_c_c_model_2_1 = []

bal_acc_model_1_0 = []
bal_acc_model_2_1 = []

youden_j_model_1_0 = []
youden_j_model_2_1 = []

auc_pr_model_1_0 = []
auc_pr_model_2_1 = []

auc_roc_model_1_0 = []
auc_roc_model_2_1 = []

for i in my_range:
    # влучність
    accuracies_model_1_0.append(round(accuracy_score(df['GT'], [int(j < i) for j in df['Model_1_0']]), 4))
    accuracies_model_2_1.append(round(accuracy_score(df['GT'], [int(j > i) for j in df['Model_2_1']]), 4))
    
    # точність
    precision_model_1_0.append(round(precision_score(df['GT'], [int(j < i) for j in df['Model_1_0']]), 4))
    precision_model_2_1.append(round(precision_score(df['GT'], [int(j > i) for j in df['Model_2_1']]), 4))

    # повнота
    recall_model_1_0.append(round(recall_score(df['GT'], [int(j < i) for j in df['Model_1_0']]), 4))
    recall_model_2_1.append(round(recall_score(df['GT'], [int(j > i) for j in df['Model_2_1']]), 4))

    # F міра
    f1_score_model_1_0.append(round(f1_score(df['GT'], [int(j < i) for j in df['Model_1_0']]), 4))
    f1_score_model_2_1.append(round(f1_score(df['GT'], [int(j > i) for j in df['Model_2_1']]), 4))

    # Коефіціент кореляції Метьюса
    metthews_c_c_model_1_0.\
        append(round(matthews_corrcoef(df['GT'], [int(j < i) for j in df['Model_1_0']]), 4))
    metthews_c_c_model_2_1.\
        append(round(matthews_corrcoef(df['GT'], [int(j > i) for j in df['Model_2_1']]), 4))

    # Збалансована влучність
    bal_acc_model_1_0.\
        append(round(balanced_accuracy_score(df['GT'], [int(j < i) for j in df['Model_1_0']]), 4))
    bal_acc_model_2_1.\
        append(round(balanced_accuracy_score(df['GT'], [int(j > i) for j in df['Model_2_1']]), 4))

    # Площа під PR-кривою
    auc_pr_model_1_0.\
        append(round(average_precision_score(df['GT'], [int(j < i) for j in df['Model_1_0']]), 4))
    auc_pr_model_2_1.\
        append(round(average_precision_score(df['GT'], [int(j > i) for j in df['Model_2_1']]), 4))
    
    # Площа під ROC-кривою
    auc_roc_model_1_0.\
        append(round(roc_auc_score(df['GT'], [int(j < i) for j in df['Model_1_0']]), 4))
    auc_roc_model_2_1.\
        append(round(roc_auc_score(df['GT'], [int(j > i) for j in df['Model_2_1']]), 4))    

# Індекс Юдена 
for bal_acc in bal_acc_model_1_0:
    youden_j_model_1_0.append(round(2 * bal_acc - 1, 4))
for bal_acc in bal_acc_model_2_1:
    youden_j_model_2_1.append(round(2 * bal_acc - 1, 4))


print(f'Влучність 1-ї моделі - {accuracies_model_1_0}')
print(f'Влучність 2-ї моделі - {accuracies_model_2_1}\n')

print(f'Точність 1-ї моделі - {precision_model_1_0}')
print(f'Точність 2-ї моделі - {precision_model_2_1}\n')

print(f'Повнота 1-ї моделі - {recall_model_1_0}')
print(f'Повнота 2-ї моделі - {recall_model_2_1}\n')

print(f'F-міра 1-ї моделі - {f1_score_model_1_0}')
print(f'F-міра 2-ї моделі - {f1_score_model_2_1}\n')

print(f'Коефіціент кореляції Метьюса 1-ї моделі - {metthews_c_c_model_1_0}')
print(f'Коефіціент кореляції Метьюса 2-ї моделі - {metthews_c_c_model_2_1}\n')

print(f'Збалансована влучність 1-ї моделі - {bal_acc_model_1_0}')
print(f'Збалансована влучність 2-ї моделі - {bal_acc_model_2_1}\n')

print(f'Індекс Юдена 1-ї моделі - {youden_j_model_1_0}')
print(f'Індекс Юдена 2-ї моделі - {youden_j_model_2_1}\n')

print(f'Площа під PR-кривою 1-ї моделі - {auc_pr_model_1_0}')
print(f'Площа під PR-кривою 2-ї моделі - {auc_pr_model_2_1}\n')

print(f'Площа під ROC-кривою 1-ї моделі - {auc_roc_model_1_0}')
print(f'Площа під ROC-кривою 2-ї моделі - {auc_roc_model_2_1}\n')
print('---  A  ---\n')

print('---  B  ---')
print('Графіки усіх метрик моделей')
def paint(values, color, label):
    plt.plot(my_range,
             values,
             color=color,
             label=label)

show_list = [accuracies_model_1_0, accuracies_model_2_1, precision_model_1_0, precision_model_2_1,
             recall_model_1_0, recall_model_2_1, f1_score_model_1_0, f1_score_model_2_1,
             metthews_c_c_model_1_0, metthews_c_c_model_2_1, bal_acc_model_1_0, bal_acc_model_2_1,
             youden_j_model_1_0, youden_j_model_2_1, auc_pr_model_1_0, auc_pr_model_2_1,
             auc_roc_model_1_0, auc_roc_model_2_1]
colors_list = ['red', 'blue', 'lightcoral', 'royalblue', 'lightsalmon', 'darkblue', 'salmon', 'purple',
               'tomato', 'mediumblue', 'orangered', 'grey', 'coral', 'teal', 'maroon', 'midnightblue',
               'green', 'black']
labels_list = ['accuracies_model_1_0', 'accuracies_model_2_1', 'precision_model_1_0', 'precision_model_2_1',
             'recall_model_1_0', 'recall_model_2_1', 'f1_score_model_1_0', 'f1_score_model_2_1',
             'metthews_c_c_model_1_0', 'metthews_c_c_model_2_1', 'bal_acc_model_1_0', 'bal_acc_model_2_1',
             'youden_j_model_1_0', 'youden_j_model_2_1', 'auc_pr_model_1_0', 'auc_pr_model_2_1',
             'auc_roc_model_1_0', 'auc_roc_model_2_1']

# input('Натисніть Enter, щоб побачити графіки метрик 1-ї моделі')
for id in range(len(show_list)):
    if id % 2 == 1:
        paint(show_list[id], colors_list[id], labels_list[id])
        plt.plot(np.array(show_list[id]).argmax()/10 + 0.1, max(show_list[id]),
                 'o', color='black', markersize=4)
plt.grid()
plt.title('Завдання 3b\nГрафік № 1 - Модель 2')  #  заголовок
plt.xlabel('Величина порогу', labelpad=5)  #  підпис осі Х
plt.ylabel('Значення метрики', labelpad=5)  #  підпис осі У
plt.legend(loc='center right',  #  виведення  легенди графіку на основі label in plt.plot()
           bbox_to_anchor=(1.5, 0.5))
plt.tight_layout()
plt.xlim(0, 0.92)
plt.show()

# input('Натисніть Enter, щоб побачити графіки метрик 2-ї моделі')
for id in range(len(show_list)):
    if id % 2 == 0:
        paint(show_list[id], colors_list[id], labels_list[id])
        plt.plot(np.array(show_list[id]).argmax()/10 + 0.1, max(show_list[id]),
                 'o', color='black', markersize=4)
plt.grid()
plt.title('Завдання 3b\nГрафік № 2 - Модель 1')  #  заголовок
plt.xlabel('Величина порогу', labelpad=5)  #  підпис осі Х
plt.ylabel('Значення метрики', labelpad=5)  #  підпис осі У
plt.legend(loc='center right',  #  виведення  легенди графіку на основі label in plt.plot()
           bbox_to_anchor=(1.5, 0.5))
plt.tight_layout()
plt.xlim(0, 0.92)
plt.show()

# input('Натисніть Enter, щоб побачити графіки метрик обох моделей разом')
for id in range(len(show_list)):
    paint(show_list[id], colors_list[id], labels_list[id])

plt.grid()
plt.title('Завдання 3b\nГрафік № 3')  #  заголовок
plt.xlabel('Величина порогу', labelpad=5)  #  підпис осі Х
plt.ylabel('Значення метрики', labelpad=5)  #  підпис осі У
plt.legend(loc='center right',  #  виведення  легенди графіку на основі label in plt.plot()
           bbox_to_anchor=(1.5, 0.5))
plt.tight_layout()
# plt.xticks(rotation=45)  #  нахил підписів на осі Х
plt.xlim(0, 0.92)
# plt.ylim(0.5, 1.05)

plt.show()
print('---  B  ---\n')

print('\n---  C  ---')
fig, ax = plt.subplots(1, 2)
fig.tight_layout()

S = 10  #  Залежить від кроку

model_1_class_0 = []
model_1_class_1 = []
model_2_class_0 = []
model_2_class_1 = []

for stp in my_range:
    model_1_class_0.append(sum([int(j < stp) for j in df['Model_1_0']]))
    model_1_class_1.append(sum([int(j > stp) for j in df['Model_1_0']]))

    model_2_class_0.append(sum([int(j < stp) for j in df['Model_2_1']]))
    model_2_class_1.append(sum([int(j > stp) for j in df['Model_2_1']]))

ax[0].set_title('Модель 1')
ax[0].set_xlabel('Значення оцінки класифікатору')
ax[0].set_ylabel('Кількість')
ax[0].yaxis.grid()
ax[0].bar(my_range, my_range)
ax[0].bar(my_range-0.015, model_1_class_0, width=0.03, color='lightcoral', label='Class 0')
ax[0].bar(my_range+0.015, model_1_class_1, width=0.03, color='red', label='Class 1')
ax[0].axes.set_xlim(0, 1)
ax[0].set_xticks(my_range)
ax[0].legend(loc='best')

ax[0].vlines(np.array(accuracies_model_1_0).argmax()/S + 0.008 + 0.1,
             0, 2_500, colors='midnightblue', label='Accuracy')
ax[0].vlines(np.array(precision_model_1_0).argmax()/S + 0.006 + 0.1,
             0, 2_500, colors='mediumblue', label='Precision')
ax[0].vlines(np.array(recall_model_1_0).argmax()/S + 0.004 + 0.1,
             0, 2_500, colors='blue', label='Recall')
ax[0].vlines(np.array(f1_score_model_1_0).argmax()/S + 0.002 + 0.1,
             0, 2_500, colors='slateblue', label='F1 score')
ax[0].vlines(np.array(metthews_c_c_model_1_0).argmax()/S + 0.1,
             0, 2_500, colors='royalblue', label='MCC')
ax[0].vlines(np.array(bal_acc_model_1_0).argmax()/S - 0.002 + 0.1,
             0, 2_500, colors='darkviolet', label='Balanced accuracy')
ax[0].vlines(np.array(youden_j_model_1_0).argmax()/S - 0.004 + 0.1,
             0, 2_500, colors='darkgrey', label='Index Youden')
ax[0].vlines(np.array(auc_pr_model_1_0).argmax()/S - 0.006 + 0.1,
             0, 2_500, colors='purple', label='AUC PR')
ax[0].vlines(np.array(auc_roc_model_1_0).argmax()/S - 0.008 + 0.1,
             0, 2_500, colors='black', label='AUC ROC')

# ----------------------------------------------------------------------------------------

ax[1].set_title('Модель 2')
ax[1].set_xlabel('Значення оцінки класифікатору')
ax[1].set_ylabel('Кількість')
ax[1].yaxis.grid()
ax[1].bar(my_range, my_range)
ax[1].bar(my_range-0.015, model_2_class_0, width=0.03, color='lightcoral')
ax[1].bar(my_range+0.015, model_2_class_1, width=0.03, color='red')
ax[1].axes.set_xlim(0, 1)
ax[1].set_xticks(my_range)

ax[1].vlines(np.array(accuracies_model_2_1).argmax()/S + 0.008 + 0.1,
             0, 3_100, colors='midnightblue', label='Accuracy')
ax[1].vlines(np.array(precision_model_2_1).argmax()/S + 0.006 + 0.1,
             0, 3_100, colors='mediumblue', label='Precision')
ax[1].vlines(np.array(recall_model_2_1).argmax()/S + 0.004 + 0.1,
             0, 3_100, colors='blue', label='Recall')
ax[1].vlines(np.array(f1_score_model_2_1).argmax()/S + 0.002 + 0.1,
             0, 3_100, colors='slateblue', label='F1 score')
ax[1].vlines(np.array(metthews_c_c_model_2_1).argmax()/S + 0.1,
             0, 3_100, colors='royalblue', label='MCC')
ax[1].vlines(np.array(bal_acc_model_2_1).argmax()/S - 0.002 + 0.1,
             0, 3_100, colors='darkviolet', label='Balanced accuracy')
ax[1].vlines(np.array(youden_j_model_2_1).argmax()/S - 0.004 + 0.1,
             0, 3_100, colors='darkgrey', label='Index Youden')
ax[1].vlines(np.array(auc_pr_model_2_1).argmax()/S - 0.006 + 0.1,
             0, 3_100, colors='purple', label='AUC PR')
ax[1].vlines(np.array(auc_roc_model_2_1).argmax()/S - 0.008 + 0.1,
             0, 3_100, colors='black', label='AUC ROC')

ax[1].legend(bbox_to_anchor=(1.79, 0.9))
plt.tight_layout()
# ----------------------------------------------------------------------------------------
# input('Натисніть Enter, щоб побачити графіки оцінки класифікаторів від кількості об єктів')

print('\nМодель 1')
print(f'Найбільше значення Accuracy = {max(accuracies_model_1_0)}, коли поріг = \
{np.array(accuracies_model_1_0).argmax()/S + 0.1}')
print(f'Найбільше значення Precision = {max(precision_model_1_0)}, коли поріг = \
{np.array(precision_model_1_0).argmax()/S + 0.1}')
print(f'Найбільше значення Recall = {max(recall_model_1_0)}, коли поріг = \
{np.array(recall_model_1_0).argmax()/S + 0.1}')
print(f'Найбільше значення F1 score = {max(f1_score_model_1_0)}, коли поріг = \
{np.array(f1_score_model_1_0).argmax()/S + 0.1}')
print(f'Найбільше значення MCC = {max(metthews_c_c_model_1_0)}, коли поріг = \
{np.array(metthews_c_c_model_1_0).argmax()/S + 0.1}')
print(f'Найбільше значення Balanced accuracy = {max(bal_acc_model_1_0)}, коли поріг = \
{np.array(bal_acc_model_1_0).argmax()/S + 0.1}')
print(f'Найбільше значення Index Youden = {max(youden_j_model_1_0)}, коли поріг = \
{np.array(youden_j_model_1_0).argmax()/S + 0.1}')
print(f'Найбільше значення AUC PR = {max(auc_pr_model_1_0)}, коли поріг = \
{np.array(auc_pr_model_1_0).argmax()/S + 0.1}')
print(f'Найбільше значення AUC ROC = {max(auc_roc_model_1_0)}, коли поріг = \
{np.array(auc_roc_model_1_0).argmax()/S + 0.1}')

print('\nМодель 2')
print(f'Найбільше значення Accuracy = {max(accuracies_model_2_1)}, коли поріг = \
{np.array(accuracies_model_2_1).argmax()/S + 0.1}')
print(f'Найбільше значення Precision = {max(precision_model_2_1)}, коли поріг = \
{np.array(precision_model_2_1).argmax()/S + 0.1}')
print(f'Найбільше значення Recall = {max(recall_model_2_1)}, коли поріг = \
{np.array(recall_model_2_1).argmax()/S + 0.1}')
print(f'Найбільше значення F1 score = {max(f1_score_model_2_1)}, коли поріг = \
{np.array(f1_score_model_2_1).argmax()/S + 0.1}')
print(f'Найбільше значення MCC = {max(metthews_c_c_model_2_1)}, коли поріг = \
{np.array(metthews_c_c_model_2_1).argmax()/S + 0.1}')
print(f'Найбільше значення Balanced accuracy = {max(bal_acc_model_2_1)}, коли поріг = \
{np.array(bal_acc_model_2_1).argmax()/S + 0.1}')
print(f'Найбільше значення Index Youden = {max(youden_j_model_2_1)}, коли поріг = \
{np.array(youden_j_model_2_1).argmax()/S + 0.1}')
print(f'Найбільше значення AUC PR = {max(auc_pr_model_2_1)}, коли поріг = \
{np.array(auc_pr_model_2_1).argmax()/S + 0.1}')
print(f'Найбільше значення AUC ROC = {max(auc_roc_model_2_1)}, коли поріг = \
{np.array(auc_roc_model_2_1).argmax()/S + 0.1}')

plt.show()
print('---  C  ---\n')

print('\n---  D  ---')
print('Графіки PR та ROC кривих')
# input('Натисніть Enter, щоб побачити графіки PR та ROC кривих')
fig, ax = plt.subplots(2, 2)
fig.tight_layout()

df['Model_1_v2'] = 1 - df['Model_1_0']

# ----------------------------------------------------------------------------------------

fpr, tpr, thresholds = roc_curve(df['GT'], df['Model_1_v2'])
gmax = np.array(np.sqrt(tpr * (1 - fpr))).argmax()
ax[0][1].set_title('ROC-curve, Model 1')
ax[0][1].set_xlabel('False Positive Rate')
ax[0][1].set_ylabel('True Positive Rate')
ax[0][1].grid()
ax[0][1].plot(fpr, tpr)
ax[0][1].text(fpr[gmax] - 0.03, tpr[gmax] + 0.03, round(thresholds[gmax], 2))
ax[0][1].plot(fpr[gmax], tpr[gmax], 'o', color='black', markersize=5)

# ----------------------------------------------------------------------------------------
fpr, tpr, thresholds = roc_curve(df['GT'], df['Model_2_1'])
gmax = np.array(np.sqrt(tpr * (1 - fpr))).argmax()
ax[1][1].set_title('ROC-curve, Model 2')
ax[1][1].set_xlabel('False Positive Rate')
ax[1][1].set_ylabel('True Positive Rate')
ax[1][1].grid()
ax[1][1].plot(fpr, tpr)
ax[1][1].text(fpr[gmax] - 0.03, tpr[gmax] + 0.03, round(thresholds[gmax], 2))
ax[1][1].plot(fpr[gmax], tpr[gmax], 'o', color='black', markersize=5)

# ----------------------------------------------------------------------------------------

precision, recall, thresholds = precision_recall_curve(df['GT'], df['Model_1_v2'])
fscoremax = np.array((2 * precision * recall) / (precision + recall)).argmax()
ax[0][0].set_title('PR-curve, Model 1')
ax[0][0].set_xlabel('Recall')
ax[0][0].set_ylabel('Precision')
ax[0][0].grid()
ax[0][0].plot(recall, precision)
ax[0][0].text(recall[fscoremax], precision[fscoremax] + 0.01, round(thresholds[gmax], 2))
ax[0][0].plot(recall[fscoremax], precision[fscoremax], 'o', color='black', markersize=5)

# ----------------------------------------------------------------------------------------
precision, recall, thresholds = precision_recall_curve(df['GT'], df['Model_2_1'])
fscoremax = np.array((2 * precision * recall) / (precision + recall)).argmax()
ax[1][0].set_title('PR-curve, Model 2')
ax[1][0].set_xlabel('Recall')
ax[1][0].set_ylabel('Precision')
ax[1][0].grid()
ax[1][0].plot(recall, precision)
ax[1][0].text(recall[fscoremax], precision[fscoremax] + 0.01, round(thresholds[gmax], 2))
ax[1][0].plot(recall[fscoremax], precision[fscoremax], 'o', color='black', markersize=5)


# ----------------------------------------------------------------------------------------
plt.tight_layout()
plt.show()
print('---  D  ---\n')
print('---  3  ---\n')

print('\n---  4  ---')
print('Хоч модель 1 - модель пошуку 0, а модель 2 - пошуку 1\n\
Оскільки вибірка даних є збалансованою, момент класифікації різних об єктів можна відкинути\n\
Обидві моделі однаково хороші при певних порогах відсічення, але перша краща, бо має трохи кращі значення\n\
і відповідно графіки')
print('---  4  ---\n')

print('\n---  5  ---')
bh = '12-06'
bh = bh.split('-')
K = int(bh[1])
rate = 50 + 10*(K%4)
df_no_zero = df[df['GT'] > 0]
df_zeros = df[df['GT'] < 1]
df_30, df_70 = train_test_split(df_no_zero, test_size=rate/100, shuffle=False)
df2 = pd.concat([df_30, df_zeros])
print(df2)
print('---  5  ---\n')

print('\n---  6  ---')
print(f'Загальна кількість об`єктів = {df2.GT.count()}')
print(f'Кількість об`єктів класу 1 = {sum(df2.GT)}')
print(f'Кількість об`єктів класу 0 = {df2.GT.count() - sum(df2.GT)}')
print(f'Відсоток видалених об`єктів класу 1 = {rate}%')
print('---  6  ---\n')

print('\n---  7  ---')
print('---  A  ---')
step = 0.1
my_range = np.arange(step, 1, step)

accuracies_model_1_0 = []
accuracies_model_2_1 = []

precision_model_1_0 = []
precision_model_2_1 = []

recall_model_1_0 = []
recall_model_2_1 = []

f1_score_model_1_0 = []
f1_score_model_2_1 = []

metthews_c_c_model_1_0 = []
metthews_c_c_model_2_1 = []

bal_acc_model_1_0 = []
bal_acc_model_2_1 = []

youden_j_model_1_0 = []
youden_j_model_2_1 = []

auc_pr_model_1_0 = []
auc_pr_model_2_1 = []

auc_roc_model_1_0 = []
auc_roc_model_2_1 = []

for i in my_range:
    # влучність
    accuracies_model_1_0.append(round(accuracy_score(df2['GT'], [int(j < i) for j in df2['Model_1_0']]), 4))
    accuracies_model_2_1.append(round(accuracy_score(df2['GT'], [int(j > i) for j in df2['Model_2_1']]), 4))
    
    # точність
    precision_model_1_0.append(round(precision_score(df2['GT'], [int(j < i) for j in df2['Model_1_0']]), 4))
    precision_model_2_1.append(round(precision_score(df2['GT'], [int(j > i) for j in df2['Model_2_1']]), 4))

    # повнота
    recall_model_1_0.append(round(recall_score(df2['GT'], [int(j < i) for j in df2['Model_1_0']]), 4))
    recall_model_2_1.append(round(recall_score(df2['GT'], [int(j > i) for j in df2['Model_2_1']]), 4))

    # F міра
    f1_score_model_1_0.append(round(f1_score(df2['GT'], [int(j < i) for j in df2['Model_1_0']]), 4))
    f1_score_model_2_1.append(round(f1_score(df2['GT'], [int(j > i) for j in df2['Model_2_1']]), 4))

    # Коефіціент кореляції Метьюса
    metthews_c_c_model_1_0.\
        append(round(matthews_corrcoef(df2['GT'], [int(j < i) for j in df2['Model_1_0']]), 4))
    metthews_c_c_model_2_1.\
        append(round(matthews_corrcoef(df2['GT'], [int(j > i) for j in df2['Model_2_1']]), 4))

    # Збалансована влучність
    bal_acc_model_1_0.\
        append(round(balanced_accuracy_score(df2['GT'], [int(j < i) for j in df2['Model_1_0']]), 4))
    bal_acc_model_2_1.\
        append(round(balanced_accuracy_score(df2['GT'], [int(j > i) for j in df2['Model_2_1']]), 4))

    # Площа під PR-кривою
    auc_pr_model_1_0.\
        append(round(average_precision_score(df2['GT'], [int(j < i) for j in df2['Model_1_0']]), 4))
    auc_pr_model_2_1.\
        append(round(average_precision_score(df2['GT'], [int(j > i) for j in df2['Model_2_1']]), 4))
    
    # Площа під ROC-кривою
    auc_roc_model_1_0.\
        append(round(roc_auc_score(df2['GT'], [int(j < i) for j in df2['Model_1_0']]), 4))
    auc_roc_model_2_1.\
        append(round(roc_auc_score(df2['GT'], [int(j > i) for j in df2['Model_2_1']]), 4))    

# Індекс Юдена 
for bal_acc in bal_acc_model_1_0:
    youden_j_model_1_0.append(round(2 * bal_acc - 1, 4))
for bal_acc in bal_acc_model_2_1:
    youden_j_model_2_1.append(round(2 * bal_acc - 1, 4))


print(f'Влучність 1-ї моделі - {accuracies_model_1_0}')
print(f'Влучність 2-ї моделі - {accuracies_model_2_1}\n')

print(f'Точність 1-ї моделі - {precision_model_1_0}')
print(f'Точність 2-ї моделі - {precision_model_2_1}\n')

print(f'Повнота 1-ї моделі - {recall_model_1_0}')
print(f'Повнота 2-ї моделі - {recall_model_2_1}\n')

print(f'F-міра 1-ї моделі - {f1_score_model_1_0}')
print(f'F-міра 2-ї моделі - {f1_score_model_2_1}\n')

print(f'Коефіціент кореляції Метьюса 1-ї моделі - {metthews_c_c_model_1_0}')
print(f'Коефіціент кореляції Метьюса 2-ї моделі - {metthews_c_c_model_2_1}\n')

print(f'Збалансована влучність 1-ї моделі - {bal_acc_model_1_0}')
print(f'Збалансована влучність 2-ї моделі - {bal_acc_model_2_1}\n')

print(f'Індекс Юдена 1-ї моделі - {youden_j_model_1_0}')
print(f'Індекс Юдена 2-ї моделі - {youden_j_model_2_1}\n')

print(f'Площа під PR-кривою 1-ї моделі - {auc_pr_model_1_0}')
print(f'Площа під PR-кривою 2-ї моделі - {auc_pr_model_2_1}\n')

print(f'Площа під ROC-кривою 1-ї моделі - {auc_roc_model_1_0}')
print(f'Площа під ROC-кривою 2-ї моделі - {auc_roc_model_2_1}\n')
print('---  A  ---\n')

print('---  B  ---')
print('Графіки усіх метрик моделей')
def paint(values, color, label):
    plt.plot(my_range,
             values,
             color=color,
             label=label)

show_list = [accuracies_model_1_0, accuracies_model_2_1, precision_model_1_0, precision_model_2_1,
             recall_model_1_0, recall_model_2_1, f1_score_model_1_0, f1_score_model_2_1,
             metthews_c_c_model_1_0, metthews_c_c_model_2_1, bal_acc_model_1_0, bal_acc_model_2_1,
             youden_j_model_1_0, youden_j_model_2_1, auc_pr_model_1_0, auc_pr_model_2_1,
             auc_roc_model_1_0, auc_roc_model_2_1]
colors_list = ['red', 'blue', 'lightcoral', 'royalblue', 'lightsalmon', 'darkblue', 'salmon', 'purple',
               'tomato', 'mediumblue', 'orangered', 'grey', 'coral', 'teal', 'maroon', 'midnightblue',
               'green', 'black']
labels_list = ['accuracies_model_1_0', 'accuracies_model_2_1', 'precision_model_1_0', 'precision_model_2_1',
             'recall_model_1_0', 'recall_model_2_1', 'f1_score_model_1_0', 'f1_score_model_2_1',
             'metthews_c_c_model_1_0', 'metthews_c_c_model_2_1', 'bal_acc_model_1_0', 'bal_acc_model_2_1',
             'youden_j_model_1_0', 'youden_j_model_2_1', 'auc_pr_model_1_0', 'auc_pr_model_2_1',
             'auc_roc_model_1_0', 'auc_roc_model_2_1']

# input('Натисніть Enter, щоб побачити графіки метрик 1-ї моделі')
for id in range(len(show_list)):
    if id % 2 == 1:
        paint(show_list[id], colors_list[id], labels_list[id])
        plt.plot(np.array(show_list[id]).argmax()/10 + 0.1, max(show_list[id]),
                 'o', color='black', markersize=4)
plt.grid()
plt.title('Завдання 7b\nГрафік № 1 - Модель 2')  #  заголовок
plt.xlabel('Величина порогу', labelpad=5)  #  підпис осі Х
plt.ylabel('Значення метрики', labelpad=5)  #  підпис осі У
plt.legend(loc='center right',  #  виведення  легенди графіку на основі label in plt.plot()
           bbox_to_anchor=(1.5, 0.5))
plt.tight_layout()
plt.xlim(0, 0.92)
plt.show()

# input('Натисніть Enter, щоб побачити графіки метрик 2-ї моделі')
for id in range(len(show_list)):
    if id % 2 == 0:
        paint(show_list[id], colors_list[id], labels_list[id])
        plt.plot(np.array(show_list[id]).argmax()/10 + 0.1, max(show_list[id]),
                 'o', color='black', markersize=4)
plt.grid()
plt.title('Завдання 7b\nГрафік № 2 - Модель 1')  #  заголовок
plt.xlabel('Величина порогу', labelpad=5)  #  підпис осі Х
plt.ylabel('Значення метрики', labelpad=5)  #  підпис осі У
plt.legend(loc='center right',  #  виведення  легенди графіку на основі label in plt.plot()
           bbox_to_anchor=(1.5, 0.5))
plt.tight_layout()
plt.xlim(0, 0.92)
plt.show()

# input('Натисніть Enter, щоб побачити графіки метрик обох моделей разом')
for id in range(len(show_list)):
    paint(show_list[id], colors_list[id], labels_list[id])

plt.grid()
plt.title('Завдання 7b\nГрафік № 3')  #  заголовок
plt.xlabel('Величина порогу', labelpad=5)  #  підпис осі Х
plt.ylabel('Значення метрики', labelpad=5)  #  підпис осі У
plt.legend(loc='center right',  #  виведення  легенди графіку на основі label in plt.plot()
           bbox_to_anchor=(1.5, 0.5))
plt.tight_layout()
# plt.xticks(rotation=45)  #  нахил підписів на осі Х
plt.xlim(0, 0.92)
# plt.ylim(0.5, 1.05)

plt.show()
print('---  B  ---\n')

print('\n---  C  ---')
fig, ax = plt.subplots(1, 2)
fig.tight_layout()

S = 10  #  Залежить від кроку

model_1_class_0 = []
model_1_class_1 = []
model_2_class_0 = []
model_2_class_1 = []

for stp in my_range:
    model_1_class_0.append(sum([int(j < stp) for j in df2['Model_1_0']]))
    model_1_class_1.append(sum([int(j > stp) for j in df2['Model_1_0']]))

    model_2_class_0.append(sum([int(j < stp) for j in df2['Model_2_1']]))
    model_2_class_1.append(sum([int(j > stp) for j in df2['Model_2_1']]))

ax[0].set_title('Модель 1')
ax[0].set_xlabel('Значення оцінки класифікатору')
ax[0].set_ylabel('Кількість')
ax[0].yaxis.grid()
ax[0].bar(my_range, my_range)
ax[0].bar(my_range-0.015, model_1_class_0, width=0.03, color='lightcoral', label='Class 0')
ax[0].bar(my_range+0.015, model_1_class_1, width=0.03, color='red', label='Class 1')
ax[0].axes.set_xlim(0, 1)
ax[0].set_xticks(my_range)
ax[0].legend(loc='best')

ax[0].vlines(np.array(accuracies_model_1_0).argmax()/S + 0.008 + 0.1,
             0, 2_300, colors='midnightblue', label='Accuracy')
ax[0].vlines(np.array(precision_model_1_0).argmax()/S + 0.006 + 0.1,
             0, 2_300, colors='mediumblue', label='Precision')
ax[0].vlines(np.array(recall_model_1_0).argmax()/S + 0.004 + 0.1,
             0, 2_300, colors='blue', label='Recall')
ax[0].vlines(np.array(f1_score_model_1_0).argmax()/S + 0.002 + 0.1,
             0, 2_300, colors='slateblue', label='F1 score')
ax[0].vlines(np.array(metthews_c_c_model_1_0).argmax()/S + 0.1,
             0, 2_300, colors='royalblue', label='MCC')
ax[0].vlines(np.array(bal_acc_model_1_0).argmax()/S - 0.002 + 0.1,
             0, 2_300, colors='darkviolet', label='Balanced accuracy')
ax[0].vlines(np.array(youden_j_model_1_0).argmax()/S - 0.004 + 0.1,
             0, 2_300, colors='darkgrey', label='Index Youden')
ax[0].vlines(np.array(auc_pr_model_1_0).argmax()/S - 0.006 + 0.1,
             0, 2_300, colors='purple', label='AUC PR')
ax[0].vlines(np.array(auc_roc_model_1_0).argmax()/S - 0.008 + 0.1,
             0, 2_300, colors='black', label='AUC ROC')

# ----------------------------------------------------------------------------------------

ax[1].set_title('Модель 2')
ax[1].set_xlabel('Значення оцінки класифікатору')
ax[1].set_ylabel('Кількість')
ax[1].yaxis.grid()
ax[1].bar(my_range, my_range)
ax[1].bar(my_range-0.015, model_2_class_0, width=0.03, color='lightcoral')
ax[1].bar(my_range+0.015, model_2_class_1, width=0.03, color='red')
ax[1].axes.set_xlim(0, 1)
ax[1].set_xticks(my_range)

ax[1].vlines(np.array(accuracies_model_2_1).argmax()/S + 0.008 + 0.1,
             0, 2_500, colors='midnightblue', label='Accuracy')
ax[1].vlines(np.array(precision_model_2_1).argmax()/S + 0.006 + 0.1,
             0, 2_500, colors='mediumblue', label='Precision')
ax[1].vlines(np.array(recall_model_2_1).argmax()/S + 0.004 + 0.1,
             0, 2_500, colors='blue', label='Recall')
ax[1].vlines(np.array(f1_score_model_2_1).argmax()/S + 0.002 + 0.1,
             0, 2_500, colors='slateblue', label='F1 score')
ax[1].vlines(np.array(metthews_c_c_model_2_1).argmax()/S + 0.1,
             0, 2_500, colors='royalblue', label='MCC')
ax[1].vlines(np.array(bal_acc_model_2_1).argmax()/S - 0.002 + 0.1,
             0, 2_500, colors='darkviolet', label='Balanced accuracy')
ax[1].vlines(np.array(youden_j_model_2_1).argmax()/S - 0.004 + 0.1,
             0, 2_500, colors='darkgrey', label='Index Youden')
ax[1].vlines(np.array(auc_pr_model_2_1).argmax()/S - 0.006 + 0.1,
             0, 2_500, colors='purple', label='AUC PR')
ax[1].vlines(np.array(auc_roc_model_2_1).argmax()/S - 0.008 + 0.1,
             0, 2_500, colors='black', label='AUC ROC')

ax[1].legend(bbox_to_anchor=(1.79, 0.9))
plt.tight_layout()
# ----------------------------------------------------------------------------------------
# input('Натисніть Enter, щоб побачити графіки оцінки класифікаторів від кількості об єктів')

print('\nМодель 1')
print(f'Найбільше значення Accuracy = {max(accuracies_model_1_0)}, коли поріг = \
{np.array(accuracies_model_1_0).argmax()/S + 0.1}')
print(f'Найбільше значення Precision = {max(precision_model_1_0)}, коли поріг = \
{np.array(precision_model_1_0).argmax()/S + 0.1}')
print(f'Найбільше значення Recall = {max(recall_model_1_0)}, коли поріг = \
{np.array(recall_model_1_0).argmax()/S + 0.1}')
print(f'Найбільше значення F1 score = {max(f1_score_model_1_0)}, коли поріг = \
{np.array(f1_score_model_1_0).argmax()/S + 0.1}')
print(f'Найбільше значення MCC = {max(metthews_c_c_model_1_0)}, коли поріг = \
{np.array(metthews_c_c_model_1_0).argmax()/S + 0.1}')
print(f'Найбільше значення Balanced accuracy = {max(bal_acc_model_1_0)}, коли поріг = \
{np.array(bal_acc_model_1_0).argmax()/S + 0.1}')
print(f'Найбільше значення Index Youden = {max(youden_j_model_1_0)}, коли поріг = \
{np.array(youden_j_model_1_0).argmax()/S + 0.1}')
print(f'Найбільше значення AUC PR = {max(auc_pr_model_1_0)}, коли поріг = \
{np.array(auc_pr_model_1_0).argmax()/S + 0.1}')
print(f'Найбільше значення AUC ROC = {max(auc_roc_model_1_0)}, коли поріг = \
{np.array(auc_roc_model_1_0).argmax()/S + 0.1}')

print('\nМодель 2')
print(f'Найбільше значення Accuracy = {max(accuracies_model_2_1)}, коли поріг = \
{np.array(accuracies_model_2_1).argmax()/S + 0.1}')
print(f'Найбільше значення Precision = {max(precision_model_2_1)}, коли поріг = \
{np.array(precision_model_2_1).argmax()/S + 0.1}')
print(f'Найбільше значення Recall = {max(recall_model_2_1)}, коли поріг = \
{np.array(recall_model_2_1).argmax()/S + 0.1}')
print(f'Найбільше значення F1 score = {max(f1_score_model_2_1)}, коли поріг = \
{np.array(f1_score_model_2_1).argmax()/S + 0.1}')
print(f'Найбільше значення MCC = {max(metthews_c_c_model_2_1)}, коли поріг = \
{np.array(metthews_c_c_model_2_1).argmax()/S + 0.1}')
print(f'Найбільше значення Balanced accuracy = {max(bal_acc_model_2_1)}, коли поріг = \
{np.array(bal_acc_model_2_1).argmax()/S + 0.1}')
print(f'Найбільше значення Index Youden = {max(youden_j_model_2_1)}, коли поріг = \
{np.array(youden_j_model_2_1).argmax()/S + 0.1}')
print(f'Найбільше значення AUC PR = {max(auc_pr_model_2_1)}, коли поріг = \
{np.array(auc_pr_model_2_1).argmax()/S + 0.1}')
print(f'Найбільше значення AUC ROC = {max(auc_roc_model_2_1)}, коли поріг = \
{np.array(auc_roc_model_2_1).argmax()/S + 0.1}')

plt.show()
print('---  C  ---\n')

print('\n---  D  ---')
# input('Натисніть Enter, щоб побачити графіки PR та ROC кривих')
print('Графіки PR та ROC кривих')
fig, ax = plt.subplots(2, 2)
fig.tight_layout()

df2['Model_1_v2'] = 1 - df2['Model_1_0']

# ----------------------------------------------------------------------------------------

fpr, tpr, thresholds = roc_curve(df2['GT'], df2['Model_1_v2'])
gmax = np.array(np.sqrt(tpr * (1 - fpr))).argmax()
ax[0][1].set_title('ROC-curve, Model 1')
ax[0][1].set_xlabel('False Positive Rate')
ax[0][1].set_ylabel('True Positive Rate')
ax[0][1].grid()
ax[0][1].plot(fpr, tpr)
ax[0][1].text(fpr[gmax] - 0.03, tpr[gmax] + 0.03, round(thresholds[gmax], 2))
ax[0][1].plot(fpr[gmax], tpr[gmax], 'o', color='black', markersize=5)

# ----------------------------------------------------------------------------------------
fpr, tpr, thresholds = roc_curve(df2['GT'], df2['Model_2_1'])
gmax = np.array(np.sqrt(tpr * (1 - fpr))).argmax()
ax[1][1].set_title('ROC-curve, Model 2')
ax[1][1].set_xlabel('False Positive Rate')
ax[1][1].set_ylabel('True Positive Rate')
ax[1][1].grid()
ax[1][1].plot(fpr, tpr)
ax[1][1].text(fpr[gmax] - 0.03, tpr[gmax] + 0.03, round(thresholds[gmax], 2))
ax[1][1].plot(fpr[gmax], tpr[gmax], 'o', color='black', markersize=5)

# ----------------------------------------------------------------------------------------

precision, recall, thresholds = precision_recall_curve(df2['GT'], df2['Model_1_v2'])
fscoremax = np.array((2 * precision * recall) / (precision + recall)).argmax()
ax[0][0].set_title('PR-curve, Model 1')
ax[0][0].set_xlabel('Recall')
ax[0][0].set_ylabel('Precision')
ax[0][0].grid()
ax[0][0].plot(recall, precision)
ax[0][0].text(recall[fscoremax], precision[fscoremax] + 0.01, round(thresholds[gmax], 2))
ax[0][0].plot(recall[fscoremax], precision[fscoremax], 'o', color='black', markersize=5)

# ----------------------------------------------------------------------------------------

precision, recall, thresholds = precision_recall_curve(df2['GT'], df2['Model_2_1'])
fscoremax = np.array((2 * precision * recall) / (precision + recall)).argmax()
ax[1][0].set_title('PR-curve, Model 2')
ax[1][0].set_xlabel('Recall')
ax[1][0].set_ylabel('Precision')
ax[1][0].grid()
ax[1][0].plot(recall, precision)
ax[1][0].text(recall[fscoremax], precision[fscoremax] + 0.01, round(thresholds[gmax], 2))
ax[1][0].plot(recall[fscoremax], precision[fscoremax], 'o', color='black', markersize=5)

# ----------------------------------------------------------------------------------------
plt.tight_layout()
plt.show()
print('---  D  ---\n')
print('---  7  ---\n')

print('\n---  8  ---')
print('Модель 1 є кращою тому що вона має більші значення метрик, і показала себе трохи краще за другу')
print('---  8  ---\n')

print('\n---  9  ---')
print('Незбалансований набір даних немає впливати на модель. Модель має бути стійкою до незбалансованості\n\
оскільки після того як було "викинуто" 70% даних, модель показала майже ті самі значення метрик\n\
,тому я вважаю її кращою ')
print('---  9  ---\n')