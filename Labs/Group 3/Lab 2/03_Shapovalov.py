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
import random
import pandas as pd
import numpy as np
import math as m
import matplotlib.pyplot as plt


df = pd.read_csv('KM-03-3.csv')


print('\n---  Завдання 2  ---')
print(f'Загальна кількість об`єктів = {df.GT.count()}')
print(f'Кількість об`єктів класу 1 = {sum(df["GT"])}')
print(f'Кількість об`єктів класу 2 = {4400 - sum(df["GT"])}')
print('---  2  ---\n')

print('\n---  Завдання 3  ---')
print('---  Підзавдання A  ---')
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
    accuracies_model_1_0.append(round(accuracy_score(df['GT'], [int(j < i) for j in df['Model_1_0']]), 4))
    accuracies_model_2_1.append(round(accuracy_score(df['GT'], [int(j > i) for j in df['Model_2_1']]), 4))
    
    precision_model_1_0.append(round(precision_score(df['GT'], [int(j < i) for j in df['Model_1_0']]), 4))
    precision_model_2_1.append(round(precision_score(df['GT'], [int(j > i) for j in df['Model_2_1']]), 4))

    recall_model_1_0.append(round(recall_score(df['GT'], [int(j < i) for j in df['Model_1_0']]), 4))
    recall_model_2_1.append(round(recall_score(df['GT'], [int(j > i) for j in df['Model_2_1']]), 4))

    f1_score_model_1_0.append(round(f1_score(df['GT'], [int(j < i) for j in df['Model_1_0']]), 4))
    f1_score_model_2_1.append(round(f1_score(df['GT'], [int(j > i) for j in df['Model_2_1']]), 4))

    metthews_c_c_model_1_0.\
        append(round(matthews_corrcoef(df['GT'], [int(j < i) for j in df['Model_1_0']]), 4))
    metthews_c_c_model_2_1.\
        append(round(matthews_corrcoef(df['GT'], [int(j > i) for j in df['Model_2_1']]), 4))

    bal_acc_model_1_0.\
        append(round(balanced_accuracy_score(df['GT'], [int(j < i) for j in df['Model_1_0']]), 4))
    bal_acc_model_2_1.\
        append(round(balanced_accuracy_score(df['GT'], [int(j > i) for j in df['Model_2_1']]), 4))

    auc_pr_model_1_0.\
        append(round(average_precision_score(df['GT'], [int(j < i) for j in df['Model_1_0']]), 4))
    auc_pr_model_2_1.\
        append(round(average_precision_score(df['GT'], [int(j > i) for j in df['Model_2_1']]), 4))
    
    auc_roc_model_1_0.\
        append(round(roc_auc_score(df['GT'], [int(j < i) for j in df['Model_1_0']]), 4))
    auc_roc_model_2_1.\
        append(round(roc_auc_score(df['GT'], [int(j > i) for j in df['Model_2_1']]), 4))    

for bal_acc in bal_acc_model_1_0:
    youden_j_model_1_0.append(round(2 * bal_acc - 1, 4))
for bal_acc in bal_acc_model_2_1:
    youden_j_model_2_1.append(round(2 * bal_acc - 1, 4))

print(f'accuracies_model_1_0 - {accuracies_model_1_0}')
print(f'accuracies_model_2_1 - {accuracies_model_2_1}\n')
print(f'precision_model_1_0 - {precision_model_1_0}')
print(f'precision_model_2_1 - {precision_model_2_1}\n')
print(f'recall_model_1_0 - {recall_model_1_0}')
print(f'recall_model_2_1 - {recall_model_2_1}\n')
print(f'f1_score_model_1_0 - {f1_score_model_1_0}')
print(f'f1_score_model_2_1 - {f1_score_model_2_1}\n')
print(f'metthews_c_c_model_1_0 - {metthews_c_c_model_1_0}')
print(f'metthews_c_c_model_2_1 - {metthews_c_c_model_2_1}\n')
print(f'bal_acc_model_1_0 - {bal_acc_model_1_0}')
print(f'bal_acc_model_2_1 - {bal_acc_model_2_1}\n')
print(f'youden_j_model_1_0 - {youden_j_model_1_0}')
print(f'youden_j_model_2_1 - {youden_j_model_2_1}\n')
print(f'auc_pr_model_1_0 - {auc_pr_model_1_0}')
print(f'auc_pr_model_2_1 - {auc_pr_model_2_1}\n')
print(f'auc_roc_model_1_0 - {auc_roc_model_1_0}')
print(f'auc_roc_model_2_1 - {auc_roc_model_2_1}\n')
print('---  A  ---\n')

print('---  Підзавдання B  ---')
print('*Графіки*')
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

for id in range(len(show_list)):
    if id % 2 == 1:
        paint(show_list[id], colors_list[id], labels_list[id])
        plt.plot(np.array(show_list[id]).argmax()/10 + 0.1, max(show_list[id]),
                 'o', color='black', markersize=4)
plt.grid()
plt.title('Завдання 3b\nГрафік № 1')
plt.xlabel('Величина порогу', labelpad=5)
plt.ylabel('Значення метрики', labelpad=5)
plt.legend(loc='center right',
           bbox_to_anchor=(1.5, 0.5))
plt.tight_layout()
plt.xlim(0, 0.92)
plt.show()

for id in range(len(show_list)):
    if id % 2 == 0:
        paint(show_list[id], colors_list[id], labels_list[id])
        plt.plot(np.array(show_list[id]).argmax()/10 + 0.1, max(show_list[id]),
                 'o', color='black', markersize=4)
plt.grid()
plt.title('Завдання 3b\nГрафік № 2')  #  заголовок
plt.xlabel('Величина порогу', labelpad=5)  #  підпис осі Х
plt.ylabel('Значення метрики', labelpad=5)  #  підпис осі У
plt.legend(loc='center right',  #  виведення  легенди графіку на основі label in plt.plot()
           bbox_to_anchor=(1.5, 0.5))
plt.tight_layout()
plt.xlim(0, 0.92)
plt.show()

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

print('\n---  Підзавдання C  ---')
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
ax[0].bar(my_range-0.015, model_1_class_0, width=0.03, color='gold', label='Кількість 0')
ax[0].bar(my_range+0.015, model_1_class_1, width=0.03, color='coral', label='Кількість 1')
ax[0].axes.set_xlim(0, 1)
ax[0].set_xticks(my_range)
ax[0].legend(loc='best')

ax[0].vlines(np.array(accuracies_model_1_0).argmax()/S + 0.008 + 0.1,
             0, 3100, colors='midnightblue', label='Accuracy')
ax[0].vlines(np.array(precision_model_1_0).argmax()/S + 0.006 + 0.1,
             0, 3100, colors='mediumblue', label='Precision')
ax[0].vlines(np.array(recall_model_1_0).argmax()/S + 0.004 + 0.1,
             0, 3100, colors='blue', label='Recall')
ax[0].vlines(np.array(f1_score_model_1_0).argmax()/S + 0.002 + 0.1,
             0, 3100, colors='slateblue', label='F1 score')
ax[0].vlines(np.array(metthews_c_c_model_1_0).argmax()/S + 0.1,
             0, 3100, colors='royalblue', label='MCC')
ax[0].vlines(np.array(bal_acc_model_1_0).argmax()/S - 0.002 + 0.1,
             0, 3100, colors='darkviolet', label='Balanced accuracy')
ax[0].vlines(np.array(youden_j_model_1_0).argmax()/S - 0.004 + 0.1,
             0, 3100, colors='darkgrey', label='Index Youden')
ax[0].vlines(np.array(auc_pr_model_1_0).argmax()/S - 0.006 + 0.1,
             0, 3100, colors='purple', label='AUC PR')
ax[0].vlines(np.array(auc_roc_model_1_0).argmax()/S - 0.008 + 0.1,
             0, 3100, colors='black', label='AUC ROC')

# ----------------------------------------------------------------------------------------

ax[1].set_title('Модель 2')
ax[1].set_xlabel('Значення оцінки класифікатору')
ax[1].set_ylabel('Кількість')
ax[1].yaxis.grid()
ax[1].bar(my_range, my_range)
ax[1].bar(my_range-0.015, model_2_class_0, width=0.03, color='gold')
ax[1].bar(my_range+0.015, model_2_class_1, width=0.03, color='coral')
ax[1].axes.set_xlim(0, 1)
ax[1].set_xticks(my_range)

ax[1].vlines(np.array(accuracies_model_2_1).argmax()/S + 0.008 + 0.1,
             0, 2300, colors='midnightblue', label='Accuracy')
ax[1].vlines(np.array(precision_model_2_1).argmax()/S + 0.006 + 0.1,
             0, 2300, colors='mediumblue', label='Precision')
ax[1].vlines(np.array(recall_model_2_1).argmax()/S + 0.004 + 0.1,
             0, 2300, colors='blue', label='Recall')
ax[1].vlines(np.array(f1_score_model_2_1).argmax()/S + 0.002 + 0.1,
             0, 2300, colors='slateblue', label='F1 score')
ax[1].vlines(np.array(metthews_c_c_model_2_1).argmax()/S + 0.1,
             0, 2300, colors='royalblue', label='MCC')
ax[1].vlines(np.array(bal_acc_model_2_1).argmax()/S - 0.002 + 0.1,
             0, 2300, colors='darkviolet', label='Balanced accuracy')
ax[1].vlines(np.array(youden_j_model_2_1).argmax()/S - 0.004 + 0.1,
             0, 2300, colors='darkgrey', label='Index Youden')
ax[1].vlines(np.array(auc_pr_model_2_1).argmax()/S - 0.006 + 0.1,
             0, 2300, colors='purple', label='AUC PR')
ax[1].vlines(np.array(auc_roc_model_2_1).argmax()/S - 0.008 + 0.1,
             0, 2300, colors='black', label='AUC ROC')

ax[1].legend(bbox_to_anchor=(1.79, 0.9))
plt.tight_layout()
# ----------------------------------------------------------------------------------------
print('*Графіки*')
plt.show()
print('---  C  ---\n')

print('\n---  D  ---')
fig, ax = plt.subplots(2, 2)
fig.tight_layout()

df['Model_1_v2'] = 1 - df['Model_1_0']

# ----------------------------------------------------------------------------------------

precision, recall, thresholds = precision_recall_curve(df['GT'], df['Model_1_v2'])
fscoremax = np.array((2 * precision * recall) / (precision + recall)).argmax()
ax[0][0].set_title('PR-curve, Model 1')
ax[0][0].set_xlabel('Recall')
ax[0][0].set_ylabel('Precision')
ax[0][0].grid()
ax[0][0].plot(recall, precision, color='red')
ax[0][0].plot(recall[fscoremax], precision[fscoremax], 'o', color='black', markersize=5)

# ----------------------------------------------------------------------------------------

fpr, tpr, thresholds = roc_curve(df['GT'], df['Model_1_v2'])
gmax = np.array(np.sqrt(tpr * (1 - fpr))).argmax()
ax[0][1].set_title('ROC-curve, Model 1')
ax[0][1].set_xlabel('False Positive Rate')
ax[0][1].set_ylabel('True Positive Rate')
ax[0][1].grid()
ax[0][1].plot(fpr, tpr, color='red')
ax[0][1].plot(fpr[gmax], tpr[gmax], 'o', color='black', markersize=5)

# ----------------------------------------------------------------------------------------
precision, recall, thresholds = precision_recall_curve(df['GT'], df['Model_2_1'])
fscoremax = np.array((2 * precision * recall) / (precision + recall)).argmax()
ax[1][0].set_title('PR-curve, Model 2')
ax[1][0].set_xlabel('Recall')
ax[1][0].set_ylabel('Precision')
ax[1][0].grid()
ax[1][0].plot(recall, precision, color='red')
ax[1][0].plot(recall[fscoremax], precision[fscoremax], 'o', color='black', markersize=5)

# ----------------------------------------------------------------------------------------
fpr, tpr, thresholds = roc_curve(df['GT'], df['Model_2_1'])
gmax = np.array(np.sqrt(tpr * (1 - fpr))).argmax()
ax[1][1].set_title('ROC-curve, Model 2')
ax[1][1].set_xlabel('False Positive Rate')
ax[1][1].set_ylabel('True Positive Rate')
ax[1][1].grid()
ax[1][1].plot(fpr, tpr, color='red')
ax[1][1].plot(fpr[gmax], tpr[gmax], 'o', color='black', markersize=5)

# ----------------------------------------------------------------------------------------
plt.tight_layout()
print('*Графіки*')
plt.show()
print('---  D  ---\n')
print('---  3  ---\n')

print('\n---  Завдання 4  ---')
print('Модель 1 краща, тому що метрики мають трохи вищі значення і на графіку здаються більш рівними')
print('---  4  ---\n')

print('\n---  Завдання 5  ---')
bh = '11-06'
bh = bh.split('-')
K = int(bh[1])
rate = 50 + 10*(K%4)
df_no_zero = df[df['GT'] > 0]
df_zeros = df[df['GT'] < 1]
df_30, df_70 = train_test_split(df_no_zero, test_size=rate/100, shuffle=False)
df2 = pd.concat([df_30, df_zeros])
print(df2)
print('---  5  ---\n')

print('\n---  Завдання 6  ---')
print(f'Загальна кількість об`єктів = {df2.GT.count()}')
print(f'Кількість об`єктів класу 1 = {sum(df2.GT)}')
print(f'Кількість об`єктів класу 0 = {df2.GT.count() - sum(df2.GT)}')
print(f'Відсоток видалених об`єктів класу 1 = {rate}%')
print('---  6  ---\n')

print('\n---  Завдання 7  ---')
print('---  Підзавдання A  ---')
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
    accuracies_model_1_0.append(round(accuracy_score(df2['GT'], [int(j < i) for j in df2['Model_1_0']]), 4))
    accuracies_model_2_1.append(round(accuracy_score(df2['GT'], [int(j > i) for j in df2['Model_2_1']]), 4))

    precision_model_1_0.append(round(precision_score(df2['GT'], [int(j < i) for j in df2['Model_1_0']]), 4))
    precision_model_2_1.append(round(precision_score(df2['GT'], [int(j > i) for j in df2['Model_2_1']]), 4))

    recall_model_1_0.append(round(recall_score(df2['GT'], [int(j < i) for j in df2['Model_1_0']]), 4))
    recall_model_2_1.append(round(recall_score(df2['GT'], [int(j > i) for j in df2['Model_2_1']]), 4))

    f1_score_model_1_0.append(round(f1_score(df2['GT'], [int(j < i) for j in df2['Model_1_0']]), 4))
    f1_score_model_2_1.append(round(f1_score(df2['GT'], [int(j > i) for j in df2['Model_2_1']]), 4))

    metthews_c_c_model_1_0.\
        append(round(matthews_corrcoef(df2['GT'], [int(j < i) for j in df2['Model_1_0']]), 4))
    metthews_c_c_model_2_1.\
        append(round(matthews_corrcoef(df2['GT'], [int(j > i) for j in df2['Model_2_1']]), 4))

    bal_acc_model_1_0.\
        append(round(balanced_accuracy_score(df2['GT'], [int(j < i) for j in df2['Model_1_0']]), 4))
    bal_acc_model_2_1.\
        append(round(balanced_accuracy_score(df2['GT'], [int(j > i) for j in df2['Model_2_1']]), 4))

    auc_pr_model_1_0.\
        append(round(average_precision_score(df2['GT'], [int(j < i) for j in df2['Model_1_0']]), 4))
    auc_pr_model_2_1.\
        append(round(average_precision_score(df2['GT'], [int(j > i) for j in df2['Model_2_1']]), 4))
    
    auc_roc_model_1_0.\
        append(round(roc_auc_score(df2['GT'], [int(j < i) for j in df2['Model_1_0']]), 4))
    auc_roc_model_2_1.\
        append(round(roc_auc_score(df2['GT'], [int(j > i) for j in df2['Model_2_1']]), 4))    

for bal_acc in bal_acc_model_1_0:
    youden_j_model_1_0.append(round(2 * bal_acc - 1, 4))
for bal_acc in bal_acc_model_2_1:
    youden_j_model_2_1.append(round(2 * bal_acc - 1, 4))

print(f'accuracies_model_1_0 - {accuracies_model_1_0}')
print(f'accuracies_model_2_1 - {accuracies_model_2_1}\n')
print(f'precision_model_1_0 - {precision_model_1_0}')
print(f'precision_model_2_1 - {precision_model_2_1}\n')
print(f'recall_model_1_0 - {recall_model_1_0}')
print(f'recall_model_2_1 - {recall_model_2_1}\n')
print(f'f1_score_model_1_0 - {f1_score_model_1_0}')
print(f'f1_score_model_2_1 - {f1_score_model_2_1}\n')
print(f'metthews_c_c_model_1_0 - {metthews_c_c_model_1_0}')
print(f'metthews_c_c_model_2_1 - {metthews_c_c_model_2_1}\n')
print(f'bal_acc_model_1_0 - {bal_acc_model_1_0}')
print(f'bal_acc_model_2_1 - {bal_acc_model_2_1}\n')
print(f'youden_j_model_1_0 - {youden_j_model_1_0}')
print(f'youden_j_model_2_1 - {youden_j_model_2_1}\n')
print(f'auc_pr_model_1_0 - {auc_pr_model_1_0}')
print(f'auc_pr_model_2_1 - {auc_pr_model_2_1}\n')
print(f'auc_roc_model_1_0 - {auc_roc_model_1_0}')
print(f'auc_roc_model_2_1 - {auc_roc_model_2_1}\n')
print('---  A  ---\n')

print('---  Підзавдання B  ---')
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


for id in range(len(show_list)):
    if id % 2 == 1:
        paint(show_list[id], colors_list[id], labels_list[id])
        plt.plot(np.array(show_list[id]).argmax()/10 + 0.1, max(show_list[id]),
                 'o', color='black', markersize=4)
plt.grid()
plt.title('Завдання 7b\nГрафік № 1')
plt.xlabel('Величина порогу', labelpad=5)
plt.ylabel('Значення метрики', labelpad=5)
plt.legend(loc='center right',
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
plt.title('Завдання 7b\nГрафік № 2')
plt.xlabel('Величина порогу', labelpad=5)
plt.ylabel('Значення метрики', labelpad=5)
plt.legend(loc='center right',
           bbox_to_anchor=(1.5, 0.5))
plt.tight_layout()
plt.xlim(0, 0.92)
plt.show()

for id in range(len(show_list)):
    paint(show_list[id], colors_list[id], labels_list[id])

plt.grid()
plt.title('Завдання 7b\nГрафік № 3') 
plt.xlabel('Величина порогу', labelpad=5)
plt.ylabel('Значення метрики', labelpad=5) 
plt.legend(loc='center right',
           bbox_to_anchor=(1.5, 0.5))
plt.tight_layout()
plt.xlim(0, 0.92)
plt.show()
print('---  B  ---\n')

print('\n---  Підзавдання C  ---')
print('*Графіки*')
fig, ax = plt.subplots(1, 2)
fig.tight_layout()

S = 10 

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

plt.show()
print('---  C  ---\n')

print('\n---  Підзавдання D  ---')
print('*Графіки*')
fig, ax = plt.subplots(2, 2)
fig.tight_layout()

df2['Model_1_v2'] = 1 - df2['Model_1_0']

precision, recall, thresholds = precision_recall_curve(df2['GT'], df2['Model_1_v2'])
fscoremax = np.array((2 * precision * recall) / (precision + recall)).argmax()
ax[0][0].set_title('PR-curve, Model 1')
ax[0][0].set_xlabel('Recall')
ax[0][0].set_ylabel('Precision')
ax[0][0].grid()
ax[0][0].plot(recall, precision, color='red')
ax[0][0].plot(recall[fscoremax], precision[fscoremax], 'o', color='black', markersize=5)

fpr, tpr, thresholds = roc_curve(df2['GT'], df2['Model_1_v2'])
gmax = np.array(np.sqrt(tpr * (1 - fpr))).argmax()
ax[0][1].set_title('ROC-curve, Model 1')
ax[0][1].set_xlabel('False Positive Rate')
ax[0][1].set_ylabel('True Positive Rate')
ax[0][1].grid()
ax[0][1].plot(fpr, tpr, color='red')
ax[0][1].plot(fpr[gmax], tpr[gmax], 'o', color='black', markersize=5)

precision, recall, thresholds = precision_recall_curve(df2['GT'], df2['Model_2_1'])
fscoremax = np.array((2 * precision * recall) / (precision + recall)).argmax()
ax[1][0].set_title('PR-curve, Model 2')
ax[1][0].set_xlabel('Recall')
ax[1][0].set_ylabel('Precision')
ax[1][0].grid()
ax[1][0].plot(recall, precision, color='red')
ax[1][0].plot(recall[fscoremax], precision[fscoremax], 'o', color='black', markersize=5)

fpr, tpr, thresholds = roc_curve(df2['GT'], df2['Model_2_1'])
gmax = np.array(np.sqrt(tpr * (1 - fpr))).argmax()
ax[1][1].set_title('ROC-curve, Model 2')
ax[1][1].set_xlabel('False Positive Rate')
ax[1][1].set_ylabel('True Positive Rate')
ax[1][1].grid()
ax[1][1].plot(fpr, tpr, color='red')
ax[1][1].plot(fpr[gmax], tpr[gmax], 'o', color='black', markersize=5)

plt.tight_layout()
plt.show()
print('---  D  ---\n')
print('---  7  ---\n')

print('\n---  Завдання 8  ---')
print('Перша модель є ліпшою, тому що метрики мають трохи вищі значення і на графіку здаються більш рівними')
print('---  8  ---\n')

print('\n---  Завдання 9  ---')
print('Незбалансований набір немає впливає на модель, тому модель 1 є ліпшою')
print('---  9  ---\n')

