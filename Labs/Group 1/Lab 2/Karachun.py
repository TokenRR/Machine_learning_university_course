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


print('\nЗавдання 1')
df = pd.read_csv('KM-02-1.csv')
print(df.head())


print('\n\nЗавдання 2')
nrows, ncols = df.shape
print(f'Загальна кількість об`єктів = {df.GT.count()}')
print(f'Кількість об`єктів класу 1 = {sum(df.GT)}')
print(f'Кількість об`єктів класу 0 = {df.GT.count()-sum(df.GT)}')


print('\n\nЗавдання 3 А')
step = 0.1
my_range = np.arange(step, 1, step)
print(f'Крок = {step}\n')

accuracies_model_1 = []
accuracies_model_2 = []
precision_model_1 = []
precision_model_2 = []
recall_model_1 = []
recall_model_2 = []
f1_score_model_1 = []
f1_score_model_2 = []
mcc_model_1 = []
mcc_model_2 = []
bal_acc_model_1 = []
bal_acc_model_2 = []
youden_j_model_1 = []
youden_j_model_2 = []
auc_pr_model_1 = []
auc_pr_model_2 = []
auc_roc_model_1 = []
auc_roc_model_2 = []

for i in my_range:
    accuracies_model_1.append(round(accuracy_score(df['GT'], [int(j > i) for j in df['Model_1']]), 4))
    accuracies_model_2.append(round(accuracy_score(df['GT'], [int(j > i) for j in df['Model_2']]), 4))

    precision_model_1.append(round(precision_score(df['GT'], [int(j > i) for j in df['Model_1']]), 4))
    precision_model_2.append(round(precision_score(df['GT'], [int(j > i) for j in df['Model_2']]), 4))
    
    recall_model_1.append(round(recall_score(df['GT'], [int(j > i) for j in df['Model_1']]), 4))
    recall_model_2.append(round(recall_score(df['GT'], [int(j > i) for j in df['Model_2']]), 4))

    f1_score_model_1.append(round(f1_score(df['GT'], [int(j > i) for j in df['Model_1']]), 4))
    f1_score_model_2.append(round(f1_score(df['GT'], [int(j > i) for j in df['Model_2']]), 4))

    mcc_model_1.append(round(matthews_corrcoef(df['GT'], [int(j > i) for j in df['Model_1']]), 4))
    mcc_model_2.append(round(matthews_corrcoef(df['GT'], [int(j > i) for j in df['Model_2']]), 4))

    bal_acc_model_1.append(round(balanced_accuracy_score(df['GT'], [int(j > i) for j in df['Model_1']]), 4))
    bal_acc_model_2.append(round(balanced_accuracy_score(df['GT'], [int(j > i) for j in df['Model_2']]), 4))

    auc_pr_model_1.append(round(average_precision_score(df['GT'], [int(j > i) for j in df['Model_1']]), 4))
    auc_pr_model_2.append(round(average_precision_score(df['GT'], [int(j > i) for j in df['Model_2']]), 4))
    
    auc_roc_model_1.append(round(roc_auc_score(df['GT'], [int(j > i) for j in df['Model_1']]), 4))
    auc_roc_model_2.append(round(roc_auc_score(df['GT'], [int(j > i) for j in df['Model_2']]), 4))    

for bal_acc in bal_acc_model_1:
    youden_j_model_1.append(round(2 * bal_acc - 1, 4))
for bal_acc in bal_acc_model_2:
    youden_j_model_2.append(round(2 * bal_acc - 1, 4))


print(f'Accuracy model 1 = {accuracies_model_1}')
print(f'Accuracy model 2 = {accuracies_model_2}\n')

print(f'Precision model 1 = {precision_model_1}')
print(f'Precision model 2 = {precision_model_2}\n')

print(f'Recall model 1 = {recall_model_1}')
print(f'Recall model 2 = {recall_model_2}\n')

print(f'F-Scores model 1 = {f1_score_model_1}')
print(f'F-Scores model 2 = {f1_score_model_2}\n')

print(f'MCC model 1 = {mcc_model_1}')
print(f'MCC model 2 = {mcc_model_2}\n')

print(f'Balanced Accuracy model 1 = {bal_acc_model_1}')
print(f'Balanced Accuracy model 2 = {bal_acc_model_2}\n')

print(f'Youden J index model 1 = {youden_j_model_1}')
print(f'Youden J index model 2 = {youden_j_model_2}\n')

print(f'AUC PRC model 1 = {auc_pr_model_1}')
print(f'AUC PRC model 2 = {auc_pr_model_2}\n')

print(f'AUC ROC model 1 = {auc_roc_model_1}')
print(f'AUC ROC model 2 = {auc_roc_model_2}\n')


print('\nЗавдання 3 В')
def paint(values, color, label):
    plt.plot(my_range,
             values,
             color=color,
             label=label)
    
def paint2(values, color, label):
    plt.plot(my_range,
             values,
             color=color,
             label=label,
             linestyle='--')

show_list = [accuracies_model_1, accuracies_model_2, precision_model_1, precision_model_2,
             recall_model_1, recall_model_2, f1_score_model_1, f1_score_model_2,
             mcc_model_1, mcc_model_2, bal_acc_model_1, bal_acc_model_2,
             youden_j_model_1, youden_j_model_2, auc_pr_model_1, auc_pr_model_2,
             auc_roc_model_1, auc_roc_model_2]
colors_list = ['red', 'black', 'red', 'black', 'red', 'black', 'red', 'black',
               'red', 'black', 'red', 'black', 'red', 'black', 'red', 'black',
               'red', 'black']
labels_list = ['Accuracy model 1', 'Accuracy model 2', 'Precision model 1', 'Precision model 1',
             'Recall model 1', 'Recall model 2', 'F-Scores model 1', 'F-Scores model 2',
             'MCC model 1', 'MCC model 2', 'Balanced Accuracy model 1', 'Balanced Accuracy model 2',
             'Youden J index model 1', 'Youden J index model 2', 'AUC PRC model 1', 'AUC PRC model 2',
             'AUC ROC model 1', 'AUC ROC model 2']

for id in range(len(show_list)):
    if id%2==0:
        paint(show_list[id], colors_list[id], labels_list[id])
        plt.plot(np.array(show_list[id]).argmax()/10 + 0.1, max(show_list[id]),
                 'o', color='green', markersize=6)
    else:
        paint2(show_list[id], colors_list[id], labels_list[id])
        plt.plot(np.array(show_list[id]).argmax()/10 + 0.1, max(show_list[id]),
                 'o', color='blue', markersize=6)

plt.grid()
plt.xlabel('Величина порогу', labelpad=5)
plt.ylabel('Значення метрики', labelpad=5)
plt.legend(loc='center right',
           bbox_to_anchor=(1.5, 0.5))
plt.tight_layout()
plt.xlim(0.05, 0.95)

plt.show()


print('\n\nЗавдання 3 С')
fig, ax = plt.subplots(1, 2)
fig.tight_layout()

S = 10

model_1_class_0 = []
model_1_class_1 = []
model_2_class_0 = []
model_2_class_1 = []

for stp in my_range:
    model_1_class_0.append(sum([int(j < stp) for j in df['Model_1']]))
    model_1_class_1.append(sum([int(j > stp) for j in df['Model_1']]))

    model_2_class_0.append(sum([int(j < stp) for j in df['Model_2']]))
    model_2_class_1.append(sum([int(j > stp) for j in df['Model_2']]))

ax[0].set_title('Model 1')
ax[0].set_xlabel('Значення оцінки класифікатору')
ax[0].set_ylabel('Кількість')
ax[0].yaxis.grid()
ax[0].bar(my_range, my_range)
ax[0].bar(my_range-0.015, model_1_class_0, width=0.03, color='royalblue', label='Class 0')
ax[0].bar(my_range+0.015, model_1_class_1, width=0.03, color='orange', label='Class 1')
ax[0].axes.set_xlim(0, 1)
ax[0].set_xticks(my_range)
ax[0].legend(loc='best')

ax[0].vlines(np.array(accuracies_model_1).argmax()/S + 0.008 + 0.1, 0, 5000, colors='red', label='Accuracy')
ax[0].vlines(np.array(precision_model_1).argmax()/S + 0.006 + 0.1, 0, 5000, colors='fuchsia', label='Precision')
ax[0].vlines(np.array(recall_model_1).argmax()/S + 0.004 + 0.1, 0, 5000, colors='green', label='Recall')
ax[0].vlines(np.array(f1_score_model_1).argmax()/S + 0.002 + 0.1, 0, 5000, colors='pink', label='F1 score')
ax[0].vlines(np.array(mcc_model_1).argmax()/S + 0.1, 0, 5000, colors='lightgreen', label='MCC')
ax[0].vlines(np.array(bal_acc_model_1).argmax()/S - 0.002 + 0.1, 0, 5000, colors='cyan', label='Balanced accuracy')
ax[0].vlines(np.array(youden_j_model_1).argmax()/S - 0.004 + 0.1, 0, 5000, colors='darkgrey', label='Youden J index')
ax[0].vlines(np.array(auc_pr_model_1).argmax()/S - 0.006 + 0.1, 0, 5000, colors='purple', label='AUC PRC')
ax[0].vlines(np.array(auc_roc_model_1).argmax()/S - 0.008 + 0.1, 0, 5000, colors='black', label='AUC ROC')

ax[1].set_title('Model 2')
ax[1].set_xlabel('Значення оцінки класифікатору')
ax[1].set_ylabel('Кількість')
ax[1].yaxis.grid()
ax[1].bar(my_range, my_range)
ax[1].bar(my_range-0.015, model_2_class_0, width=0.03, color='royalblue')
ax[1].bar(my_range+0.015, model_2_class_1, width=0.03, color='orange')
ax[1].axes.set_xlim(0, 1)
ax[1].set_xticks(my_range)

ax[1].vlines(np.array(accuracies_model_2).argmax()/S + 0.008 + 0.1, 0, 3800, colors='red', label='Accuracy')
ax[1].vlines(np.array(precision_model_2).argmax()/S + 0.006 + 0.1, 0, 3800, colors='fuchsia', label='Precision')
ax[1].vlines(np.array(recall_model_2).argmax()/S + 0.004 + 0.1, 0, 3800, colors='green', label='Recall')
ax[1].vlines(np.array(f1_score_model_2).argmax()/S + 0.002 + 0.1, 0, 3800, colors='pink', label='F1 score')
ax[1].vlines(np.array(mcc_model_2).argmax()/S + 0.1, 0, 3800, colors='lightgreen', label='MCC')
ax[1].vlines(np.array(bal_acc_model_2).argmax()/S - 0.002 + 0.1, 0, 3800, colors='cyan', label='Balanced accuracy')
ax[1].vlines(np.array(youden_j_model_2).argmax()/S - 0.004 + 0.1, 0, 3800, colors='darkgrey', label='Youden J index')
ax[1].vlines(np.array(auc_pr_model_2).argmax()/S - 0.006 + 0.1, 0, 3800, colors='purple', label='AUC PRC')
ax[1].vlines(np.array(auc_roc_model_2).argmax()/S - 0.008 + 0.1, 0, 3800, colors='black', label='AUC ROC')

ax[1].legend(bbox_to_anchor=(1.79, 0.9))
plt.tight_layout()

print('\nModel 1')
print(f'Max Accuracy = {max(accuracies_model_1)}   threshold = {round(np.array(accuracies_model_1).argmax()/S + 0.1, 1)}')
print(f'Max Precision = {max(precision_model_1)}   threshold = {round(np.array(precision_model_1).argmax()/S + 0.1, 1)}')
print(f'Max Recall = {max(recall_model_1)}   threshold = {round(np.array(recall_model_1).argmax()/S + 0.1, 1)}')
print(f'Max F1 score = {max(f1_score_model_1)}   threshold = {round(np.array(f1_score_model_1).argmax()/S + 0.1, 1)}')
print(f'Max MCC = {max(mcc_model_1)}   threshold = {round(np.array(mcc_model_1).argmax()/S + 0.1, 1)}')
print(f'Max Balanced accuracy = {max(bal_acc_model_1)}   threshold = {round(np.array(bal_acc_model_1).argmax()/S + 0.1, 1)}')
print(f'Max Index Youden = {max(youden_j_model_1)}   threshold = {round(np.array(youden_j_model_1).argmax()/S + 0.1, 1)}')
print(f'Max AUC PR = {max(auc_pr_model_1)}   threshold = {round(np.array(auc_pr_model_1).argmax()/S + 0.1, 1)}')
print(f'Max AUC ROC = {max(auc_roc_model_1)}   threshold = {round(np.array(auc_roc_model_1).argmax()/S + 0.1, 1)}')

print('\nModel 2')
print(f'Max Accuracy = {max(accuracies_model_2)}   threshold = {round(np.array(accuracies_model_2).argmax()/S + 0.1, 1)}')
print(f'Max Precision = {max(precision_model_2)}   threshold = {round(np.array(precision_model_2).argmax()/S + 0.1, 1)}')
print(f'Max Recall = {max(recall_model_2)}   threshold = {round(np.array(recall_model_2).argmax()/S + 0.1, 1)}')
print(f'Max F1 score = {max(f1_score_model_2)}   threshold = {round(np.array(f1_score_model_2).argmax()/S + 0.1, 1)}')
print(f'Max MCC = {max(mcc_model_2)}   threshold = {round(np.array(mcc_model_2).argmax()/S + 0.1, 1)}')
print(f'Max Balanced accuracy = {max(bal_acc_model_2)}   threshold = {round(np.array(bal_acc_model_2).argmax()/S + 0.1, 1)}')
print(f'Max Index Youden = {max(youden_j_model_2)}   threshold = {round(np.array(youden_j_model_2).argmax()/S + 0.1, 1)}')
print(f'Max AUC PR = {max(auc_pr_model_2)}   threshold = {round(np.array(auc_pr_model_2).argmax()/S + 0.1, 1)}')
print(f'Max AUC ROC = {max(auc_roc_model_2)}   threshold = {round(np.array(auc_roc_model_2).argmax()/S + 0.1, 1)}')

plt.show()


print('\n\nЗавдання 3 D')
fig, ax = plt.subplots(2, 2)
fig.tight_layout()

fpr, tpr, thresholds = roc_curve(df['GT'], df['Model_1'])
gmax = np.array(np.sqrt(tpr * (1 - fpr))).argmax()
ax[0][1].set_title('ROC-curve, Model 1')
ax[0][1].set_xlabel('False Positive Rate')
ax[0][1].set_ylabel('True Positive Rate')
ax[0][1].grid()
ax[0][1].plot(fpr, tpr)
ax[0][1].text(fpr[gmax] - 0.03, tpr[gmax] + 0.03, round(thresholds[gmax], 2))
ax[0][1].plot(fpr[gmax], tpr[gmax], 'o', color='black', markersize=5)

precision, recall, thresholds = precision_recall_curve(df['GT'], df['Model_1'])
fscoremax = np.array((2 * precision * recall) / (precision + recall)).argmax()
ax[0][0].set_title('PR-curve, Model 1')
ax[0][0].set_xlabel('Recall')
ax[0][0].set_ylabel('Precision')
ax[0][0].grid()
ax[0][0].plot(recall, precision)
ax[0][0].text(recall[fscoremax], precision[fscoremax] + 0.01, round(thresholds[gmax], 2))
ax[0][0].plot(recall[fscoremax], precision[fscoremax], 'o', color='black', markersize=5)

fpr, tpr, thresholds = roc_curve(df['GT'], df['Model_2'])
gmax = np.array(np.sqrt(tpr * (1 - fpr))).argmax()
ax[1][1].set_title('ROC-curve, Model 2')
ax[1][1].set_xlabel('False Positive Rate')
ax[1][1].set_ylabel('True Positive Rate')
ax[1][1].grid()
ax[1][1].plot(fpr, tpr)
ax[1][1].text(fpr[gmax] - 0.03, tpr[gmax] + 0.03, round(thresholds[gmax], 2))
ax[1][1].plot(fpr[gmax], tpr[gmax], 'o', color='black', markersize=5)

precision, recall, thresholds = precision_recall_curve(df['GT'], df['Model_2'])
fscoremax = np.array((2 * precision * recall) / (precision + recall)).argmax()
ax[1][0].set_title('PR-curve, Model 2')
ax[1][0].set_xlabel('Recall')
ax[1][0].set_ylabel('Precision')
ax[1][0].grid()
ax[1][0].plot(recall, precision)
ax[1][0].text(recall[fscoremax], precision[fscoremax] + 0.01, round(thresholds[gmax], 2))
ax[1][0].plot(recall[fscoremax], precision[fscoremax], 'o', color='black', markersize=5)

plt.tight_layout()
plt.show()


print('\n\nЗавдання 4')
print('Вибірка є збалансованою.\n\
Модель 1 все ж таки буде кращою за модель 2,\n\
оскільки вона має більші значення метрик при певних порогах відсічення\n\
*(це краще помітно при кроці 0.01 та менше)')


print('\n\nЗавдання 5')
bh = '04-07'
bh = bh.split('-')
K = int(bh[1])
rate = 50 + 10*(K%4)
df_no_zero = df[df['GT'] > 0]
df_zeros = df[df['GT'] < 1]
df_20, df_80 = train_test_split(df_no_zero, test_size=rate/100, shuffle=False)
df2 = pd.concat([df_20, df_zeros])
print(df2.head())


print('\n\nЗавдання 6')
print(f'Загальна кількість об`єктів до видалення = {df.GT.count()}')
print(f'Загальна кількість об`єктів після видалення = {df2.GT.count()}')
print(f'Кількість об`єктів класу 1 після видалення = {sum(df2.GT)}')
print(f'Кількість об`єктів класу 0 після видалення = {df2.GT.count() - sum(df2.GT)}')
print(f'Відсоток видалених об`єктів класу 1 = {rate}%')


print('\n\nЗавдання 7 А')
step = 0.1
my_range = np.arange(step, 1, step)
print(f'Крок = {step}\n')

accuracies_model_1 = []
accuracies_model_2 = []
precision_model_1 = []
precision_model_2 = []
recall_model_1 = []
recall_model_2 = []
f1_score_model_1 = []
f1_score_model_2 = []
mcc_model_1 = []
mcc_model_2 = []
bal_acc_model_1 = []
bal_acc_model_2 = []
youden_j_model_1 = []
youden_j_model_2 = []
auc_pr_model_1 = []
auc_pr_model_2 = []
auc_roc_model_1 = []
auc_roc_model_2 = []

for i in my_range:
    accuracies_model_1.append(round(accuracy_score(df2['GT'], [int(j > i) for j in df2['Model_1']]), 4))
    accuracies_model_2.append(round(accuracy_score(df2['GT'], [int(j > i) for j in df2['Model_2']]), 4))

    precision_model_1.append(round(precision_score(df2['GT'], [int(j > i) for j in df2['Model_1']]), 4))
    precision_model_2.append(round(precision_score(df2['GT'], [int(j > i) for j in df2['Model_2']]), 4))
    
    recall_model_1.append(round(recall_score(df2['GT'], [int(j > i) for j in df2['Model_1']]), 4))
    recall_model_2.append(round(recall_score(df2['GT'], [int(j > i) for j in df2['Model_2']]), 4))

    f1_score_model_1.append(round(f1_score(df2['GT'], [int(j > i) for j in df2['Model_1']]), 4))
    f1_score_model_2.append(round(f1_score(df2['GT'], [int(j > i) for j in df2['Model_2']]), 4))

    mcc_model_1.append(round(matthews_corrcoef(df2['GT'], [int(j > i) for j in df2['Model_1']]), 4))
    mcc_model_2.append(round(matthews_corrcoef(df2['GT'], [int(j > i) for j in df2['Model_2']]), 4))

    bal_acc_model_1.append(round(balanced_accuracy_score(df2['GT'], [int(j > i) for j in df2['Model_1']]), 4))
    bal_acc_model_2.append(round(balanced_accuracy_score(df2['GT'], [int(j > i) for j in df2['Model_2']]), 4))

    auc_pr_model_1.append(round(average_precision_score(df2['GT'], [int(j > i) for j in df2['Model_1']]), 4))
    auc_pr_model_2.append(round(average_precision_score(df2['GT'], [int(j > i) for j in df2['Model_2']]), 4))
    
    auc_roc_model_1.append(round(roc_auc_score(df2['GT'], [int(j > i) for j in df2['Model_1']]), 4))
    auc_roc_model_2.append(round(roc_auc_score(df2['GT'], [int(j > i) for j in df2['Model_2']]), 4))    

for bal_acc in bal_acc_model_1:
    youden_j_model_1.append(round(2 * bal_acc - 1, 4))
for bal_acc in bal_acc_model_2:
    youden_j_model_2.append(round(2 * bal_acc - 1, 4))


print(f'Accuracy model 1 = {accuracies_model_1}')
print(f'Accuracy model 2 = {accuracies_model_2}\n')

print(f'Precision model 1 = {precision_model_1}')
print(f'Precision model 2 = {precision_model_2}\n')

print(f'Recall model 1 = {recall_model_1}')
print(f'Recall model 2 = {recall_model_2}\n')

print(f'F-Scores model 1 = {f1_score_model_1}')
print(f'F-Scores model 2 = {f1_score_model_2}\n')

print(f'MCC model 1 = {mcc_model_1}')
print(f'MCC model 2 = {mcc_model_2}\n')

print(f'Balanced Accuracy model 1 = {bal_acc_model_1}')
print(f'Balanced Accuracy model 2 = {bal_acc_model_2}\n')

print(f'Youden J index model 1 = {youden_j_model_1}')
print(f'Youden J index model 2 = {youden_j_model_2}\n')

print(f'AUC PRC model 1 = {auc_pr_model_1}')
print(f'AUC PRC model 2 = {auc_pr_model_2}\n')

print(f'AUC ROC model 1 = {auc_roc_model_1}')
print(f'AUC ROC model 2 = {auc_roc_model_2}\n')


print('\n\nЗавдання 7 В')
def paint(values, color, label):
    plt.plot(my_range,
             values,
             color=color,
             label=label)
    
def paint2(values, color, label):
    plt.plot(my_range,
             values,
             color=color,
             label=label,
             linestyle='--')

show_list = [accuracies_model_1, accuracies_model_2, precision_model_1, precision_model_2,
             recall_model_1, recall_model_2, f1_score_model_1, f1_score_model_2,
             mcc_model_1, mcc_model_2, bal_acc_model_1, bal_acc_model_2,
             youden_j_model_1, youden_j_model_2, auc_pr_model_1, auc_pr_model_2,
             auc_roc_model_1, auc_roc_model_2]
colors_list = ['red', 'black', 'red', 'black', 'red', 'black', 'red', 'black',
               'red', 'black', 'red', 'black', 'red', 'black', 'red', 'black',
               'red', 'black']
labels_list = ['Accuracy model 1', 'Accuracy model 2', 'Precision model 1', 'Precision model 1',
             'Recall model 1', 'Recall model 2', 'F-Scores model 1', 'F-Scores model 2',
             'MCC model 1', 'MCC model 2', 'Balanced Accuracy model 1', 'Balanced Accuracy model 2',
             'Youden J index model 1', 'Youden J index model 2', 'AUC PRC model 1', 'AUC PRC model 2',
             'AUC ROC model 1', 'AUC ROC model 2']

for id in range(len(show_list)):
    if id%2==0:
        paint(show_list[id], colors_list[id], labels_list[id])
        plt.plot(np.array(show_list[id]).argmax()/10 + 0.1, max(show_list[id]),
                 'o', color='green', markersize=6)
    else:
        paint2(show_list[id], colors_list[id], labels_list[id])
        plt.plot(np.array(show_list[id]).argmax()/10 + 0.1, max(show_list[id]),
                 'o', color='blue', markersize=6)

plt.grid()
plt.xlabel('Величина порогу', labelpad=5)
plt.ylabel('Значення метрики', labelpad=5)
plt.legend(loc='center right',
           bbox_to_anchor=(1.5, 0.5))
plt.tight_layout()
plt.xlim(0.05, 0.95)

plt.show()

print('\n\nЗавдання 7 С')
fig, ax = plt.subplots(1, 2)
fig.tight_layout()

S = 10

model_1_class_0 = []
model_1_class_1 = []
model_2_class_0 = []
model_2_class_1 = []

for stp in my_range:
    model_1_class_0.append(sum([int(j < stp) for j in df['Model_1']]))
    model_1_class_1.append(sum([int(j > stp) for j in df['Model_1']]))

    model_2_class_0.append(sum([int(j < stp) for j in df['Model_2']]))
    model_2_class_1.append(sum([int(j > stp) for j in df['Model_2']]))

ax[0].set_title('Model 1')
ax[0].set_xlabel('Значення оцінки класифікатору')
ax[0].set_ylabel('Кількість')
ax[0].yaxis.grid()
ax[0].bar(my_range, my_range)
ax[0].bar(my_range-0.015, model_1_class_0, width=0.03, color='royalblue', label='Class 0')
ax[0].bar(my_range+0.015, model_1_class_1, width=0.03, color='orange', label='Class 1')
ax[0].axes.set_xlim(0, 1)
ax[0].set_xticks(my_range)
ax[0].legend(loc='best')

ax[0].vlines(np.array(accuracies_model_1).argmax()/S + 0.008 + 0.1, 0, 5000, colors='red', label='Accuracy')
ax[0].vlines(np.array(precision_model_1).argmax()/S + 0.006 + 0.1, 0, 5000, colors='fuchsia', label='Precision')
ax[0].vlines(np.array(recall_model_1).argmax()/S + 0.004 + 0.1, 0, 5000, colors='green', label='Recall')
ax[0].vlines(np.array(f1_score_model_1).argmax()/S + 0.002 + 0.1, 0, 5000, colors='pink', label='F1 score')
ax[0].vlines(np.array(mcc_model_1).argmax()/S + 0.1, 0, 5000, colors='lightgreen', label='MCC')
ax[0].vlines(np.array(bal_acc_model_1).argmax()/S - 0.002 + 0.1, 0, 5000, colors='cyan', label='Balanced accuracy')
ax[0].vlines(np.array(youden_j_model_1).argmax()/S - 0.004 + 0.1, 0, 5000, colors='darkgrey', label='Youden J index')
ax[0].vlines(np.array(auc_pr_model_1).argmax()/S - 0.006 + 0.1, 0, 5000, colors='purple', label='AUC PRC')
ax[0].vlines(np.array(auc_roc_model_1).argmax()/S - 0.008 + 0.1, 0, 5000, colors='black', label='AUC ROC')

ax[1].set_title('Model 2')
ax[1].set_xlabel('Значення оцінки класифікатору')
ax[1].set_ylabel('Кількість')
ax[1].yaxis.grid()
ax[1].bar(my_range, my_range)
ax[1].bar(my_range-0.015, model_2_class_0, width=0.03, color='royalblue')
ax[1].bar(my_range+0.015, model_2_class_1, width=0.03, color='orange')
ax[1].axes.set_xlim(0, 1)
ax[1].set_xticks(my_range)

ax[1].vlines(np.array(accuracies_model_2).argmax()/S + 0.008 + 0.1, 0, 3800, colors='red', label='Accuracy')
ax[1].vlines(np.array(precision_model_2).argmax()/S + 0.006 + 0.1, 0, 3800, colors='fuchsia', label='Precision')
ax[1].vlines(np.array(recall_model_2).argmax()/S + 0.004 + 0.1, 0, 3800, colors='green', label='Recall')
ax[1].vlines(np.array(f1_score_model_2).argmax()/S + 0.002 + 0.1, 0, 3800, colors='pink', label='F1 score')
ax[1].vlines(np.array(mcc_model_2).argmax()/S + 0.1, 0, 3800, colors='lightgreen', label='MCC')
ax[1].vlines(np.array(bal_acc_model_2).argmax()/S - 0.002 + 0.1, 0, 3800, colors='cyan', label='Balanced accuracy')
ax[1].vlines(np.array(youden_j_model_2).argmax()/S - 0.004 + 0.1, 0, 3800, colors='darkgrey', label='Youden J index')
ax[1].vlines(np.array(auc_pr_model_2).argmax()/S - 0.006 + 0.1, 0, 3800, colors='purple', label='AUC PRC')
ax[1].vlines(np.array(auc_roc_model_2).argmax()/S - 0.008 + 0.1, 0, 3800, colors='black', label='AUC ROC')

ax[1].legend(bbox_to_anchor=(1.79, 0.9))
plt.tight_layout()

print('\nModel 1')
print(f'Max Accuracy = {max(accuracies_model_1)}   threshold = {round(np.array(accuracies_model_1).argmax()/S + 0.1, 1)}')
print(f'Max Precision = {max(precision_model_1)}   threshold = {round(np.array(precision_model_1).argmax()/S + 0.1, 1)}')
print(f'Max Recall = {max(recall_model_1)}   threshold = {round(np.array(recall_model_1).argmax()/S + 0.1, 1)}')
print(f'Max F1 score = {max(f1_score_model_1)}   threshold = {round(np.array(f1_score_model_1).argmax()/S + 0.1, 1)}')
print(f'Max MCC = {max(mcc_model_1)}   threshold = {round(np.array(mcc_model_1).argmax()/S + 0.1, 1)}')
print(f'Max Balanced accuracy = {max(bal_acc_model_1)}   threshold = {round(np.array(bal_acc_model_1).argmax()/S + 0.1, 1)}')
print(f'Max Index Youden = {max(youden_j_model_1)}   threshold = {round(np.array(youden_j_model_1).argmax()/S + 0.1, 1)}')
print(f'Max AUC PR = {max(auc_pr_model_1)}   threshold = {round(np.array(auc_pr_model_1).argmax()/S + 0.1, 1)}')
print(f'Max AUC ROC = {max(auc_roc_model_1)}   threshold = {round(np.array(auc_roc_model_1).argmax()/S + 0.1, 1)}')

print('\nModel 2')
print(f'Max Accuracy = {max(accuracies_model_2)}   threshold = {round(np.array(accuracies_model_2).argmax()/S + 0.1, 1)}')
print(f'Max Precision = {max(precision_model_2)}   threshold = {round(np.array(precision_model_2).argmax()/S + 0.1, 1)}')
print(f'Max Recall = {max(recall_model_2)}   threshold = {round(np.array(recall_model_2).argmax()/S + 0.1, 1)}')
print(f'Max F1 score = {max(f1_score_model_2)}   threshold = {round(np.array(f1_score_model_2).argmax()/S + 0.1, 1)}')
print(f'Max MCC = {max(mcc_model_2)}   threshold = {round(np.array(mcc_model_2).argmax()/S + 0.1, 1)}')
print(f'Max Balanced accuracy = {max(bal_acc_model_2)}   threshold = {round(np.array(bal_acc_model_2).argmax()/S + 0.1, 1)}')
print(f'Max Index Youden = {max(youden_j_model_2)}   threshold = {round(np.array(youden_j_model_2).argmax()/S + 0.1, 1)}')
print(f'Max AUC PR = {max(auc_pr_model_2)}   threshold = {round(np.array(auc_pr_model_2).argmax()/S + 0.1, 1)}')
print(f'Max AUC ROC = {max(auc_roc_model_2)}   threshold = {round(np.array(auc_roc_model_2).argmax()/S + 0.1, 1)}')

plt.show()


print('\n\nЗавдання 7 D')
fig, ax = plt.subplots(2, 2)
fig.tight_layout()

fpr, tpr, thresholds = roc_curve(df2['GT'], df2['Model_1'])
gmax = np.array(np.sqrt(tpr * (1 - fpr))).argmax()
ax[0][1].set_title('ROC-curve, Model 1')
ax[0][1].set_xlabel('False Positive Rate')
ax[0][1].set_ylabel('True Positive Rate')
ax[0][1].grid()
ax[0][1].plot(fpr, tpr)
ax[0][1].text(fpr[gmax] - 0.03, tpr[gmax] + 0.03, round(thresholds[gmax], 2))
ax[0][1].plot(fpr[gmax], tpr[gmax], 'o', color='black', markersize=5)

precision, recall, thresholds = precision_recall_curve(df2['GT'], df2['Model_1'])
fscoremax = np.array((2 * precision * recall) / (precision + recall)).argmax()
ax[0][0].set_title('PR-curve, Model 1')
ax[0][0].set_xlabel('Recall')
ax[0][0].set_ylabel('Precision')
ax[0][0].grid()
ax[0][0].plot(recall, precision)
ax[0][0].text(recall[fscoremax], precision[fscoremax] + 0.01, round(thresholds[gmax], 2))
ax[0][0].plot(recall[fscoremax], precision[fscoremax], 'o', color='black', markersize=5)

fpr, tpr, thresholds = roc_curve(df2['GT'], df2['Model_2'])
gmax = np.array(np.sqrt(tpr * (1 - fpr))).argmax()
ax[1][1].set_title('ROC-curve, Model 2')
ax[1][1].set_xlabel('False Positive Rate')
ax[1][1].set_ylabel('True Positive Rate')
ax[1][1].grid()
ax[1][1].plot(fpr, tpr)
ax[1][1].text(fpr[gmax] - 0.03, tpr[gmax] + 0.03, round(thresholds[gmax], 2))
ax[1][1].plot(fpr[gmax], tpr[gmax], 'o', color='black', markersize=5)

precision, recall, thresholds = precision_recall_curve(df2['GT'], df2['Model_2'])
fscoremax = np.array((2 * precision * recall) / (precision + recall)).argmax()
ax[1][0].set_title('PR-curve, Model 2')
ax[1][0].set_xlabel('Recall')
ax[1][0].set_ylabel('Precision')
ax[1][0].grid()
ax[1][0].plot(recall, precision)
ax[1][0].text(recall[fscoremax], precision[fscoremax] + 0.01, round(thresholds[gmax], 2))
ax[1][0].plot(recall[fscoremax], precision[fscoremax], 'o', color='black', markersize=5)

plt.tight_layout()
plt.show()


print('\n\nЗавдання 8')
print('Модель 1 є кращою, оскільки при видаленні 80% об`єктів класу 1\n\
модель 1 все ще має кращі значення метрик')


print('\n\nЗавдання 9')
print('Незбалансований набір даних немає впливати на модель. Модель має бути стійкою до незбалансованості')