import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import ShuffleSplit,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as sm
pd.set_option('display.max_columns', None)

def num_6(y_test, y_pred, y_proba):
    dff = pd.DataFrame(
        sm.classification_report(y_test, y_pred, labels=[5, 6, 7, 4, 8, 3], zero_division=0, output_dict=True))
    res_list = []
    res_list.append([dff['weighted avg'][0], \
                     dff['weighted avg'][1], \
                     dff['weighted avg'][2], \
                     sm.balanced_accuracy_score(y_test, y_pred), \
                     sm.matthews_corrcoef(y_test, y_pred), \
                     sm.roc_auc_score(y_test, y_proba, average="weighted", multi_class="ovr"), \
                     sm.log_loss(y_test, y_proba)])

    df_res = pd.DataFrame(res_list, columns=['Precision_score', \
                                             'Recall_score', \
                                             'F1_score', \
                                             'Balanced_accuracy_score', \
                                             'Matthews_corrcoef', \
                                             'Roc_Auc_score', \
                                             'log_loss'])
    mt = sm.confusion_matrix(y_test, y_pred, labels=[5, 6, 7, 4, 8, 3])
    sm.ConfusionMatrixDisplay(mt, display_labels=[5, 6, 7, 4, 8, 3]).plot(cmap=plt.cm.Blues)
    plt.title(f'6: Confusion Matrix ')
    plt.tight_layout()
    plt.show()

    return df_res


print('\nЗавдання 1')
df = pd.read_csv("WQ-R.csv", sep = ";")
df.head()
print('Дані успішно скачано')


print('\nЗавдання 2')
print(f'Кількість рядків : {df.shape[0]} \nКількість стовпчиків : {df.shape[1]}\n')


print('\nЗавдання 3')
print(df.iloc[:, :-1].head())


print('\nЗавдання 4')
X = df.drop(columns = ['quality'],axis = 1)
y = df.quality

rs = ShuffleSplit(n_splits=10, test_size=.33, random_state=42)
rs.get_n_splits()
rs.split(X)

i=0
for train_index, test_index in rs.split(X):
    if i != 7:
        i+=1
        continue
    else:
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

print(f'\nПодивимося вибірка TRAIN сбалансована \n {y_train.value_counts()}')
print(f'\nПодивимося чи вибірка TEST сбалансована \n {y_test.value_counts()}')


print('\nЗавдання 5')
neigh = KNeighborsClassifier()
neigh.fit(X_train, y_train)

y_pred = neigh.predict(X_test)
y_proba = neigh.predict_proba(X_test)


print('\nЗавдання 6')
df_res = pd.DataFrame()
df_res = num_6(y_test, y_pred, y_proba)

y_pred_1 = neigh.predict(X_train)
y_proba_1 = neigh.predict_proba(X_train)
df_res_1 = num_6(y_train, y_pred_1, y_proba_1)
print(df_res)

fig, ax1 = plt.subplots(figsize=(13, 5))

ax1.bar(['Precision_score','Recall_score','F1_score','Balanced_accuracy_score','Matthews_corrcoef','Roc_Auc_score','log_loss'],df_res_1.iloc[0].values, width=0.5, edgecolor="white", linewidth=0.7,label = 'train', color='black')
ax1.bar(['Precision_score','Recall_score','F1_score','Balanced_accuracy_score','Matthews_corrcoef','Roc_Auc_score','log_loss'],df_res.iloc[0].values, width=0.5, edgecolor="white", linewidth=0.7,label = 'text', color='red')
print('\n')
ax1.set_title("Metrics")
ax1.legend(loc='upper left')
plt.grid(axis='y')
plt.tight_layout()
plt.show()


print('\nЗавдання 7')
clf_grid = KNeighborsClassifier()
parameters = {'n_neighbors': range(1,21)}
grid = GridSearchCV(clf_grid, parameters, cv=10,scoring = 'f1_weighted')
grid.fit(X_train,y_train)
df_n7 = pd.DataFrame(grid.cv_results_)

fig, ax = plt.subplots()
ax.plot(df_n7['mean_test_score'], color='black', marker='.')
ax.set_title('n_neighbors')
ax.set_xlabel('count')
ax.set_ylabel('score')
plt.tight_layout()
plt.show()