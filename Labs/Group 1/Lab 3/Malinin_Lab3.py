import pandas as pd
from matplotlib import pyplot as plt
import sklearn.metrics as sm
from sklearn import tree,ensemble
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
import graphviz

pd.set_option('display.max_columns', None)


def num_7(y_test, y_pred, y_proba, name):
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
    sm.ConfusionMatrixDisplay(mt, display_labels=[5, 6, 7, 4, 8, 3]).plot(cmap = plt.cm.Blues)
    plt.title(f'7: Confusion Matrix - {name}')
    plt.show()

    return df_res


print('\nЗавдання 1')
df = pd.read_csv("WQ-R.csv", sep = ";")
print('Дані успішно скачано')


print('\nЗавдання 2')
print(f'\nКількість рядків : {df.shape[0]} \nКількість стовпчиків : {df.shape[1]}')
print(f'\nПодивимося скільки у нас є ункальних результатів класифікаторів і чи вибірка сбалансована \n {df["quality"].value_counts()}')


print('\nЗавдання 3')
print(f'{df.head(10)}')


print('\nЗавдання 4')
X = df.drop(columns = ['quality'],axis = 1)
y = df.quality
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)
print('Дані успішно поділено на тренувальну і тестову вибірку, 33% і 66% відповідно')


print('\nЗавдання 5')
clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 5) ## дерево сласификации
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)


print('\nЗавдання 6')
dot_graph = tree.export_graphviz(clf, filled=True, rounded=True)
graph = graphviz.Source(dot_graph)
graph.format = "png"
graph.render("tree_viz")


print('\nЗавдання 7')

print('Classificator with ENTROPY: \n')
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)  ## дерево сласификации
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

df_res_1 = pd.DataFrame()
df_res_1 = num_7(y_test, y_pred, y_proba, 'entropy')

# -------------------------------------------------------

print('Classificator with GINI: \n')
clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=5 )
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)

df_res_2 = pd.DataFrame()
df_res_2 = pd.DataFrame()
df_res_2 = num_7(y_test, y_pred, y_proba,'gini')
print(df_res_2)



fig5, ax5 = plt.subplots(figsize=(13, 5))

ax5.bar(['Precision_score','Recall_score','F1_score','Balanced_accuracy_score','Matthews_corrcoef','Roc_Auc_score','log_loss'],df_res_1.iloc[0].values, width=0.5, edgecolor="white", linewidth=0.7, label = 'entropy',color='black')
ax5.bar(['Precision_score','Recall_score','F1_score','Balanced_accuracy_score','Matthews_corrcoef','Roc_Auc_score','log_loss'],df_res_2.iloc[0].values, width=0.5, edgecolor="white", linewidth=0.7,label = 'gini', color='red')
print('\n')
ax5.set_title("Metrics")
ax5.legend(loc='upper left')
plt.grid(axis='y')
plt.tight_layout()
plt.show()


print('\nЗавдання 8')

clf_grid = tree.DecisionTreeClassifier()
parameters = {'criterion': ['gini','entropy'],'min_samples_leaf' :range(1,10), 'max_depth' : range(1,10) }
grid = GridSearchCV(clf_grid, parameters, cv=5, scoring = 'f1_weighted') ## дерево сласификации порівнює моделі за f1_weighted
grid.fit(X_train,y_train)
best_tree = grid.best_estimator_
print(f' Найкращі параметри : {grid.best_params_}')
print('\n ми беремо кращий параметр і змінюємо інший параметр відносно першого, щоб подивитися який вплив  глибини  дерева  та  мінімальної  кількості  елементів  в листі дерева на результати класифікації')
df_n8 = pd.DataFrame(grid.cv_results_)


fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4, figsize=(14, 5))

ax1.plot(df_n8.query('param_criterion == "gini" &  param_max_depth== 4')['param_min_samples_leaf'], df_n8.query('param_criterion == "gini" & param_max_depth == 4')['mean_test_score'])
ax1.set_title('gini and max_depth = 4')
ax1.set_xlabel('param_min_samples_leaf')
ax1.set_ylabel('mean_test_score')

ax2.plot(df_n8.query('param_criterion == "entropy" &  param_max_depth== 4')['param_min_samples_leaf'], df_n8.query('param_criterion == "entropy" & param_max_depth == 4')['mean_test_score'],color='tab:orange')
ax2.set_title('entropy and max_depth = 4')
ax2.set_xlabel('param_min_samples_leaf')
ax2.set_ylabel('mean_test_score')

ax3.plot(df_n8.query('param_criterion == "entropy" &  param_min_samples_leaf== 9')['param_max_depth'], df_n8.query('param_criterion == "gini" & param_min_samples_leaf== 9')['mean_test_score'])
ax3.set_title('gini and min_samples_leaf = 9')
ax3.set_xlabel('param_max_depth')
ax3.set_ylabel('mean_test_score')

ax4.plot(df_n8.query('param_criterion == "entropy" &  param_min_samples_leaf== 9')['param_max_depth'], df_n8.query('param_criterion == "entropy" & param_min_samples_leaf== 9')['mean_test_score'],color='tab:orange')
ax4.set_title('entropy and min_samples_leaf = 9')
ax4.set_xlabel('param_max_depth')
ax4.set_ylabel('mean_test_score')
plt.tight_layout()
plt.show()


print('\nЗавдання 9')
imp = pd.DataFrame(best_tree.feature_importances_, index=X.columns, columns=['importance'])
imp.sort_values('importance').plot(kind='barh', figsize=(10, 5), color='black')
plt.title(f'Важливості атрибутів')
plt.grid(axis='x')
plt.tight_layout()
plt.show()
