import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
import pandas as pd
import graphviz


print('\n---  Завдання 1  ---')
df = pd.read_csv('dataset3.csv')
print('Датасет завантажено')
print('---  Завдання 1  ---\n')

print('\n---  Завдання 2  ---')
print(f'Записів: {df.shape[0]}\nПолів: {df.shape[1]}')
print('---  Завдання 2  ---\n')

print('\n---  Завдання 3  ---')
print('Перші 10 записів')
print(df.head(10))
print('---  Завдання 3  ---\n')

print('\n---  Завдання 4  ---')
X, y = df.iloc[:,:-1], df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f'Кількість записів у тренувальній вибірці: {len(y_train)}')
print(f'Кількість записів у тестовій вибірці: {len(y_test)}')
print('---  Завдання 4  ---\n')

print('\n---  Завдання 5  ---')
clf_gini = tree.DecisionTreeClassifier(max_depth=5, random_state=42, criterion='gini')
clf_gini = clf_gini.fit(X_train, y_train)
print(clf_gini)
print('---  Завдання 5  ---\n')


print('\n---  Завдання 6  ---')
dot_tree = export_graphviz(clf_gini, out_file=None, filled=True, precision=2, special_characters=True)
graph = graphviz.Source(dot_tree, format='png') 
graph.render('tree', view=True, cleanup=True)
input('Press Enter => DELETE png file\n')
os.remove('tree.png')
print('---  Завдання 6  ---\n')

def measure_metrics(X, y, model):
    y_pred = model.predict(X)
    scores = [['accuracy_score', round(accuracy_score(y, y_pred), 4)]]
    scores.append(['precision', round(precision_score(y, y_pred, average='weighted', zero_division=0), 4)])
    scores.append(['recall', round(recall_score(y, y_pred, average='weighted'), 4)])
    scores.append(['f1_score', round(f1_score(y, y_pred, average='weighted'), 4)])
    return scores

def view(lst):
    for i in lst:
        print(i)
    return ''

print('\n---  Завдання 7  ---')
clf_entropy = tree.DecisionTreeClassifier(max_depth=5, random_state=42, criterion='entropy')
clf_entropy = clf_entropy.fit(X_train, y_train)

print('Gini Train metrics:')
print(view(measure_metrics(X_train, y_train, clf_gini)))

print('Gini Test metrics:')
print(view(measure_metrics(X_test, y_test, clf_gini)))

print('Entropy Train metrics:')
print(view(measure_metrics(X_train, y_train, clf_entropy)))

print('Entropy Test metrics:')
print(view(measure_metrics(X_test, y_test, clf_entropy)))

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
for criterion, i in zip(['gini', 'entropy'], range(2)):
    model = DecisionTreeClassifier(criterion=criterion, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    if i == 0:
        ax[i].set_title('Матриця помилок (Джині)')
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap=plt.cm.Oranges, ax=ax[i])
    else:
        ax[i].set_title('Матриця помилок (Ентропія)')
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap=plt.cm.Greens, ax=ax[i])
plt.tight_layout()
plt.show()
print('---  Завдання 7  ---\n')

print('\n---  Завдання 8  ---')
print('Графіки')
max_leaf_nodes_report = []
for max_leaf in range(2, 1250, 100):
    clf = tree.DecisionTreeClassifier(max_leaf_nodes=max_leaf)
    clf.fit(X_train, y_train)

    predicted_max_leaf = clf.predict(X_test)
    predicted_correct_count = 0

    for test, pred in zip(y_test, predicted_max_leaf):
        if test == pred:
            predicted_correct_count += 1

    max_leaf_nodes_report.append([max_leaf, predicted_correct_count / len(y_test)])

min_samples_report = []
for min_samples in range(2, 1250, 100):
    clf = tree.DecisionTreeClassifier(min_samples_split=min_samples)
    clf.fit(X_train, y_train)

    y_test_predict_min_samples_split = clf.predict(X_test)
    predicted_correct_count = 0

    for test, pred in zip(y_test, y_test_predict_min_samples_split):
        if test == pred:
            predicted_correct_count += 1

    min_samples_report.append([min_samples, predicted_correct_count / len(y_test)])

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

max_leaf_nodes_report_values = [el[1] for el in max_leaf_nodes_report]
min_samples_report_values = [el[1] for el in min_samples_report]

ax[0].plot(list(range(2, 1250, 100)), max_leaf_nodes_report_values, marker='o')
ax[0].set_title('Changing maximum count of leaf nodes')
ax[0].set_xlabel('Maximum leaf nodes parameter')
ax[0].set_ylabel('Accuracy')
ax[0].grid()

ax[1].plot(list(range(2, 1250, 100)), min_samples_report_values, marker='o')
ax[1].set_title('Changing minimum count of elements in inner node')
ax[1].set_xlabel('Minimum samples nodes parameter')
ax[1].set_ylabel('Accuracy')
ax[1].grid()

plt.tight_layout()
plt.show()
print('---  Завдання 8  ---\n')

print('\n---  Завдання 9  ---')
print('Стовпчикова діаграма важливості атрибутів')
clf_gini_importances = pd.Series(clf_gini.feature_importances_)
clf_entropy_importances = pd.Series(clf_entropy.feature_importances_)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

clf_gini_importances.plot.bar(ax=ax[0])
clf_entropy_importances.plot.bar(ax=ax[1])

for i in [0, 1]:
    ax[i].set_xlabel('Features')
    ax[i].set_ylabel('Importance')
    ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=0)

ax[0].set_title('Feature importance (Gini)')
ax[1].set_title('Feature importance (Entropy)')
plt.tight_layout()
plt.show()
print('---  Завдання 9  ---\n')
