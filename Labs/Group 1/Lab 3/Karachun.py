from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import scikitplot as skplt
import matplotlib.pyplot as plt
import graphviz
import pandas as pd
import numpy as np


print('\n1) Відкрити та зчитати дані')
df = pd.read_csv('wq-r.csv', delimiter=';')


print('\n2) Кількість записів та полів')
print(f'Кількість записів: {df.shape[0]}')
print(f'Кількість полів: {df.shape[1]}')


print('\n3) Перші 10 записів')
print(df.head(10))


print('\n4) Розділення даних на навчальну та тестову вибірки')
X = df.drop('quality', axis=1)   # вихідні аргументи (усі стовпчики, окрім останнього)
y = df['quality']   # цільова характеристика (останній стовпець)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


print('\n5) Побудова та навчання дерева')
def classify(X, y, criterion='gini', max_depth=None, min_samples_leaf=2, random_state=0):
    clf = tree.DecisionTreeClassifier(criterion=criterion,
                                      max_depth=max_depth,
                                      min_samples_leaf=min_samples_leaf,
                                      random_state=random_state)
    
    return clf.fit(X, y)

# інформаційний приріст на основі неоднорідності Джині
clf_gini = classify(X_train, y_train, criterion='gini', max_depth=5)

# інформаційний приріст на основі ентропії
clf_entropy = classify(X_train, y_train, criterion='entropy', max_depth=5)

print('\n6) Графічна побудова дерева')
dot_data = tree.export_graphviz(clf_gini, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.format = 'png'
graph.view('decision tree')   # створення файлу з деревом та його відкриття


print('\n7) Обчислення метрик та їх графічне представлення')
pred_test_gini = clf_gini.predict(X_test)
pred_test_entropy = clf_entropy.predict(X_test)

pred_train_gini = clf_gini.predict(X_train)
pred_train_entropy = clf_entropy.predict(X_train)

# класифікаційні метрики для тестової вибірки (неоднорідность Джині)
accuracy_gini = accuracy_score(y_test, pred_test_gini)
precision_gini = precision_score(y_test, pred_test_gini, average=None, zero_division=1)
recall_gini = recall_score(y_test, pred_test_gini, average=None)
f1_gini = f1_score(y_test, pred_test_gini, average=None)

# класифікаційні метрики для тестової вибірки (ентропія)
accuracy_entropy = accuracy_score(y_test, pred_test_entropy)
precision_entropy = precision_score(y_test, pred_test_entropy, average=None, zero_division=1)
recall_entropy = recall_score(y_test, pred_test_entropy, average=None)
f1_entropy = f1_score(y_test, pred_test_entropy, average=None)

# класифікаційні метрики для тренувальної вибірки (неоднорідность Джині)
accuracy_gini = accuracy_score(y_train, pred_train_gini)
precision_gini = precision_score(y_train, pred_train_gini, average=None, zero_division=1)
recall_gini = recall_score(y_train, pred_train_gini, average=None)
f1_gini = f1_score(y_train, pred_train_gini, average=None)

# класифікаційні метрики для тренувальної вибірки (ентропія)
accuracy_entropy = accuracy_score(y_train, pred_train_entropy)
precision_entropy = precision_score(y_train, pred_train_entropy, average=None, zero_division=1)
recall_entropy = recall_score(y_train, pred_train_entropy, average=None)
f1_entropy = f1_score(y_train, pred_train_entropy, average=None)

print(f'Classification report (test gini):\n{classification_report(y_test, pred_test_gini, zero_division=1)}\n')
print(f'Classification report (train gini):\n{classification_report(y_train, pred_train_gini, zero_division=1)}\n')
print(f'Classification report (test entropy):\n{classification_report(y_test, pred_test_entropy, zero_division=1)}\n')
print(f'Classification report (train entropy):\n{classification_report(y_train, pred_train_entropy, zero_division=1)}\n')

# порівняння результатів при застосуванні різних критеріїв розщеплення
print('Графік порівняння результатів при застосуванні різних критеріїв розщеплення\n')

fig, ax = plt.subplots()
ax.set_title('Результати роботи моделі')
ax.set_xlabel('Індекс')
ax.set_ylabel('Якість')
ax.plot(pred_test_gini, label='Неоднорідність Джині')
ax.plot(pred_test_entropy, label='Ентропія')
ax.legend(loc='lower right')
plt.show()

# 7 visualization

print('Графік матриць помилок\n')
fig = plt.figure()

# матриця помилок (тестова вибірка, неоднорідність Джині)
ax1 = fig.add_subplot(221)
skplt.metrics.plot_confusion_matrix(pred_test_gini, y_test, title='Test gini', ax=ax1)

# матриця помилок (тестова вибірка, ентропія)
ax2 = fig.add_subplot(222)
skplt.metrics.plot_confusion_matrix(pred_test_entropy, y_test, title='Test entropy', ax=ax2)

# матриця помилок (тренувальна вибірка, неоднорідність Джині)
ax3 = fig.add_subplot(223)
skplt.metrics.plot_confusion_matrix(pred_train_gini, y_train, title='Train gini', ax=ax3)

# матриця помилок (тренувальна вибірка, ентропія)
ax4 = fig.add_subplot(224)
skplt.metrics.plot_confusion_matrix(pred_train_entropy, y_train, title='Train entropy', ax=ax4)

plt.tight_layout()
plt.show()


print('\n8) Вплив глибини дерева та графічне представлення')
# 3D
max_depths = range(2, 7)
min_samples_leafs = range(2, 7)
accuracy = []
for max_depth in max_depths:
    for min_samples_leaf in min_samples_leafs:
        clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy.append((max_depth, min_samples_leaf, accuracy_score(y_test, y_pred)))

accuracy = np.array(accuracy)
accuracy_label = [i[-1] for i in accuracy]
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_trisurf(accuracy[:, 0], accuracy[:, 1], accuracy[:, 2], cmap='viridis')
ax.set_xlabel('Max depth')
ax.set_ylabel('Min samples leaf')
ax.set_zlabel('Accuracy')
m = plt.cm.ScalarMappable(cmap='viridis')
m.set_array(accuracy_label)
cbar = plt.colorbar(m, ax=ax)
plt.show()

# вплив глибини дерева
depth_range = range(2, 7)
cv_scores = []

for depth in depth_range:
    clf = DecisionTreeClassifier(max_depth=depth)
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    cv_scores.append((depth, scores.mean()))
cv_scores_array = np.array(cv_scores)

depth_range = np.array(depth_range)
depth_range, = np.meshgrid(depth_range)

plt.plot(depth_range, cv_scores_array[:, 1])
plt.xlabel('Глибина дерева')
plt.ylabel('Середня точність')
plt.title('Вплив глибини дерева')
plt.show()

# вплив мінімальної кількості елементів у листі
min_samples_range = range(2, 7)
cv_scores = []

for min_samples_leaf in min_samples_range:
    clf = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf)
    scores = cross_val_score(clf, X_train, y_train, cv=5)
    cv_scores.append((min_samples_leaf, scores.mean()))
cv_scores_array = np.array(cv_scores)

min_samples_leaf = np.array(min_samples_range)

plt.plot(min_samples_leaf, cv_scores_array[:, 1])
plt.xlabel('Мінімальна кількість елементів в листі')
plt.ylabel('Середня точність')
plt.title('Вплив мінімальної кількості елементів у листі')
plt.show()


print('\n9) Графік важливості атрибутів')
importances = clf_gini.feature_importances_

fig, ax = plt.subplots()
ax.bar(df.drop('quality', axis=1).columns, importances)
plt.xticks(rotation=30)
ax.set_title('Feature importances')
ax.set_xlabel('Features')
ax.grid(axis='y')
plt.show()
