from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


print('\n\n1) Відкрити та зчитати дані')
df = pd.read_csv('WQ-R.csv', delimiter=';')


print('\n2) Кількість записів та полів')
print(f'Кількість записів: {df.shape[0]}')
print(f'Кількість полів: {df.shape[1]}')


print('\n3) Атрибути набору даних')
for col in df.columns:
    print(col)


print('''\n4) Отримати десять варіантів перемішування набору даних та розділення
\tйого на навчальну (тренувальну) та тестову вибірки, використовуючи
\tфункцію 'ShuffleSplit'. Сформувати навчальну та тестові вибірки на
\tоснові восьмого варіанту. З’ясувати збалансованість набору даних''')

print(f'Подивимося на збалансованість класів: {list(df.iloc[:,-1].value_counts())}')
print(df.iloc[:,-1].value_counts())
print('Класи не збалансовані')

ss = ShuffleSplit(n_splits=10, test_size=0.25, random_state=42)
splitted = ss.split(df.drop('quality', axis=1), df['quality'])

for _ in range(7):
    next(splitted)

train_index, test_index = next(splitted)

train = df.loc[train_index]
test = df.loc[test_index]

X_train = train.drop('quality', axis=1)
y_train = train['quality']
X_test = test.drop('quality', axis=1)
y_test = test['quality']


print('''\n5) Використовуючи функцію 'KNeighborsClassifier' бібліотеки scikit-learn, 
\tзбудувати класифікаційну модель на основі методу k найближчих 
\tсусідів (значення всіх параметрів залишити за замовчуванням) та 
\tнавчити її на тренувальній вибірці, вважаючи, що цільова 
\tхарактеристика визначається стовпчиком 'quality', а всі інші виступають 
\tв ролі вихідних аргументів''')

neigh = KNeighborsClassifier()
neigh.fit(X_train, y_train)

predTest = neigh.predict(X_test)


print('''\n6) Обчислити класифікаційні метрики збудованої моделі для тренувальної 
\tта тестової вибірки. Представити результати роботи моделі на тестовій 
\tвибірці графічно.''')

accuracy = accuracy_score(y_test, predTest)
precision = precision_score(y_test, predTest, average='weighted', zero_division=1)
recall = recall_score(y_test, predTest, average='weighted')
f1 = f1_score(y_test, predTest, average='weighted')

clf_report = classification_report(y_test, predTest, output_dict=True, zero_division=1)
clf_report_for_view = classification_report(y_test, predTest, zero_division=1)

print(f'\nAccuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nf1: {f1}\n')
print(f'Classification report:\n{clf_report_for_view}\n')

sns.heatmap(pd.DataFrame(clf_report).T, annot=True, cmap='Greens', fmt='.2f')
ax = plt.gca()
ax.set_title('Завдання 6\nКласифікаційні метрики збудованої моделі\n')
plt.tight_layout()
plt.show()


print('''\n7) З’ясувати вплив кількості сусідів (від 1 до 20) на результати
\tкласифікації. Результати представити графічно''')

rang = np.arange(1, 21)
accuracy = []
for r in rang:
    neigh = KNeighborsClassifier(n_neighbors=r)
    neigh.fit(X_train, y_train)

    predTest = neigh.predict(X_test)

    accuracy.append(accuracy_score(y_test, predTest))

fig, ax = plt.subplots()
ax.set_title('Завдання 7\nВплив кількості сусідів')
ax.set_xlabel('Кількість сусідів')
ax.set_ylabel('Точність')
ax.plot([str(el) for el in rang], accuracy, marker='.')
plt.tight_layout()
plt.show()
