import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


print('\n---  Завдання 1  ---')
df = pd.read_csv('dataset3_l4.csv', delimiter=',')
print('Датасет завантажено')
print('---  Завдання 1  ---\n')


print('\n---  Завдання 2  ---')
print(f'Записів: {df.shape[0]}')
print('---  Завдання 2  ---\n')


print('\n---  Завдання 3  ---')
for col in df.columns:
    print(str(col))
print('---  Завдання 3  ---\n')


print('\n---  Завдання 4  ---')
RUN = True
while RUN:
    try:
        n = int(input('Введіть кількість перемішувань (>=3)  '))
        if n >= 3:
            RUN = False
    except:
        continue
ss = ShuffleSplit(n_splits=n, test_size=0.25, random_state=42)
splitted = ss.split(df.drop('NObeyesdad', axis=1), df['NObeyesdad'])

for _ in range(1):
    next(splitted)

train_index, test_index = next(splitted)

train = df.loc[train_index]
test = df.loc[test_index]

X_train = train.drop('NObeyesdad', axis=1)
y_train = train['NObeyesdad']
X_test = test.drop('NObeyesdad', axis=1)
y_test = test['NObeyesdad']

print(df.iloc[:,-1].value_counts())
print('\nКласи не є збалансованими')
print('---  Завдання 4  ---\n')


print('\n---  Завдання 5  ---')
label_encoder = LabelEncoder()
X_train_encoded = X_train.copy()
X_test_encoded = X_test.copy()

# Застосуємо кодування міток до кожного категорійного стовпчика
for col in ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']:
    X_train_encoded[col] = label_encoder.fit_transform(X_train[col])
    X_test_encoded[col] = label_encoder.transform(X_test[col])

neigh = KNeighborsClassifier()
neigh.fit(X_train_encoded, y_train)

predTest = neigh.predict(X_test_encoded)
predTrain = neigh.predict(X_train_encoded)
print('---  Завдання 5  ---\n')


print('\n---  Завдання 6  ---')
clf_report_train_for_view = classification_report(y_train, predTrain, zero_division=1)
clf_report_test_for_view = classification_report(y_test, predTest, zero_division=1)

print(f'Classification report train:\n{clf_report_train_for_view}\n')
print(f'Classification report test:\n{clf_report_test_for_view}\n')

clf_report = classification_report(y_test, predTest, output_dict=True, zero_division=1)

fig, ax = plt.subplots()

sns.heatmap(pd.DataFrame(clf_report).iloc[:-1,:].T, annot=True, fmt='.2f')

plt.tight_layout()
plt.show()
print('---  Завдання 6  ---\n')


print('\n---  Завдання 7  ---')
numbers = np.arange(1, 21)
accuracy = []

for p in numbers:
    neigh = KNeighborsClassifier(p=p)
    neigh.fit(X_train_encoded, y_train)
    predTest = neigh.predict(X_test_encoded)
    accuracy.append(accuracy_score(y_test, predTest))

fig, ax = plt.subplots()
ax.set_xlabel('Степінь метрики Мінковського')
ax.set_ylabel('Accuracy')
ax.plot([str(el) for el in numbers], accuracy, marker='.', color='black')
plt.tight_layout()
plt.show()
print('---  Завдання 7  ---\n')
