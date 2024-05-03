from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


def task_1(filename, delimeter):
    '''Відкрити та зчитати наданий файл з даними.'''
    
    print('\n---  1  ---')
    df = pd.read_csv(f'{filename}', delimiter=delimeter)
    print('Дані імпортовано')
    print('---  1  ---\n')
    return df


def task_2(df):
    '''Визначити та вивести кількість записів.'''

    print('\n---  2  ---')
    print(f'Записів: {df.shape[0]}')
    print('---  2  ---\n')


def task_3(df):
    '''Вивести атрибути набору даних.'''

    print('\n---  3  ---')
    print('Атрибути:')
    for atribute in df.columns:
        print(f'\t{atribute}')
    print('---  3  ---\n')


def task_4(df):
    '''Ввести з клавіатури кількість варіантів перемішування ( не менше 
    трьох) та отримати відповідну кількість варіантів перемішування 
    набору даних та розділення його на навчальну (тренувальну) та тестову 
    вибірки,  використовуючи  функцію  ShuffleSplit.  Сформувати  навчальну 
    та тестову вибірки на основі другого варіанту. З’ясувати 
    збалансованість набору даних.'''

    print('\n---  4  ---')
    while True:
        try:
            n_splt = int(input('Введіть кількість варіантів перемішування (не менше 3)  '))
            if n_splt > 2:
                break
        except:
            pass
    ss = ShuffleSplit(n_splits=n_splt, test_size=0.2, random_state=42)
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

    print(f'\nПодивимося на збалансованість класів: {list(df.iloc[:,-1].value_counts())}')
    print(df.iloc[:,-1].value_counts())
    print('\nКласи не збалансовані, тому що один із класів має 351 спостереження, а інший 272')
    print('В якості зважування візьмемо "weighted"')
    print('---  4  ---\n')

    return X_train, X_test, y_train, y_test


def task_5(X_train, X_test, y_train):
    '''Використовуючи  функцію  KNeighborsClassifier  бібліотеки  scikit-learn,
    збудувати класифікаційну модель на основі методу k найближчих
    сусідів ( значення всіх параметрів залишити за замовчуванням) та
    навчити її на тренувальній вибірці, вважаючи, що цільова
    характеристика визначається стовпчиком NObeyesdad, а всі інші
    виступають в ролі вихідних аргументів.'''

    print('\n---  5  ---')
    label_encoder = LabelEncoder()
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()

    # Застосуємо кодування міток до кожного категорійного стовпчика
    for col in ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']:
        X_train_encoded[col] = label_encoder.fit_transform(X_train[col])
        X_test_encoded[col] = label_encoder.transform(X_test[col])

    # Gender, Age, Height, Weight, family_history_with_overweight, FAVC, FCVC, NCP, CAEC,      SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS,                NObeyesdad
    # Female, 21,  1.62,   64,     yes,                            no,   2,    3,   Sometimes, no,    2,    no,  0,   1,   no,   Public_Transportation, Normal_Weight

    neigh = KNeighborsClassifier()
    neigh.fit(X_train_encoded, y_train)

    predTest = neigh.predict(X_test_encoded)
    predTrain = neigh.predict(X_train_encoded)
    print('Класифікаційна модель на основі k найближчих сусідів створена')
    print('---  5  ---\n')

    return predTest, predTrain


def task_6(predTest, predTrain, y_test, y_train):
    '''Обчислити класифікаційні метрики збудованої моделі для тренувальної
    та тестової вибірки. Представити результати роботи моделі на тестовій
    вибірці графічно.'''

    print('\n---  6  ---')
    accuracy = accuracy_score(y_test, predTest)
    precision = precision_score(y_test, predTest, average='weighted', zero_division=1)
    recall = recall_score(y_test, predTest, average='weighted')
    f1 = f1_score(y_test, predTest, average='weighted')

    clf_report_train_for_view = classification_report(y_train, predTrain, zero_division=1)
    clf_report_test_for_view = classification_report(y_test, predTest, zero_division=1)

    print(f'\nAccuracy: {accuracy}\nPrecision: {precision}\nRecall: {recall}\nf1: {f1}\n')
    print(f'Classification report test:\n{clf_report_test_for_view}\n')
    print(f'Classification report train:\n{clf_report_train_for_view}\n')

    clf_report = classification_report(y_test, predTest, output_dict=True, zero_division=1)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Завдання 6\nКласифікаційні метрики збудованої моделі для тестової вибірки\n')

    sns.heatmap(pd.DataFrame(clf_report).iloc[:-1,:].T, annot=True, cmap='Greens', fmt='.2f', ax=ax[0])
    sns.heatmap(pd.DataFrame(clf_report).T, annot=True, cmap='Greens', fmt='.2f', ax=ax[1])

    plt.tight_layout()
    plt.show()
    print('---  6  ---\n')


def task_7(X_train, X_test, y_test, y_train):
    '''З’ясувати вплив степеня метрики Мінковського ( від 1 до 20) на
    результати класифікації. Результати представити графічно.'''
    
    print('\n---  7  ---')
    print('*Графік впливу степеня метрики Мінковського на результат класифікації*')
    numbers = np.arange(1, 21)
    accuracy = []

    label_encoder = LabelEncoder()
    X_train_encoded = X_train.copy()
    X_test_encoded = X_test.copy()

    # Застосуємо кодування міток до кожного категорійного стовпчика
    for col in ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']:
        X_train_encoded[col] = label_encoder.fit_transform(X_train[col])
        X_test_encoded[col] = label_encoder.transform(X_test[col])

    for p in numbers:
        neigh = KNeighborsClassifier(p=p)
        neigh.fit(X_train_encoded, y_train)

        predTest = neigh.predict(X_test_encoded)

        accuracy.append(accuracy_score(y_test, predTest))

    fig, ax = plt.subplots()
    fig.suptitle('Завдання 7\nВплив степеня метрики Мінковського на результат класифікації\n')
    ax.set_xlabel('Степінь метрики Мінковського')
    ax.set_ylabel('Точність')
    ax.plot([str(el) for el in numbers], accuracy, marker='.')
    plt.tight_layout()
    plt.show()
    print('---  7  ---\n')


if __name__ == '__main__':
    df = task_1('dataset3_l4.csv', ',')

    task_2(df)

    task_3(df)

    X_train, X_test, y_train, y_test = task_4(df)

    predTest, predTrain = task_5(X_train, X_test, y_train)

    task_6(predTest, predTrain, y_test, y_train)

    task_7(X_train, X_test, y_test, y_train)
