import matplotlib.pyplot as plt
import pandas as pd
import graphviz
import os

from sklearn.model_selection import train_test_split

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay


def task_1(filename, delimeter):
    '''Відкрити та зчитати наданий файл з даними.'''
    print('\n---  1  ---')
    df = pd.read_csv(f'{filename}', delimiter=delimeter)
    print('Дані імпортовано')
    print('---  1  ---\n')
    return df

def task_2(df):
    '''Визначити та вивести кількість записів та кількість полів у завантаженому наборі даних.'''
    print('\n---  2  ---')
    print(f'Записів: {df.shape[0]}, полів: {df.shape[1]}')
    print('---  2  ---\n')

def task_3(df):
    '''Вивести перші 10 записів набору даних.'''
    print('\n---  3  ---')
    print('Перші 10 записів')
    print(df.head(10))
    print('---  3  ---\n')

def task_4(df):
    '''Розділити набір даних на навчальну (тренувальну) та тестову вибірки,
    попередньо перемішавши початковий набір даних.'''
    print('\n---  4  ---')
    X, y = df.iloc[:,:-1], df.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10, stratify=y)
    print(f'Кількість записів у тренувальній вибірці: {len(y_train)} (80%)')
    print(f'Кількість записів у тестовій вибірці: {len(y_test)} (20%)')
    print('---  4  ---\n')
    return X_train, X_test, y_train, y_test

def task_5(X_train, y_train):
    '''Використовуючи відповідні функції бібліотеки scikit-learn, збудувати
    класифікаційну модель дерева прийняття рішень глибини 5 та навчити
    її на тренувальній вибірці, вважаючи, що в наданому наборі даних
    цільова характеристика визначається останнім стовпчиком, а всі інші
    виступають в ролі вихідних аргументів.'''
    print('\n---  5  ---')
    clf_gini = tree.DecisionTreeClassifier(max_depth=5, random_state=10, criterion='gini')
    clf_gini = clf_gini.fit(X_train, y_train)

    clf_entropy = tree.DecisionTreeClassifier(max_depth=5, random_state=10, criterion='entropy')
    clf_entropy = clf_entropy.fit(X_train, y_train)

    print(f'{clf_gini}\n{clf_entropy}')
    print('*Дерева побудовано та навчено на тренувальному наборі*')
    print('---  5  ---\n')
    return clf_gini, clf_entropy

def task_6(clf_gini, clf_entropy):
    '''Представити графічно побудоване дерево за допомогою бібліотеки graphviz.'''
    def subtask(tree, format, name):
        '''Робить файл із деревом'''
        dot_tree = export_graphviz(tree, out_file=None, filled=True, precision=2,
                                        special_characters=True)
        graph = graphviz.Source(dot_tree, format=format) 
        graph.render(name, view=True, cleanup=True)

    print('\n---  6  ---')
    tries = 3
    while tries:
        try:
            MODE = int(input('\t1 - PDF format\n\t2 - PNG format\n'))
            if MODE == 1:
                subtask(clf_gini, 'pdf', 'graph_gini')
                subtask(clf_entropy, 'pdf', 'graph_entropy')
                input('Закрийте файли та натисніть Enter, щоб видалити PDF файли дерев\n')
                os.remove('graph_gini.pdf')
                os.remove('graph_entropy.pdf')
            elif MODE == 2:
                subtask(clf_gini, 'png', 'graph_gini')
                subtask(clf_entropy, 'png', 'graph_entropy')
                input('Закрийте файли та натисніть Enter, щоб видалити PNG файли дерев\n')
                os.remove('graph_gini.png')
                os.remove('graph_entropy.png')
            tries = 0
        except:
            tries-=1
            if tries != 0:
                print('Введіть значення 1 або 2')

    print('---  6  ---\n')

def task_7(df, X_train, y_train, clf_gini, clf_entropy):
    '''Обчислити класифікаційні метрики збудованої моделі для тренувальної
    та тестової вибірки. Представити результати роботи моделі на тестовій
    вибірці графічно. Порівняти  результати, отриманні при застосуванні
    різних критеріїв розщеплення: інформаційний приріст на основі
    ентропії чи неоднорідності Джині.'''
    def subtask_7(X, y, model):
        '''Допоміжна функція для знаходження значень метрик'''
        y_pred = model.predict(X)
        scores = [('accuracy', round(accuracy_score(y, y_pred), 3))]
        scores.append(('precision', round(precision_score(y, y_pred, average='weighted', zero_division=0), 3)))
        scores.append(('recall', round(recall_score(y, y_pred, average='weighted'), 3)))
        scores.append(('f1_score', round(f1_score(y, y_pred, average='weighted'), 3)))
        return scores
    print('\n---  7  ---')

    print(f'Подивимося на збалансованість класів: {list(df.iloc[:,-1].value_counts())}')
    print('Класи не збалансовані, тому в якості зважування візьмемо "weighted"')
    print('\nGini')
    print(f'Train metrics: {subtask_7(X_train, y_train, clf_gini)}')
    print(f'Test metrics: {subtask_7(X_test, y_test, clf_gini)}')
    print('\nEntropy')
    print(f'Train metrics: {subtask_7(X_train, y_train, clf_entropy)}')
    print(f'Test metrics: {subtask_7(X_test, y_test, clf_entropy)}')
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    for criterion, i in zip(['gini', 'entropy'], range(2)):
        model = DecisionTreeClassifier(criterion=criterion, max_depth=5, random_state=10)
        model.fit(X_train, y_train)
        if i == 0:
            ax[i].set_title('Матриця помилок на основі Джині')
            display = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap=plt.cm.Blues,
                                                            ax=ax[i], colorbar=False)
            cb = fig.colorbar(display.im_, ax=ax[i], location='left', pad=0.06)
            cb.ax.yaxis.set_ticks_position('left')
        else:
            ax[i].set_title('Матриця помилок на основі ентропії')
            ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap=plt.cm.Blues, ax=ax[i])
    plt.tight_layout()
    plt.show()
    print('---  7  ---\n')

def task_8(X_train, X_test, y_train, y_test):
    '''З’ясувати вплив максимальної кількості листів та мінімальної кількості
    елементів у внутрішньому вузлі для його подальшого розбиття на
    результати класифікації.Результати представити графічно.'''
    print('\n---  8  ---')
    print('Графіки впливу максимальної кількості листів')
    print('та мінімальної кількості елементів у внутрішньому вузлів')
    print('*Середній час на малювання графіку = 20 секунд*')
    my_range = range(2, 1000, 50)

    max_leaf_nodes_report = []
    for max_leaf in my_range:
        clf = tree.DecisionTreeClassifier(max_leaf_nodes=max_leaf)
        clf.fit(X_train, y_train)
        predicted_max_leaf = clf.predict(X_test)
        predicted_correct_count = 0
        for test, pred in zip(y_test, predicted_max_leaf):
            if test == pred:
                predicted_correct_count += 1
        max_leaf_nodes_report.append([max_leaf, predicted_correct_count / len(y_test)])
    # print(pd.Series(max_leaf_nodes_report))

    min_samples_report = []
    for min_samples in my_range:
        clf = tree.DecisionTreeClassifier(min_samples_split=min_samples)
        clf.fit(X_train, y_train)
        y_test_predict_min_samples_split = clf.predict(X_test)
        predicted_correct_count = 0
        for test, pred in zip(y_test, y_test_predict_min_samples_split):
            if test == pred:
                predicted_correct_count += 1
        min_samples_report.append([min_samples, predicted_correct_count / len(y_test)])
    # print(pd.Series(min_samples_report))

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    max_leaf_nodes_report_values = [el[1] for el in max_leaf_nodes_report]
    min_samples_report_values = [el[1] for el in min_samples_report]

    ax[0].plot(list(my_range), max_leaf_nodes_report_values, marker='o')
    ax[0].set_title('Вплив максимальної кількості листів')
    ax[0].set_xlabel('Максимальна кількість листів')
    ax[0].set_ylabel('Точність')
    ax[0].grid()

    ax[1].plot(list(my_range), min_samples_report_values, marker='o')
    ax[1].set_title('Вплив мінімальної кількості елементів у внутрішньому вузлі')
    ax[1].set_xlabel('Мінімальна кількість елементів у вузлі')
    ax[1].set_ylabel('Точність')
    ax[1].grid()

    plt.tight_layout()
    plt.show()
    print('---  8  ---\n')

def task_9(clf_gini, clf_entropy):
    '''Навести стовпчикову діаграму важливості атрибутів, які
    використовувалися для класифікації (див. feature_importances_).
    Пояснити, яким чином – на Вашу думку – цю важливість можна
    підрахувати.'''
    print('\n---  9  ---')
    print('Стовпчикова діаграма важливості атрибутів')
    clf_gini_importances = pd.Series(clf_gini.feature_importances_)
    clf_entropy_importances = pd.Series(clf_entropy.feature_importances_)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    clf_gini_importances.plot.bar(ax=ax[0])
    clf_entropy_importances.plot.bar(ax=ax[1])

    ax[0].set_title('Важливість атрибутів дерева на основі Джині')
    ax[0].set_xlabel('Атрибути')
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=0)
    ax[0].set_ylabel('Важливість')
    ax[0].grid(axis='y')

    ax[1].set_title('Важливість атрибутів дерева на основі ентропії')
    ax[1].set_xlabel('Атрибути')
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=0)
    ax[1].set_ylabel('Важливість')
    ax[1].grid(axis='y')

    plt.tight_layout()
    plt.show()
    print('---  9  ---\n')


if __name__ == '__main__':
    df = task_1('dataset3.csv', ',')

    task_2(df)

    task_3(df)

    X_train, X_test, y_train, y_test = task_4(df)

    clf_gini, clf_entropy = task_5(X_train, y_train)

    task_6(clf_gini, clf_entropy)

    task_7(df, X_train, y_train, clf_gini, clf_entropy)

    task_8(X_train, X_test, y_train, y_test)

    task_9(clf_gini, clf_entropy)
