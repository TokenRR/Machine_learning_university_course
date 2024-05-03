import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder


# Task 1
def loadFile(path):
    data = pd.read_csv(path)
    df = pd.DataFrame(data)

    return df


# Task 2
def countNumOfValues(df):
    return len(df.values)


def printColumnsNames(df):
    for attribute in df.columns:
        print(attribute, end=" ")


def shuffleAndSplit(df):
    n_splits = 0
    while n_splits < 3 :
        n_splits = int(input("Input number of splits(must be bigger than 2): "))

    shuffle_split = ShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)

    selected_variant = 5

    for i, (train_index, test_index) in enumerate(shuffle_split.split(df)):
        if i == selected_variant:
            train_set = df.iloc[train_index]
            test_set = df.iloc[test_index]
            break

    # Виведення розмірів навчальної та тестової вибірок
    print("Розмір навчальної вибірки:", len(train_set))
    print("Розмір тестової вибірки:", len(test_set))

    return train_set, test_set


def buildClassificationModel(df):
    label_encoder = LabelEncoder()
    encoded_df = df.copy()
    categorical_columns = df.dtypes[df.dtypes == 'object'].index.tolist()

    for column in categorical_columns:
        encoded_df[column] = label_encoder.fit_transform(df[column])

    train_set, test_set = shuffleAndSplit(encoded_df)
    X_train = train_set.drop('NObeyesdad', axis=1)
    y_train = train_set['NObeyesdad']

    X_test = test_set.drop('NObeyesdad', axis=1)
    y_test = test_set['NObeyesdad']

    k = 5
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)

    accuracy = knn_model.score(X_test, y_test)
    print("Точність моделі:", accuracy)

    target_names = ['Obesity_Type_I', 'Obesity_Type_III', 'Obesity_Type_II', 'Overweight_Level_I', 'Overweight_Level_II', 'Normal_Weight', 'Insufficient_Weight']
    y_train_pred = knn_model.predict(X_train)
    train_report = classification_report(y_train, y_train_pred, target_names=target_names)

    print("Метрики для тренувальної вибірки:\n", train_report)

    y_test_pred = knn_model.predict(X_test)
    test_report = classification_report(y_test, y_test_pred)
    print("Метрики для тестової вибірки:\n", test_report)

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    class_report_test = pd.DataFrame(classification_report(y_test, y_test_pred, target_names=target_names, output_dict=True)).transpose()
    class_report_test = class_report_test.iloc[:, :3]
    plot = ax.imshow(class_report_test, cmap='PuOr', interpolation='nearest')

    ax.set_xticks(range(len(class_report_test.columns)))
    ax.set_yticks(range(len(class_report_test.index)))
    ax.set_xticklabels(class_report_test.columns)
    ax.set_yticklabels(class_report_test.index)

    title = "Classification report of testing model"
    ax.set_title(title)

    plt.colorbar(plot, ax=ax)
    plt.show()

    cm = confusion_matrix(y_test, y_test_pred)

    plt.figure(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix of testing model")
    plt.show()

    p_values = np.arange(1, 21)

    accuracy_scores = []

    # Перебір ступенів метрики Мінковського та обчислення точності
    for p in p_values:
        knn_model = KNeighborsClassifier(n_neighbors=k, p=p)
        knn_model.fit(X_train, y_train)
        accuracy = knn_model.score(X_test, y_test)
        accuracy_scores.append(accuracy)

    plt.figure(figsize=(10, 6))
    plt.plot(p_values, accuracy_scores, marker='o')
    plt.xlabel('Степінь метрики Мінковського (p)')
    plt.ylabel('Точність')
    plt.title('Вплив степеня метрики Мінковського на точність класифікації')
    plt.grid(True)
    plt.show()


def main():
    path = 'dataset3_l4.csv'
    df = loadFile(path)

    print("Task #2: ")
    print("Num of records: ", countNumOfValues(df))

    print("Task #3: ")
    print("Namings of columns: ")
    printColumnsNames(df)

    class_counts = df['NObeyesdad'].value_counts()
    print("\nЗбалансованість набору даних:")
    print(class_counts)

    print("Task #4: ")
    shuffleAndSplit(df)

    print("Task #5,6,7: ")
    buildClassificationModel(df)


main()

