from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_samples, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

print('1. Відкрити та зчитати наданий файл з даними')
df = pd.read_csv('dataset3_l5.csv', sep=';')

print('Очищення даних...')
for col, dtype in zip(df.columns, df.dtypes):
    if dtype == 'object':
        df[col] = df[col].apply(lambda x: float(x.strip().replace(',', '.')))

print('2. Визначити та вивести кількість записів.')
print(df.shape)

print('3. Видалити атрибут Concrete compressive strength.')
df = df.drop(df.columns[-1], axis=1)

print('4. Вивести атрибути, що залишилися.')
print(df.columns)

print(
'''5. Використовуючи функцію KMeans бібліотеки scikit-learn, виконати
розбиття набору даних на кластери з випадковою початковою
ініціалізацією і вивести координати центрів кластерів.
Оптимальну кількість кластерів визначити на основі початкового
набору даних трьsома різними способами:
 1) elbow method;
 2) average silhouette method;
 3) prediction strength method;
Отримані результати порівняти і пояснити, який метод дав кращий
результат і чому так (на Вашу думку).
''')

print('Elbow method')
wcss = []
cluster_list = range(1, 11)
for cluster_count in cluster_list:
    kmeans = KMeans(n_clusters=cluster_count, init='random', random_state=42)
    kmeans.fit(df)
    wcss.append(kmeans.inertia_)

plt.plot(cluster_list, wcss, marker='o')
plt.xlabel('Clusters')
plt.ylabel('Within class sum of squares')
plt.xticks(cluster_list)
plt.show()

print(
'''
Оптимальна кількость кластерів - 2, бо у відрізку від 1 до 2 кластеру
різниця між сумою квадратів найбільша серед інших відрізків.
''')

print('Average silhouette method')
negative_values = {}

for cluster_count in range(2, 11):
    kmeans = KMeans(n_clusters=cluster_count, init='random', random_state=42)
    kmeans.fit(df)
    silhouette_score_individual = silhouette_samples(df, kmeans.predict(df))

    for each_silhouette in silhouette_score_individual:
        if each_silhouette < 0:
            if cluster_count not in negative_values:
                negative_values[cluster_count] = 1
            else:
                negative_values[cluster_count] += 1
            
for key, val in negative_values.items():
    print(f'Кількість кластерів: {key}. Average silhouette score: {val}')

print('Згідно Average silhouette method маємо найменшу кількість негативних значень на кластері 8.')

print('Prediction strength method')
cluster_strength_score = {}
for cluster_count in range(2, 11):
    kmeans = KMeans(n_clusters=cluster_count, init='random', random_state=42)
    kmeans.fit(df)
    labels = kmeans.labels_
    cluster_strength_score[cluster_count] = calinski_harabasz_score(df, labels) / davies_bouldin_score(df, labels)

for key, val in cluster_strength_score.items():
    print(f'Кількість кластерів: {key}. Prediction strength score: {val}')

print(
'''
Згідно Prediction strength method маємо найбільшу кількість негативних значень на кластері 2.
Дві методи дали наоптимальнішу оцінку кількості кластерів - 2
''')

print('Визначення центрів кластерів')
kmeans = KMeans(n_clusters=2, init='random', random_state=42)
kmeans.fit(df)
centers_kmeans = kmeans.cluster_centers_
centers_kmeans = [list(centers_kmeans[0]), list(centers_kmeans[1])]
centers_kmeans

print(
'''
6. За раніш обраної кількості кластерів багаторазово проведіть
кластеризацію методом k-середніх, використовуючи для початкової
ініціалізації метод k-means++.
Виберіть найкращий варіант кластеризації. Який кількісний критерій
Ви обрали для відбору найкращої кластеризації?
''')
kmeans_inert = []
for cluster_count in range(1, 11):
    kmeans = KMeans(n_clusters=cluster_count, init='k-means++')
    kmeans.fit(df)
    kmeans_inert.append(kmeans.inertia_)

plt.plot(cluster_list, kmeans_inert)
plt.xlabel('Clusters')
plt.ylabel('Within class sum of squares')
plt.xticks(cluster_list)
plt.show()
print('Обрано Elbow method. Оптимальна кількість кластерів: 2')


print(
'''7. Використовуючи функцію AgglomerativeClustering бібліотеки scikitlearn, виконати розбиття набору даних на кластери. Кількість кластерів
 обрати такою ж самою, як і в попередньому методі. Вивести
 координати центрів кластерів.
''')

agglomerative_clustering= AgglomerativeClustering(n_clusters=2)
labels = agglomerative_clustering.fit_predict(df)
centers_agglomerative = []
df['Agglomerative'] = labels

for label in set(labels):
    x = df[df['Agglomerative'] == label].mean(axis=0)
    centers_agglomerative.append(list(x[:-1]))

df.drop('Agglomerative', axis=1)
print('Координати центрів кластерів ')
print(centers_agglomerative)

print('8. Порівняти результати двох використаних методів кластеризації.')
plt.scatter(centers_kmeans[0], centers_kmeans[1], c='blue', label='KMeans Method')
plt.scatter(centers_agglomerative[0], centers_agglomerative[1], c='yellow', label='AgglomerativeClustering Method')
plt.legend()
plt.show()