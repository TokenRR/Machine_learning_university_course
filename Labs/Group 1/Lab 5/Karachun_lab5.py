import warnings
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


warnings.filterwarnings('ignore')


print('\n1) Відкрити та зчитати наданий файл з даними.')
df = pd.read_csv('WQ-R.csv', delimiter=';')
print(df.head())


print(f'\n2) Визначити та вивести кількість записів.')
print(f'Записів: {df.shape[0]}, полів: {df.shape[1]}')


print(f'\n3) Видалити атрибут quality.')
df = df.drop('quality', axis=1)


print(f'\n4) Вивести атрибути, що залишилися.')
for col in df.columns:
    print(col)


print(f'''
5) Використовуючи функцію KMeans бібліотеки scikit-learn, виконати 
розбиття набору даних на кластери з випадковою початковою 
ініціалізацією і вивести координати центрів кластерів. 
Оптимальну кількість кластерів визначити на основі початкового 
набору даних трьома різними способами: 
    1) elbow method; 
    2) average silhouette method; 
    3)  prediction  strength  method
Отримані  результати  порівняти  і  пояснити,  який  метод  дав  кращий 
результат і чому так (на Вашу думку).''')
print('\nElbow method')
wcss = []
cluster_list = range(1, 11)
for cluster_count in cluster_list:
    kmeans = KMeans(n_clusters=cluster_count, init='random', random_state=42)
    kmeans.fit(df)
    wcss.append(kmeans.inertia_)
print('Найбільша різниця сум квадратів між 1-м та 2-м кластерами, це означає, що оптимальна кількість кластерів = 2')
plt.plot(cluster_list, wcss, marker='.')
plt.title('Elbow method')
plt.xlabel('Кластери')
plt.ylabel('WCSS')
plt.xticks(cluster_list)
plt.tight_layout()
plt.show()

print('\nAverage silhouette method')
n_val = {}
for cluster_count in range(2, 11):
    kmeans = KMeans(n_clusters=cluster_count, init='random', random_state=42).fit(df)
    silhouette_score_individual = silhouette_samples(df, kmeans.predict(df))

    for each_silhouette in silhouette_score_individual:
        if each_silhouette < 0:
            if cluster_count not in n_val:
                n_val[cluster_count] = 1
            else:
                n_val[cluster_count] += 1 
for key, val in n_val.items():
    print(f'Кількість кластерів = {key}. Average silhouette score = {val}')
print('Найменша кількість негативних значень при 2-х кластерах')

print('\nPrediction strength method')
cluster_strength_score = {}
for cluster_count in range(2, 11):
    kmeans = KMeans(n_clusters=cluster_count, init='random', random_state=42)
    kmeans.fit(df)
    labels = kmeans.labels_
    cluster_strength_score[cluster_count] = calinski_harabasz_score(df, labels) / davies_bouldin_score(df, labels)
for key, val in cluster_strength_score.items():
    print(f'Кількість кластерів: {key}. Prediction strength score: {round(val)}')
print('Найбільша кількість негативних значень при 3-х кластерах')

print('''
Elbow method та Average silhouette method кажуть, що оптимально розбивати на 2 кластери,
а `Prediction strength method`, що краще на 3.
Розбити на 2 кластери буде краще, тому значення 4626 і 4572 майже однакові, якщо порівнювати 4626 і 2622
''')

kmeans = KMeans(n_clusters=2, init='random', random_state=42)
kmeans.fit(df)
cluster_centers = kmeans.cluster_centers_
for i, center in enumerate(cluster_centers):
    print(f"Координати центру кластеру № {i+1}: \n{center}\n")


print(f'''\n
6) За раніш обраної кількості кластерів багаторазово проведіть 
кластеризацію методом k-середніх, використовуючи для початкової 
ініціалізації метод k-means++. 
Виберіть  найкращий  варіант  кластеризації.  Який  кількісний  критерій 
Ви обрали для відбору найкращої кластеризації?''')
best_elbow_score = float('inf')
best_avg_silhouette_score = float('-inf')
best_prediction_strength_score = float('-inf')

for cluster_count in range(2, 11):
    kmeans = KMeans(n_clusters=cluster_count, init='k-means++', random_state=42)
    kmeans.fit(df)
    labels = kmeans.labels_
    
    elbow_score = kmeans.inertia_
    avg_silhouette_score = silhouette_score(df, labels)
    prediction_strength_score = calinski_harabasz_score(df, labels) / davies_bouldin_score(df, labels)
    
    if elbow_score < best_elbow_score:
        best_elbow_score = elbow_score
        
    if avg_silhouette_score >= best_avg_silhouette_score:
        best_avg_silhouette_score = avg_silhouette_score
    
    if prediction_strength_score > best_prediction_strength_score:
        best_prediction_strength_score = prediction_strength_score

print(f'''Найкраща кластеризація з використанням:\n
Elbow method = {round(best_elbow_score)}
Average silhouette method = {round(best_avg_silhouette_score, 4)} 
Prediction strength method = {round(best_prediction_strength_score)}
''')
print('''
Найкращий метод кластеризації -- Average silhouette method, оскільки він дозволяє визначити оптимальну
кількість кластерів і вибрати найкращий варіант кластеризації на основі схожості об'єктів всередині кластеру
та відмінності між кластерами.''')


print(f'''\n
7) Використовуючи функцію AgglomerativeClustering бібліотеки scikit-
learn, виконати розбиття набору даних на кластери. Кількість кластерів 
обрати такою ж самою, як і в попередньому методі. Вивести 
координати центрів кластерів.''')
clusterer = AgglomerativeClustering(n_clusters=2)
cluster_labels = clusterer.fit_predict(df)

cluster_centers = []
for cluster_label in set(cluster_labels):
    cluster_center = df[cluster_labels == cluster_label].mean(axis=0)
    cluster_centers.append(cluster_center)

for i, cluster_center in enumerate(cluster_centers):
    print(f"\nКоординати центру кластеру № {i+1}: \n{cluster_center}")


print(f'\n8) Порівняти результати двох використаних методів кластеризації.')
pca = PCA(n_components=2)
normalized_df = MinMaxScaler().fit_transform(df)
df_pca = pca.fit_transform(normalized_df)

agglomerative_clusterer = AgglomerativeClustering(n_clusters=2)
agglomerative_labels = agglomerative_clusterer.fit_predict(df_pca)

kmeans = KMeans(n_clusters=2, init='k-means++', random_state=42).fit(df_pca)
kmeans_labels = kmeans.predict(df_pca)
centers = kmeans.cluster_centers_ 

fig, ax = plt.subplots(1, 2, figsize=(13, 7))

ax[0].scatter(df_pca[:, 0], df_pca[:, 1], c=agglomerative_labels, s=30, cmap='tab20')
ax[0].scatter(centers[:, 0], centers[:, 1], c='red', s=70)
ax[0].set_title('Agglomerative кластеризація')
ax[0].set_xlabel('Principal Component 1')
ax[0].set_ylabel('Principal Component 2')
ax[0].grid()

ax[1].scatter(df_pca[:, 0], df_pca[:, 1], c=kmeans_labels, s=30, cmap='tab20')
ax[1].scatter(centers[:, 0], centers[:, 1], c='red', s=70)
ax[1].set_title('k-means++ кластеризація')
ax[1].set_xlabel('Principal Component 1')
ax[1].set_ylabel('Principal Component 2')
ax[1].grid()

plt.tight_layout()
plt.show()
