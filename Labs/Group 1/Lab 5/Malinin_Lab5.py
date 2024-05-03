import warnings
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_samples, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


print('\nЗавдання 1')
df = pd.read_csv('WQ-R.csv', delimiter=';')
pd.set_option('display.max_columns', None)
warnings.filterwarnings('ignore')
print(df.head())


print(f'\nЗавдання 2')
print(f'Записів: {df.shape[0]}')


print(f'\nЗавдання 3')
df = df.drop('quality', axis=1)
print('Атрибут `quality` видалено')


print(f'\nЗавдання 4')
for col in df.columns:
    print(col)


print(f'\nЗавдання 5')
print('Elbow method')
wcss = []
cluster_list = range(1, 7)
for cluster_count in cluster_list:
    kmeans = KMeans(n_clusters=cluster_count, init='random', random_state=42)
    kmeans.fit(df)
    wcss.append(kmeans.inertia_)
print('Найбільша різниця сум квадратів між 1 та 2 кластерами. Тому оптимальна кількість кластерів = 2')
plt.plot(cluster_list, wcss, marker='o', color='red')
plt.title('Elbow method')
plt.xlabel('Кластери')
plt.ylabel('Сума квадратів')
plt.xticks(cluster_list)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

print('\n\nAverage silhouette method')
n_val = {}
for cluster_count in range(2, 7):
    kmeans = KMeans(n_clusters=cluster_count, init='random', random_state=42)
    kmeans.fit(df)
    silhouette_score_individual = silhouette_samples(df, kmeans.predict(df))

    for each_silhouette in silhouette_score_individual:
        if each_silhouette < 0:
            if cluster_count not in n_val:
                n_val[cluster_count] = 1
            else:
                n_val[cluster_count] += 1 
for key, val in n_val.items():
    print(f'Кількість кластерів = {key}. Оцінка Average silhouette = {val}')
print('Найменша кількість негативних значень при 2-х кластерах')

print('\n\nPrediction strength method')
cluster_strength_score = {}
for cluster_count in range(2, 7):
    kmeans = KMeans(n_clusters=cluster_count, init='random', random_state=42)
    kmeans.fit(df)
    labels = kmeans.labels_
    cluster_strength_score[cluster_count] = calinski_harabasz_score(df, labels) / davies_bouldin_score(df, labels)
for key, val in cluster_strength_score.items():
    print(f'Кількість кластерів: {key}. Prediction strength score: {round(val)}')
print('Найбільша кількість негативних значень при 3-х кластерах')

print('''\n\nВисновок: \n\
\tElbow method. Оптимально - 2 кластери
\tAverage silhouette method - 2 кластери
\tPrediction strength method - 3 кластери
2-і оцінки з 3-х кажуть про те, що оптимально розбивати на 2 кластери, а `Prediction strength method`, що краще на 3.
На мою думку, краще розбивати на 2, оскільки при оцінці `Prediction strength method` значення між 2-а та 3-акластерами дуже близькі
''')

print('\nКоординати центрів кластерів:')
kmeans = KMeans(n_clusters=2, init='random', random_state=42)
kmeans.fit(df)
cluster_centers = kmeans.cluster_centers_
for center in cluster_centers:
    print(center)


print(f'\nЗавдання 6')
best_cluster_count = 0
best_elbow_score = float('inf')
best_avg_silhouette_score = float('-inf')
best_prediction_strength_score = float('-inf')

for cluster_count in range(2, 7):
    kmeans = KMeans(n_clusters=cluster_count, init='k-means++', random_state=42)
    kmeans.fit(df)
    labels = kmeans.labels_
    
    elbow_score = kmeans.inertia_
    avg_silhouette_score = silhouette_score(df, labels)
    prediction_strength_score = calinski_harabasz_score(df, labels) / davies_bouldin_score(df, labels)
    
    if elbow_score < best_elbow_score:
        best_elbow_score = elbow_score
        best_cluster_count_elbow = cluster_count
        
    if avg_silhouette_score >= best_avg_silhouette_score:
        best_avg_silhouette_score = avg_silhouette_score
        best_cluster_count_avg_silhouette = cluster_count
    
    if prediction_strength_score > best_prediction_strength_score:
        best_prediction_strength_score = prediction_strength_score
        best_cluster_count_prediction_strength = cluster_count

print(f'''Найкраща кластеризація з використанням:\n
\tElbow method: найкращий показник = {round(best_elbow_score)}
\tAverage silhouette method: найкращий  показник = {round(best_avg_silhouette_score, 4)} 
\tPrediction strength method: найкращий показник = {round(best_prediction_strength_score)}''')
print('''
Найкращий метод кластеризації залежить від конкретної мети дослідження. Проте мені більше до вподоби\
`Average silhouette method` оскільки він є стійким до викидів і шуму''')


print(f'\nЗавдання 7')
clusterer = AgglomerativeClustering(n_clusters=2)
cluster_labels = clusterer.fit_predict(df)

cluster_centers = []
for cluster_label in set(cluster_labels):
    cluster_center = df[cluster_labels == cluster_label].mean(axis=0)
    cluster_centers.append(cluster_center)

for i, cluster_center in enumerate(cluster_centers):
    print(f"Координати центру кластеру {i+1}: \n{cluster_center}")


print(f'\nЗавдання 8')
print('*Графік*')

pca = PCA(n_components=2)
normalized_df = MinMaxScaler().fit_transform(df)
df_pca = pca.fit_transform(normalized_df)

agglomerative_clusterer = AgglomerativeClustering(n_clusters=2)
agglomerative_labels = agglomerative_clusterer.fit_predict(df_pca)

kmeans = KMeans(n_clusters=2, init='k-means++', random_state=42).fit(df_pca)
kmeans_labels = kmeans.predict(df_pca)
centers = kmeans.cluster_centers_ 

fig, ax = plt.subplots(1, 2, figsize=(13, 7))

ax[0].scatter(df_pca[:, 0], df_pca[:, 1], c=agglomerative_labels)
ax[0].scatter(centers[:, 0], centers[:, 1], c='red')
ax[0].set_title('Agglomerative Clustering')
ax[0].set_xlabel('Principal Component 1')
ax[0].set_ylabel('Principal Component 2')
ax[0].grid()

ax[1].scatter(df_pca[:, 0], df_pca[:, 1], c=kmeans_labels)
ax[1].scatter(centers[:, 0], centers[:, 1], c='red')
ax[1].set_title('k-means++ Clustering')
ax[1].set_xlabel('Principal Component 1')
ax[1].set_ylabel('Principal Component 2')
ax[1].grid()

plt.tight_layout()
plt.show()
