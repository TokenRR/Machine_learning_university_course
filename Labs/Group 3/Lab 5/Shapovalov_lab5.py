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


print('\n---  Завдання 1  ---')
df = pd.read_csv('dataset3_l5.csv', delimiter=';')
for col, dtype in zip(df.columns, df.dtypes):
    if dtype == 'object':
        df[col] = df[col].apply(lambda x: float(x.strip().replace(',', '.')))
print(df.head())
print('---  Завдання 1  ---\n')


print('\n---  Завдання 2  ---')
print(f'Записів: {df.shape[0]}')
print('---  Завдання 2  ---\n')


print('\n---  Завдання 3  ---')
df = df.drop(df.columns[-1], axis=1)
print('---  Завдання 3  ---\n')


print('\n---  Завдання 4  ---')
for col in df.columns:
    print(col)
print('---  Завдання 4  ---\n')


print('\n---  Завдання 5  ---')
print('Elbow method')
wcss = []
cluster_list = range(1, 11)
for cluster_count in cluster_list:
    kmeans = KMeans(n_clusters=cluster_count, init='random', random_state=10)
    kmeans.fit(df)
    wcss.append(kmeans.inertia_)
print('Найбільша різниця сум квадратів між 1-м та 2-м кластерами, це означає, що оптимальна кількість кластерів = 2')
plt.plot(cluster_list, wcss, marker='o')
plt.title('Elbow method')
plt.xlabel('Кластери')
plt.ylabel('WCSS')
plt.xticks(cluster_list)
plt.tight_layout()
plt.show()

print('\nAverage silhouette method')
n_val = {}
for cluster_count in range(2, 9):
    kmeans = KMeans(n_clusters=cluster_count, init='random', random_state=10).fit(df)
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
for cluster_count in range(2, 9):
    kmeans = KMeans(n_clusters=cluster_count, init='random', random_state=10)
    kmeans.fit(df)
    labels = kmeans.labels_
    cluster_strength_score[cluster_count] = calinski_harabasz_score(df, labels) / davies_bouldin_score(df, labels)
for key, val in cluster_strength_score.items():
    print(f'Кількість кластерів: {key}. Prediction strength score: {round(val)}')
print('Найбільша кількість негативних значень при 3-х кластерах')

print('\nКраще розбити на 2 кластери, бо два методи з трьох дають оптимальну оцінку саме на такій кількості')

kmeans = KMeans(n_clusters=2, init='random', random_state=10)
kmeans.fit(df)
cluster_centers = kmeans.cluster_centers_
for i, center in enumerate(cluster_centers):
    print(f"\nКоординати центру кластеру {i+1}: \n{center}")
print('---  Завдання 5  ---\n')


print('\n---  Завдання 6  ---')
best_elbow_score = float('inf')
best_avg_silhouette_score = float('-inf')
best_prediction_strength_score = float('-inf')

for cluster_count in range(2, 11):
    kmeans = KMeans(n_clusters=cluster_count, init='k-means++', random_state=10)
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

print(f'Найкраща кластеризація з використанням:')
print(f'Elbow method = {round(best_elbow_score)}')
print(f'Average silhouette method = {round(best_avg_silhouette_score, 4)}')
print(f'Prediction strength method = {round(best_prediction_strength_score)}')

print('\nElbow method - найкращий метод кластеризації в цьому випадку, тому що він дає хорошу оцінку')
print('Але це не означає, що він буде хорош всюди. Для різних задач підходять різні методи')

print('---  Завдання 6  ---\n')


print('\n---  Завдання 7  ---')
clusterer = AgglomerativeClustering(n_clusters=2)
cluster_labels = clusterer.fit_predict(df)

cluster_centers = []
for cluster_label in set(cluster_labels):
    cluster_center = df[cluster_labels == cluster_label].mean(axis=0)
    cluster_centers.append(cluster_center)

for i, cluster_center in enumerate(cluster_centers):
    print(f"\nКоординати центру кластеру {i+1}: \n{cluster_center}")
print('---  Завдання 7  ---\n')


print('\n---  Завдання 8  ---')
print('Нормалізуємо датасет для побудови графіків')
pca = PCA(n_components=2)
normalized_df = MinMaxScaler().fit_transform(df)
df_pca = pca.fit_transform(normalized_df)

agglomerative_clusterer = AgglomerativeClustering(n_clusters=2)
agglomerative_labels = agglomerative_clusterer.fit_predict(df_pca)

kmeans = KMeans(n_clusters=2, init='k-means++', random_state=10).fit(df_pca)
kmeans_labels = kmeans.predict(df_pca)
centers = kmeans.cluster_centers_ 

plt.scatter(df_pca[:, 0], df_pca[:, 1], c=agglomerative_labels)
plt.scatter(centers[:, 0], centers[:, 1], c='red')
plt.title('Agglomerative кластеризація')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid()
plt.tight_layout()
plt.show()

plt.scatter(df_pca[:, 0], df_pca[:, 1], c=kmeans_labels)
plt.scatter(centers[:, 0], centers[:, 1], c='red')
plt.title('k-means++ кластеризація')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid()

plt.tight_layout()
plt.show()
print('---  Завдання 8  ---\n')
