import warnings
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import silhouette_score, pairwise_distances, davies_bouldin_score, calinski_harabasz_score


pd.set_option('display.max_columns', None)
warnings.filterwarnings("ignore")


def rename_columns(column_name):
    '''Функція перейменування колонок яка заміняє пробіли на `_`, видаляє дужки та їх вміст'''
    new_name = column_name.replace(' ', '_').split('(')[0].rstrip('_')
    return new_name


def elbow_method(data, k_range, init_method):
    '''Допоможна функція для обрахування ліктьового методу'''
    optimal_clusters['elbow'] = list()  # очищуємо минулі значення метрик
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, init=init_method, n_init=10).fit(data)
        optimal_clusters['elbow'].append(kmeans.inertia_)   # додаємо WCSS або "інерцію"


def silhouette_method(data, k_range, init_method, p=2):
    '''Допоможна функція для обрахування методу середнього силуету'''
    optimal_clusters['silhouette'] = list()     # очищуємо минулі значення метрик
    if p == 2:                          # якщо "p = 2 - метрика Мінковського є Евклідовою
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, init=init_method, n_init=10).fit(data)
            optimal_clusters['silhouette'].append(silhouette_score(data, kmeans.labels_, metric='euclidean', random_state=42))
    else:
        minkowski = pairwise_distances(data, metric='minkowski', p=p)   # вручну рахуємо відстані з визначеним "p"
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, init=init_method, n_init=10).fit(data)
            optimal_clusters['silhouette'].append(silhouette_score(minkowski, kmeans.labels_, random_state=42))


def ps_method(data, k_range, init_method, n_boot):
    '''Допоміжна функція для обрахування методу прогнозуючої сили'''
    optimal_clusters['ps'] = list()     # очищуємо минулі значення метрик

    for k in k_range:
        clusterings = list()
        for _ in range(n_boot):     # n_boot - кількість бутстрап вибірок
            resampled_data = resample(data, replace=True, n_samples=data.shape[0])
            kmeans = KMeans(n_clusters=k, random_state=42, init=init_method, n_init=10).fit(resampled_data)
            clusterings.append(kmeans.labels_)

        ps = pairwise_distances(np.array(clusterings), metric='hamming').mean()     # рахуємо PS для кожної к-сті підвибірок
        optimal_clusters['ps'].append(1 - ps)


def compute_methods(data, k_max, method):
    '''Обрахування трьох метрик з заданими параметрами'''
    elbow_method(data, range(2, k_max), method)
    silhouette_method(data, range(2, k_max), method)
    ps_method(data, range(2, k_max), method, 2)


def compute_cluster_centers(data, labels, num_clusters):
    '''Визначення центру кластера
       P.S. AgglomerativeClustering не передбачує атрибут .clusters_)'''
    cluster_centers = list()
    for cluster in range(num_clusters):
        cluster_samples = data[np.where(labels == cluster)[0]]
        cluster_center = np.mean(cluster_samples, axis=0)
        cluster_centers.append(cluster_center)
    return np.array(cluster_centers)


if __name__ == '__main__':
    data_scale = ['no scale', 'normalized', 'standardized']  #  labels для візуалізації різниці масштабувань даних
    colors = ['red', 'blue', 'green']
    ylabels = ['WCSS', 'Silhouette score', 'Prediction strength']  #  labels для візуалізації методів
    optimal_clusters = {'elbow': list(), 'silhouette': list(), 'ps': list()}  #  dict для зберігання метрик
    k_max = 6

    print('\n---  1  ---')
    df = pd.read_csv('dataset3_l5.csv', sep=';')
    print('Дані імпортовано, тепер їх треба очистити, оскільки числа містять кому замість крапки')
    print('Наприклад: 504,09 => 504.09')
    for col, dtype in zip(df.columns, df.dtypes):
        if dtype == 'object':
            df[col] = df[col].apply(lambda x: float(x.strip().replace(',', '.')))
    print('Дані очищено, але назви містять пробіли та довгі назви. Пропоную переназвати колонки\n')

    df.columns = df.columns.map(rename_columns)
    print(df.head())
    print('---  1  ---\n')


    print('\n---  2  ---')
    print(f'Записів: {df.shape[0]}, полів (до видалення `Concrete compressive strength`): {df.shape[1]}')
    print('---  2  ---\n')


    print('\n---  3  ---')
    df = df.drop('Concrete_compressive_strength', axis=1)
    print(f'Полів (після видалення `Concrete compressive strength`): {df.shape[1]}')
    print('---  3  ---\n')


    print('\n---  4  ---')
    for name in df.columns:
        print(name)
    print('---  4  ---\n')


    print('\n---  5  ---')
    normalized_df = MinMaxScaler().fit_transform(df)
    standardized_df = StandardScaler().fit_transform(df)
    
    print('*Графік для Elbow method*')
    print('Оптимально розбити на 2 кластери, тому що ми можемо бачити, що найбільша різниця WCSS досягається між 1 та 2-а кластерами')
    fig, ax = plt.subplots(3, 1, figsize=(8, 6))
    for i, k, color, scale in zip(range(3), [df, normalized_df, standardized_df], colors, data_scale):
        elbow_method(k, range(1, k_max), 'random')
        ax[i].plot(list(range(1, k_max)), optimal_clusters['elbow'], 
                   marker='o', markersize=6, label=scale, color=color)
        ax[i].set_ylabel('WCSS')
        ax[i].set_xlabel('N clusters')
        ax[i].set_title('WCSS (inertia) vs Number of clusters')
        ax[i].set_ylim(None, 1.1 * np.max(optimal_clusters['elbow']))
        ax[i].set_xticks(range(1, k_max))
        ax[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax[i].grid()
    plt.tight_layout()
    plt.show()

    print('\n*Графік для Average silhouette method*')
    print('Також оптимально розбити на 2 кластери, тому що НАЙМЕНШЕ значення(краще) досягається саме при 2-х')
    for data, color, scale in zip([df, normalized_df, standardized_df], colors, data_scale):
        compute_methods(data, k_max, 'random')
        plt.plot(list(range(2, k_max)), optimal_clusters['silhouette'],
                 marker='o', markersize=6, color=color, label=scale)
    plt.ylabel(ylabels[1])
    plt.xlabel('N clusters')
    plt.title('Average silhouette method')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(range(2, k_max))
    plt.grid()
    plt.tight_layout()
    plt.show()

    print('\n*Графік для Prediction strength method*')
    print('Також оптимально розбити на 2 кластери, тому що НАЙБІЛЬШЕ значення(краще) досягається саме при 2-х')
    for data, color, scale in zip([df, normalized_df, standardized_df], colors, data_scale):
        compute_methods(data, k_max, 'random')
        plt.plot(list(range(2, k_max)), optimal_clusters['ps'],
                 marker='o', markersize=6, color=color, label=scale)
    plt.ylabel(ylabels[1])
    plt.xlabel('N clusters')
    plt.title('Prediction strength method')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(range(2, k_max))
    plt.grid()
    plt.tight_layout()
    plt.show()

    print('''\nВисновки:
    Всі методи дали однаковий результат, проте, я більш схильний до методу середнього силуету та
    прогнозуючої сили, оскільки вони враховують як компактність кластерів (Average silhouette), так і 
    стійкість кластерної структури (Prediction strength), що дає більш об'єктивну оцінку кількості кластерів''')
    print('---  5  ---\n')
    time.sleep(1.5)


    print('\n---  6  ---')
    N = 2           # визначена раніше оптимальна кількість кластерів
    n_iter = 15     # кількість ітерацій кластеризації
    best_elbow_score = float('inf')
    best_avg_silhouette_score = float('-inf')
    best_prediction_strength_score = float('-inf')

    for cluster_count in range(2, n_iter):
        kmeans = KMeans(n_clusters=N, init='k-means++', random_state=42).fit(normalized_df)
        labels = kmeans.labels_
        
        elbow_score = kmeans.inertia_
        avg_silhouette_score = silhouette_score(normalized_df, labels)
        prediction_strength_score = calinski_harabasz_score(normalized_df, labels) / \
                                    davies_bouldin_score(normalized_df, labels)
        
        if elbow_score < best_elbow_score:
            best_elbow_score = elbow_score
            
        if avg_silhouette_score >= best_avg_silhouette_score:
            best_avg_silhouette_score = avg_silhouette_score
        
        if prediction_strength_score > best_prediction_strength_score:
            best_prediction_strength_score = prediction_strength_score

    print(f'Найкраща кластеризація з використанням:')
    print(f'Elbow method = {round(best_elbow_score)}')
    print(f'Average silhouette method (*1000) = {round(best_avg_silhouette_score*1000)}')
    print(f'Prediction strength method = {round(best_prediction_strength_score)}')
    print('\nЦі результати кажуть нам про те що `Elbow method` є найкращим у даному випадку.')
    print('Проте вибір методу залежить від багатьох факторів і задач, тому `Elbow method` не завжди буде кращим варіантом')
    print('---  6  ---\n')
    time.sleep(1.5)


    print('\n---  7  ---')
    print('Кількість кластерів = 2')
    print('Як можемо бачити - координату центру це не просто точка, а точка у 8-вимірному просторі')

    aggcluster = AgglomerativeClustering(n_clusters=2)
    y_aggcluster = aggcluster.fit_predict(normalized_df)
    agg_centers = compute_cluster_centers(normalized_df, y_aggcluster, 2) 
    for i in [1, 2]:
        print(f'\nAgglomerativeClustering. Координати центру кластеру {i}: \n{agg_centers}')
    print('---  7  ---\n')
    time.sleep(1.5)

    print('\n---  8  ---')
    print('Для того щоб порівняти кластери отримані різними методами, можемо їх намалювати')
    print('Для того щоб намалювати використаємо PCA - метод головних компонент\n')

    pca = PCA(n_components=2)
    df_pca = pca.fit_transform(normalized_df)
    fig, ax = plt.subplots(1, 3, figsize=(13, 5))

    kmeans = KMeans(n_clusters=2, random_state=42, init='k-means++', n_init=10).fit(df_pca)
    y_kmeans = kmeans.predict(df_pca)
    centers = kmeans.cluster_centers_

    aggcluster_ward = AgglomerativeClustering(n_clusters=2, linkage='ward').fit_predict(df_pca)
    agg_centers_ward = compute_cluster_centers(df_pca, aggcluster_ward, 2)

    aggcluster_complete = AgglomerativeClustering(n_clusters=2, linkage='complete').fit_predict(df_pca)
    agg_centers_complete = compute_cluster_centers(df_pca, aggcluster_complete, 2)

    print(f'Центри кластерів K-Means: k-means++: \n{centers}\n')
    print(f'Центри кластерів AgglomerativeClustering: ward: \n{agg_centers_ward}\n')
    print(f'Центри кластерів AgglomerativeClustering: complete: \n{agg_centers_complete}')

    ax[0].scatter(df_pca[:, 0], df_pca[:, 1], c=y_kmeans, s=30, cmap='Accent')
    ax[0].scatter(centers[:, 0], centers[:, 1], c='red', s=100)
    ax[0].set_title('K-Means: k-means++')

    ax[1].scatter(df_pca[:, 0], df_pca[:, 1], c=aggcluster_ward, s=30, cmap='Accent')
    ax[1].scatter(agg_centers_ward[:, 0], agg_centers_ward[:, 1], c='red', s=100)
    ax[1].set_title('AgglomerativeClustering: ward')

    ax[2].scatter(df_pca[:, 0], df_pca[:, 1], c=aggcluster_complete, s=30, cmap='Accent')
    ax[2].scatter(agg_centers_complete[:, 0], agg_centers_complete[:, 1], c='red', s=100)
    ax[2].set_title(f'AgglomerativeClustering: complete')
    
    for i in [0, 1, 2]:
        ax[i].grid()
    plt.tight_layout()
    plt.show()
    print('---  8  ---\n')
