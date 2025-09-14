import time
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, KMeansSMOTE, ADASYN

#%%%
oversamplers = {
    "ROS": RandomOverSampler(),
    "SMOTE": SMOTE(),
    "BLSMOTE": BorderlineSMOTE(),
    "KMSMOTE": KMeansSMOTE(cluster_balance_threshold=0.0000000001),
    "ADASYN": ADASYN()
}

datasets = [
    "ecoli-0-1_vs_5",
    "ecoli-0-2-3-4_vs_5",
    "ecoli-0-3-4-6_vs_5",
    "ecoli-0-3-4_vs_5",
    "ecoli-0-6-7_vs_5",
    "ecoli2",
    "glass1",
    "glass2",
    "haberman",
    "led7digit-0-2-4-5-6-7-8-9_vs_1",
    "paw02a-600-5-30-BI",
    "Pima",
    "vehicle1",
    "vehicle2",
    "vehicle3",
    "vowel0",
    "yeast-0-2-5-6_vs_3-7-8-9",
    "yeast-0-2-5-7-9_vs_3-6-8",
    "yeast-0-3-5-9_vs_7-8",
    "yeast-0-5-6-7-9_vs_4",
    "yeast-1-2-8-9_vs_7",
    "yeast-1_vs_7",
    "yeast-2_vs_4",
    "yeast-2_vs_8",
    "yeast3",
    "yeast4",
    "yeast5",
]

def measure_oversampler_time(oversampler, datasets):
    total_time = 0
    for dataset_name in datasets:
        data_path = f"C:\\Users\\USER\\Documents\\Kevin\\data\\{dataset_name}.csv"
        data = pd.read_csv(data_path)
        x = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        start_time = time.time()
        oversampler.fit_resample(x, y)
        end_time = time.time()
        total_time += end_time - start_time
    return total_time

oversampler_times = {}

for oversampler_name, oversampler in oversamplers.items():
    total_time_taken = measure_oversampler_time(oversampler, datasets)
    oversampler_times[oversampler_name] = total_time_taken

# Save the results
results = [(name, time) for name, time in oversampler_times.items()]
results_df = pd.DataFrame(results, columns=['Oversampler', 'Total Time (seconds)'])
csv_file_path = f"C:\\Users\\USER\\Documents\\Kevin\\Computational time\\Total_Computational_Time.csv"
results_df.to_csv(csv_file_path, index=False)
print(f"所有 oversamplers 的計算時間已儲存到 {csv_file_path}")

#%% AP-iB-SMOTE
import time
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import os

datasets = [
    "ecoli-0-1_vs_5",
    "ecoli-0-2-3-4_vs_5",
    "ecoli-0-3-4-6_vs_5",
    "ecoli-0-3-4_vs_5",
    "ecoli-0-6-7_vs_5",
    "ecoli2",
    "glass1",
    "glass2",
    "haberman",
    "led7digit-0-2-4-5-6-7-8-9_vs_1",
    "paw02a-600-5-30-BI",
    "Pima",
    "vehicle1",
    "vehicle2",
    "vehicle3",
    "vowel0",
    "yeast-0-2-5-6_vs_3-7-8-9",
    "yeast-0-2-5-7-9_vs_3-6-8",
    "yeast-0-3-5-9_vs_7-8",
    "yeast-0-5-6-7-9_vs_4",
    "yeast-1-2-8-9_vs_7",
    "yeast-1_vs_7",
    "yeast-2_vs_4",
    "yeast-2_vs_8",
    "yeast3",
    "yeast4",
    "yeast5",
]

version_number = 1

def ap_ib_smote(x_train_fold, y_train_fold):
    scaler = StandardScaler()  # 1) Data normalization
    x_train_fold = scaler.fit_transform(x_train_fold)

    # AP-iB-SMOTE parameters
    alpha = 0.001          # parameter for merging the clusters (biger--> more clusters)
    beta = 0.01            # parameter for gini index. If GINI(c) >= beta, generate minority data in cluster c.
    n_k = 5                # parameter for the number of nearest neighbors

    # AP-iB-SMOTE
    # STAGE 1: CLUSTERING
    # Use Affinity Propagation to cluster the data.
    AP = AffinityPropagation().fit(x_train_fold)
    cluster_labels = AP.labels_

    num_AP_clusters = len(np.unique(cluster_labels))

    # Merge clusters
    def merge_clusters(x_train_fold, cluster_labels):
        compressed_distance_vector = pdist(x_train_fold)
        linked = linkage(compressed_distance_vector, 'ward')
        
        n_clusters = len(np.unique(cluster_labels)) - 1
        n_clusters = max(2, n_clusters)  
        new_labels = fcluster(linked, n_clusters, criterion='maxclust')
        
        return new_labels

    prev_Sil = metrics.silhouette_score(x_train_fold, cluster_labels, metric='euclidean')
    
    while len(np.unique(cluster_labels)) > 2:
        new_labels = merge_clusters(x_train_fold, cluster_labels)
        new_Sil = metrics.silhouette_score(x_train_fold, new_labels, metric='euclidean')
        diff = abs(new_Sil - prev_Sil)
        if (diff < alpha):
            break

        prev_Sil = new_Sil  
        cluster_labels = new_labels

    final_labels = cluster_labels
    num_clusters = len(np.unique(final_labels))

    # STAGE 2: GENERATE ARTIFICIAL MINORITY DATA 
    As = []
    Ad = []

    for c in range(num_clusters + 1):
        cluster_indices = np.where(final_labels == c)[0]
        cluster_y = y_train_fold[cluster_indices]
        cluster_data = x_train_fold[cluster_indices]

        Nc = len(cluster_indices)
        Mj = np.sum(cluster_y == np.bincount(y_train_fold).argmax())
        Mi = Nc - Mj
        
        if Nc <= n_k:
            continue

        GINI_c = 1 - ((Mj/Nc)**2) - ((Mi/Nc)**2)

        if (GINI_c >= beta):
            knn = NearestNeighbors(n_neighbors=n_k)
            minority_indices = np.where(y_train_fold[cluster_indices] == 1)[0]
            minority_data = cluster_data[minority_indices]
            knn.fit(cluster_data)
            k_neighbors = knn.kneighbors(minority_data, return_distance=False)

            labels = np.zeros(len(minority_data))

            for i in range(len(minority_data)):
                neighbors_indices = k_neighbors[i]
                neighbors_labels = y_train_fold[neighbors_indices]

                if np.all(neighbors_labels == 1):
                    labels[i] = 2  # Safe
                elif np.all(neighbors_labels == 0):
                    labels[i] = 0  # Noise
                else:
                    labels[i] = 1  # Danger

            safe_data = minority_data[labels == 2]
            danger_data = minority_data[labels == 1]

            t_As = int((len(safe_data) / Mi) * (Mj - Mi))
            t_Ad = (Mj - Mi) - t_As

            if len(safe_data) >= 2:
                for _ in range(t_As):
                    indices = np.random.choice(len(safe_data), size=2, replace=False)
                    x_i, x_j = safe_data[indices]
                    x_k = x_i + np.random.rand() * (x_j - x_i)
                    As.append(x_k)

            if len(danger_data) >= 2:
                for _ in range(t_Ad):
                    indices = np.random.choice(len(danger_data), size=2, replace=False)
                    x_i, x_j = danger_data[indices]
                    x_k = x_i + 0.5 * np.random.rand() * (x_j - x_i)
                    Ad.append(x_k)

    As = np.array(As) if As else np.array([])  # Ensure As is a numpy array
    Ad = np.array(Ad) if Ad else np.array([])  # Ensure Ad is a numpy array

    new_data_points = np.vstack((As, Ad)) if As.size > 0 and Ad.size > 0 else (As if As.size > 0 else Ad)
    if new_data_points.size > 0:
        new_x_train_fold = np.vstack((x_train_fold, new_data_points))
        new_y_train_fold = np.concatenate((y_train_fold, np.full(new_data_points.shape[0], 1)))
    else:
        new_x_train_fold = x_train_fold
        new_y_train_fold = y_train_fold

    return new_x_train_fold, new_y_train_fold

total_time = 0

for dataset_name in datasets:
    data_path = f"C:\\Users\\USER\\Documents\\Kevin\\data\\{dataset_name}.csv"
    data = pd.read_csv(data_path)
    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    start_time = time.time()  # Start time measurement
    new_x, new_y = ap_ib_smote(x, y)  # Run AP-iB-SMOTE once
    end_time = time.time()  # End time measurement
    elapsed_time = end_time - start_time
    total_time += elapsed_time

# Save total computation time
total_time_df = pd.DataFrame([['AP-iB-SMOTE', total_time]], columns=['Algorithm', 'Total Time (seconds)'])
total_time_csv_file_path = f"C:\\Users\\USER\\Documents\\Kevin\\Computational time\\Total_Computation_Time.csv"
total_time_df.to_csv(total_time_csv_file_path, index=False)
print(f"AP-iB-SMOTE 的總計算時間已儲存到 {total_time_csv_file_path}")

# %%
