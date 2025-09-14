#%%%
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import pandas as pd
import numpy as np
from sklearn.cluster import AffinityPropagation
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import os
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

#%%%
classifiers = {
    "RF": RandomForestClassifier(),
    "DT": DecisionTreeClassifier(),
    "SVM": SVC(probability=True),
    "XGBoost": xgb.XGBClassifier(),
    "KNN": KNeighborsClassifier(),
    "MLP": MLPClassifier()
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

# Parameter setting
# beta = 0.01
#       1      2       3         4      5       6         7      8       9      

# beta = 0.1
#       10     11      12        13     14      15        16     17      18


# alpha 0.001  0.0001  0.00001   0.001  0.0001  0.00001   0.001  0.0001  0.00001
# n_k   3                        5                        7
version_number = 1

def train_and_evaluate(classifier, x, y):
    KFold = StratifiedKFold(n_splits=5)
    scaler = StandardScaler()  # 1) Data normalization
    precisions, recalls, f1_scores, auc = [], [], [], []
    tps = []
    for train_index, test_index in KFold.split(x, y):
        x_train_fold, x_test_fold = x[train_index], x[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]
        
        x_train_fold = scaler.fit_transform(x_train_fold)
        x_test_fold = scaler.transform(x_test_fold)

        # 檢查 version_number 編號!
        alpha = 0.001          # parameter for merging the clusters (biger--> more clusters)
        beta = 0.01             # parameter for gini index. If GINI(c) >= beta, generate minority data in cluster c.
        n_k = 5                 # parameter for the number of nearest neighbors
        
        # AP-iB-SMOTE
        # STAGE 1: CLUSTERING
        # 2) Use Affinity Propagation to cluster the data.
        AP = AffinityPropagation().fit(x_train_fold)
        cluster_labels = AP.labels_

        num_AP_clusters = len(np.unique(cluster_labels))

        # 3) Cluster processing (Merge clusters)
        # Merge clusters
        # Define the merge_clusters function, which is used to merge neighboring clusters. 
        # Calculate the distance vector and perform cluster merging.
        def merge_clusters(x_train_fold, cluster_labels):

            # dist_matrix = pairwise_distances(x_train_fold)
            # linked = linkage(dist_matrix, 'ward') 
            
            compressed_distance_vector = pdist(x_train_fold)
            linked = linkage(compressed_distance_vector, 'ward')
            
            n_clusters = len(np.unique(cluster_labels)) - 1
            n_clusters = max(2, n_clusters)  
            print("n_clusters ",n_clusters)
            new_labels = fcluster(linked, n_clusters, criterion='maxclust')
            
            return new_labels

        # Calculate the initial cluster's silhouette score, prev_Sil.
        prev_Sil = metrics.silhouette_score(x_train_fold, cluster_labels, metric='euclidean')
        print(prev_Sil)
        
        # When the number of clusters is greater than 2, continuously merge the clusters and calculate the new silhouette score, new_Sil. 
        # If the difference from the previous score is less than alpha, stop merging.
        while len(np.unique(cluster_labels)) > 2:
            new_labels = merge_clusters(x_train_fold, cluster_labels)
            new_Sil = metrics.silhouette_score(x_train_fold, new_labels, metric='euclidean')
            diff = abs(new_Sil - prev_Sil) 
            print("new sill ",new_Sil)
            print("diff ",diff)
            if (diff < alpha):
                break

            prev_Sil = new_Sil  
            cluster_labels = new_labels

        final_labels = cluster_labels
        num_clusters = len(np.unique(final_labels))

        print('------------------------------------------------------')
        print('AP divided into {} groups.'.format(num_AP_clusters))
        print("After merging AP, divided into {} groups.".format(num_clusters))

        # ib-SOMTE
        # STAGE 2: GENERATE ARTIFICIAL MINORITY DATA 
        As = []
        Ad = []

        for c in range(num_clusters+1):
            
            print("\ncluster ", c)
            # 找出每一群C的資料
            cluster_indices = np.where(final_labels == c)[0]
            cluster_y = y_train_fold[cluster_indices]
            cluster_data = x_train_fold[cluster_indices]
            
            num_dat = len(cluster_y)
            
            overall_class_counts = np.unique(y_train_fold, return_counts=True)
            majority_class = overall_class_counts[0][np.argmax(overall_class_counts[1])]
            minority_class = overall_class_counts[0][np.argmin(overall_class_counts[1])]
            
            Nc = len(cluster_indices)
            Mj = np.sum(cluster_y == majority_class)
            Mi = np.sum(cluster_y == minority_class)
            
            print("Nc ", Nc)
            print("Mj ", Mj)
            print("Mi ", Mi)
            print("Target ", Mj-Mi)
            
            if Nc <= n_k:    # n_k = 3, 5, 7
                continue

            GINI_c = 1 - ((Mj/Nc)**2) - ((Mi/Nc)**2)
            
            print("Gini ",GINI_c)

            # 2. If GINI(c) >= beta, generate minority data in cluster c
            if (GINI_c >= beta):
                
                print("process in generating")

                
                knn = NearestNeighbors(n_neighbors=n_k)

                minority_indices = np.where(y_train_fold[cluster_indices] == minority_class)[0]
                minority_data = cluster_data[minority_indices]

                knn.fit(cluster_data)  

                k_neighbors = knn.kneighbors(minority_data, return_distance=False)


                labels = np.zeros(len(minority_data))


                for i in range(len(minority_data)):

                    neighbors_indices = k_neighbors[i]

                    neighbors_labels = y_train_fold[neighbors_indices]

                    if np.all(neighbors_labels == minority_class):
                        labels[i] = 2  # Safe
                    elif np.all(neighbors_labels == majority_class):
                        labels[i] = 0  # Noise
                    else:
                        labels[i] = 1  # Danger

                safe_data = minority_data[labels == 2]
                noise_data = minority_data[labels == 0]
                danger_data = minority_data[labels == 1]
                
                n_B = len(noise_data)
                n_S = len(safe_data)
                n_D = len(danger_data)
                print("n safe ",n_S)
                print("n danger ",n_D)
                print("n noise ",n_B)
                
                # 2.2
                A = Mj-Mi
                ratio = n_S/Mi
                
                t_As = int(ratio*A)
                t_Ad = A - t_As

                print("t_As ",t_As)
                print("t_Ad ",t_Ad)

                if n_S >= 2 :
                    
                    n_As = 0
                    while n_As < t_As:

                        indices = np.random.choice(safe_data.shape[0], size=2, replace=False)
                        x_i, x_j = safe_data[indices]
                        
                        x_k = x_i + np.random.rand() * (x_i - x_j)
                        
                        As.append(x_k)
                        n_As += 1
                # print("As ",As)
                print("size AS ", len(As))
                
                if n_D >= 2 :

                    for i in range(t_Ad):
                        indices = np.random.choice(danger_data.shape[0], size=2, replace=False)
                        x_i, x_j = danger_data[indices]
                    
                        x_k = x_i + 0.5*np.random.rand() * (x_i - x_j)
                    
                        Ad.append(x_k)
                
                # print("Ad ",Ad)
                print("size Ad ", len(Ad))
                

        Ad = np.array(Ad) 
        As = np.array(As)

        print('------------------------------------------------------')
        print('Safe New data',len(As))
        print('Danger New data',len(Ad))
        print('------------------------------------------------------')


        # Check if As and Ad have data
        if As.size > 0 and Ad.size > 0:
            new_data_points = np.vstack((As, Ad))
        elif As.size > 0:
            new_data_points = As
        elif Ad.size > 0:
            new_data_points = Ad
        else:
            new_data_points = np.array([])

        # Merge new data points with original data if new data points are available
        if new_data_points.size > 0:
            new_x_train_fold = np.vstack((x_train_fold, new_data_points))
            new_y_labels = np.full(new_data_points.shape[0], minority_class)
            new_y_train_fold = np.concatenate((y_train_fold, new_y_labels))
        else:
            # If no new data points, use original data
            new_x_train_fold = x_train_fold
            new_y_train_fold = y_train_fold


        classifier.fit(new_x_train_fold, new_y_train_fold)

        y_test_pred = classifier.predict(x_test_fold)

        # Confusion Matrix
        cm = confusion_matrix(y_test_fold, y_test_pred)
        tp = cm[1, 1]  
        tps.append(tp)  
        print(f'Confusion Matrix:\n{cm}')
        print(f'True Positives (TP): {tp}')  
        # 顯示分類報告
        print(classification_report(y_test_fold, y_test_pred))

        # SVM need decision_function
        if hasattr(classifier, "predict_proba"):
            y_test_proba = classifier.predict_proba(x_test_fold)[:, 1]
            auc_score = roc_auc_score(y_test_fold, y_test_proba)
        elif hasattr(classifier, "decision_function"):
            y_test_scores = classifier.decision_function(x_test_fold)
            auc_score = roc_auc_score(y_test_fold, y_test_scores)
        else:
            auc_score = np.nan

        precisions.append(precision_score(y_test_fold, y_test_pred, zero_division=0))
        recalls.append(recall_score(y_test_fold, y_test_pred, zero_division=0))
        f1_scores.append(f1_score(y_test_fold, y_test_pred, zero_division=0))
        auc.append(auc_score)

    return np.mean(precisions), np.mean(recalls), np.mean(f1_scores), np.mean(auc)

all_average_results = {clf_name: [] for clf_name in classifiers}

for dataset_name in datasets:
    data_path = f"C:\\Users\\USER\\Documents\\Kevin\\Wistron data\\{dataset_name}.csv"
    data = pd.read_csv(data_path)
    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    for clf_name, clf in classifiers.items():
        results = []
        for i in range(10):
            test_precision, test_recall, test_f1_score, test_auc = train_and_evaluate(clf, x, y)
            results.append([test_precision, test_recall, test_f1_score, test_auc])

        average_results = np.mean(results, axis=0)
        all_average_results[clf_name].append([dataset_name.replace('.csv', ''), *average_results])

        csv_file_path = f"C:\\Users\\USER\\Documents\\Kevin\\AP_ib_SMOTE_Wistron_{version_number}\\{clf_name}\\{dataset_name}.csv"
        
        directory_path = os.path.dirname(csv_file_path)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)
        
    
        results_df = pd.DataFrame(results, columns=['Precision', 'Recall', 'F1 Score', 'AUC'])
        results_df.to_csv(csv_file_path, index=False)
        print(f"{dataset_name} and {clf_name} The results have been saved to {csv_file_path}")

for clf_name in classifiers:
    average_results_df = pd.DataFrame(all_average_results[clf_name], columns=['Dataset', 'Precision', 'Recall', 'F1 Score', 'AUC'])
    average_csv_file_path = f"C:\\Users\\USER\\Documents\\Kevin\\AP_ib_SMOTE_Wistron_{version_number}\\{clf_name}\\Average_{clf_name}.csv"
    average_directory_path = os.path.dirname(average_csv_file_path)
    if not os.path.exists(average_directory_path):
        os.makedirs(average_directory_path, exist_ok=True)

    average_results_df.to_csv(average_csv_file_path, index=False)
    print(f"The average results across all datasets for {clf_name} have been stored to {average_csv_file_path}")

# %%
