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
import shutil
import os
from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, KMeansSMOTE, ADASYN

#%%% 

classifiers = {
    "RF": RandomForestClassifier(),
    "DT": DecisionTreeClassifier(),
    "XGB": xgb.XGBClassifier(),      
    "SVM": SVC(),
    "KNN": KNeighborsClassifier(),
    "MLP": MLPClassifier()  
}

oversamplers = {
    "ROS": RandomOverSampler(),
    "SMOTE": SMOTE(),
    "BLSMOTE": BorderlineSMOTE(),
    "KMSMOTE": KMeansSMOTE(), # If there is an error you need to add this # cluster_balance_threshold=0.0000000001
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

def train_and_evaluate(classifier, oversampler, x, y):
    KFold = StratifiedKFold(n_splits=5)
    scaler = StandardScaler()

    precisions, recalls, f1_scores, auc = [], [], [], []

    for train_index, test_index in KFold.split(x, y):
        x_train_fold, x_test_fold = x[train_index], x[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        x_train_fold = scaler.fit_transform(x_train_fold)
        x_test_fold = scaler.transform(x_test_fold)

        x_train_resampled, y_train_resampled = oversampler.fit_resample(x_train_fold, y_train_fold)
        classifier.fit(x_train_resampled, y_train_resampled)

        y_test_pred = classifier.predict(x_test_fold)
        
        # Auc_score - SVM need decision_function
        if hasattr(classifier, "predict_proba"):
            y_test_proba = classifier.predict_proba(x_test_fold)[:, 1]
            auc_score = roc_auc_score(y_test_fold, y_test_proba)
            
        elif hasattr(classifier, "decision_function"):
            y_test_scores = classifier.decision_function(x_test_fold)
            auc_score = roc_auc_score(y_test_fold, y_test_scores)
        else:
            auc_score = np.nan

        precisions.append(precision_score(y_test_fold, y_test_pred, zero_division=0) )
        recalls.append(recall_score(y_test_fold, y_test_pred, zero_division=0) )
        f1_scores.append(f1_score(y_test_fold, y_test_pred, zero_division=0) )
        auc.append(auc_score)

    return np.std(precisions), np.std(recalls), np.std(f1_scores), np.std(auc)

all_results = {}

for dataset_name in datasets:
    data_path = f"D:\Google Drive\Prof. R.J. Kuo\Kevin 2023\Paper\Kevin paper\data\\{dataset_name}.csv"
    data = pd.read_csv(data_path)
    x = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    for oversampler_name, oversampler in oversamplers.items():
        for clf_name, clf in classifiers.items():
            results = []
            for i in range(10):
                test_precision, test_recall, test_f1_score, test_auc = train_and_evaluate(clf, oversampler, x, y)
                results.append([test_precision, test_recall, test_f1_score, test_auc])

            average_results = np.std(results, axis=0)
            
            combined_result = [dataset_name] + average_results.tolist()
            
            if (oversampler_name, clf_name) not in all_results:
                all_results[(oversampler_name, clf_name)] = []
            all_results[(oversampler_name, clf_name)].append(combined_result)


            results_df = pd.DataFrame(results, columns=['Precision', 'Recall', 'F1 Score', 'AUC'])
            csv_file_path = f"D:\Google Drive\Prof. R.J. Kuo\Kevin 2023\Paper\30 data Results\\{oversampler_name}_{clf_name}\\{dataset_name}_{oversampler_name}.csv"
            results_df.to_csv(csv_file_path, index=False)
            print(f"{dataset_name} use {oversampler_name}and {clf_name}The results have been saved to {csv_file_path}")

# Save the average result of 10 times
for (oversampler_name, clf_name), results in all_results.items():
    average_results_df = pd.DataFrame(results, columns=['Dataset', 'Precision', 'Recall', 'F1 Score', 'AUC'])
    average_csv_file_path = f"D:\Google Drive\Prof. R.J. Kuo\Kevin 2023\Paper\30 data Results\\{oversampler_name}_{clf_name}\\std_{oversampler_name}_{clf_name}.csv"
    
    average_results_df.to_csv(average_csv_file_path, index=False)

    
    print(f"{clf_name} 的所有數據集的平均結果已儲存到 {average_csv_file_path}")


    # Copy all average results to ALL_Results folder
    
    new_directory_path = f"D:\Google Drive\Prof. R.J. Kuo\Kevin 2023\Paper\30 data Results\\{clf_name}"

    if not os.path.exists(new_directory_path):
        os.makedirs(new_directory_path)
    destination_file_path = f"{new_directory_path}\\std_{oversampler_name}_{clf_name}.csv"

    shutil.copy(average_csv_file_path, destination_file_path)
