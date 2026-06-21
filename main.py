import os
import math
import sys
import pandas as pd
import numpy as np

FILEPATH = "student_lifestyle_dataset.csv"
TARGET_COL = "Stress_Level"
RANDOM_SEED = 42
THRESHOLD_OFFSET = -0.5


def load_and_preprocess(filepath):
    # Membaca dataset, download otomatis jika belum ada, dan imputasi nilai kosong (modus & mean).
    print(f"\n[STEP 1] Memuat dataset dari '{filepath}'...")
    if not os.path.exists(filepath):
        print(f"  [INFO] Mengunduh dataset otomatis...")
        try:
            import kagglehub
            import shutil
            download_path = kagglehub.dataset_download(
                "sridevilavanyacse/student-lifestyle-and-stress-prediction-dataset"
            )
            shutil.copy(os.path.join(download_path, "student-lifestyle-and-stress-dataset.csv"), filepath)
            print("  Dataset berhasil diunduh.")
        except Exception as e:
            print(f"  [ERROR] Gagal download dataset: {e}")
            sys.exit(1)

    df = pd.read_csv(filepath)
    print(f"  -> Data dimuat: {len(df)} baris, {len(df.columns)} kolom.")

    # Imputasi Missing Values
    print("\n[STEP 2] Preprocessing & Imputasi Data...")
    missing_before = df.isnull().sum().sum()
    
    if "Student_Type" in df.columns:
        df["Student_Type"] = df["Student_Type"].fillna(df["Student_Type"].mode()[0])
        
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if TARGET_COL in num_cols:
        num_cols.remove(TARGET_COL)
    for col in num_cols:
        df[col] = df[col].fillna(df[col].mean())
        
    print(f"  -> Selesai. Total missing values diimputasi: {missing_before} -> {df.isnull().sum().sum()}")
    return df


def run_eda(df):
    # Menampilkan statistik dasar data: dimensi, distribusi target, dan rata-rata fitur per kelas.
    print("\n" + "=" * 60)
    print(" EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)
    print(f"Dimensi Data: {df.shape[0]} baris, {df.shape[1]} kolom")
    
    counts = df[TARGET_COL].value_counts()
    pcts = df[TARGET_COL].value_counts(normalize=True) * 100
    print("\n1. Distribusi Kelas Target (Stress_Level):")
    for cls in sorted(counts.index):
        label = "Rendah (0)" if cls == 0 else "Tinggi (1)"
        print(f"   Kelas {cls} [{label:^10}]: {counts[cls]:>5} baris ({pcts[cls]:.2f}%)")
        
    print("\n2. Rata-rata Nilai Fitur per Kelas:")
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    print(df.groupby(TARGET_COL)[num_cols].mean().to_string(float_format=lambda x: f"{x:.2f}"))
    print("=" * 60 + "\n")


def train_val_test_split(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=RANDOM_SEED):
    # Membagi data secara acak menjadi: 70% Train, 15% Val, dan 15% Test.
    shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    print(f"\n[STEP 3] Split Data -> Train: {train_end}, Val: {val_end - train_end}, Test: {n - val_end}")
    return shuffled.iloc[:train_end], shuffled.iloc[train_end:val_end], shuffled.iloc[val_end:]


class NaiveBayes:
    # Mixed Naive Bayes Classifier.
    def __init__(self, cat_cols=None, num_cols=None):
        self.cat_cols = cat_cols if cat_cols is not None else []
        self.num_cols = num_cols if num_cols is not None else []
        self.classes = []
        self.priors = {}
        self.cat_likelihoods = {}
        self.num_params = {}
        self.cat_unique_vals = {}

    def fit(self, X, y):
        # Melatih model dengan menghitung prior, probabilitas kategorikal (Laplace), dan mean/variansi numerik.
        self.classes = sorted(y.unique().tolist())
        total_samples = len(y)
        
        for class_label in self.classes:
            self.priors[class_label] = int(np.sum(y == class_label)) / total_samples
            self.cat_likelihoods[class_label] = {}
            self.num_params[class_label] = {}
            
            X_class = X[y == class_label]
            class_sample_count = len(X_class)
            
            # Likelihood Kategorikal dengan Laplace Smoothing
            for feature_name in self.cat_cols:
                self.cat_unique_vals[feature_name] = sorted(X[feature_name].unique().tolist())
                counts = X_class[feature_name].value_counts()
                unique_vals = self.cat_unique_vals[feature_name]
                
                self.cat_likelihoods[class_label][feature_name] = {
                    val: (counts.get(val, 0) + 1) / (class_sample_count + len(unique_vals))
                    for val in unique_vals
                }
            
            # Parameter Gaussian untuk Numerik
            for feature_name in self.num_cols:
                mean_val = float(X_class[feature_name].mean())
                variance = float(X_class[feature_name].var())
                self.num_params[class_label][feature_name] = (mean_val, variance if variance > 0 else 1e-9)

    def predict(self, X, threshold_offset=0.0):
        # Melakukan prediksi untuk kumpulan data (vektorisasi numpy).
        N = len(X)
        log_scores_matrix = np.zeros((N, len(self.classes)))
        
        if self.num_cols:
            X_num = X[self.num_cols].to_numpy(dtype=float)
            
        for i, class_label in enumerate(self.classes):
            log_prob = np.full(N, math.log(self.priors[class_label]))
            
            # Categorical Likelihoods
            for col in self.cat_cols:
                mapping = {val: math.log(p) for val, p in self.cat_likelihoods[class_label][col].items()}
                fallback = math.log(1.0 / (len(self.cat_unique_vals[col]) + 1))
                log_prob += X[col].map(mapping).fillna(fallback).to_numpy()
                
            # Gaussian PDF Likelihoods (Vectorized)
            if self.num_cols:
                means = np.array([self.num_params[class_label][col][0] for col in self.num_cols])
                variances = np.array([self.num_params[class_label][col][1] for col in self.num_cols])
                
                coeff = 1.0 / np.sqrt(2 * np.pi * variances)
                exponent = -((X_num - means) ** 2) / (2 * variances)
                pdf = np.clip(coeff * np.exp(exponent), 1e-15, None)
                log_prob += np.log(pdf).sum(axis=1)
                
            log_scores_matrix[:, i] = log_prob
            
        if len(self.classes) == 2 and self.classes == [0, 1]:
            return np.where(log_scores_matrix[:, 1] - log_scores_matrix[:, 0] > threshold_offset, 1, 0).tolist()
        return np.argmax(log_scores_matrix, axis=1).tolist()

    def predict_single(self, sample, threshold_offset=0.0):
        # Melakukan prediksi untuk satu baris data.
        df_temp = pd.DataFrame([sample])
        preds = self.predict(df_temp, threshold_offset)
        
        # Hitung log-scores untuk return value 
        log_scores = {}
        for class_label in self.classes:
            lp = math.log(self.priors[class_label])
            for col in self.cat_cols:
                lp += math.log(self.cat_likelihoods[class_label][col].get(sample[col], 1.0 / (len(self.cat_unique_vals[col]) + 1)))
            for col in self.num_cols:
                mean, var = self.num_params[class_label][col]
                coeff = 1.0 / math.sqrt(2 * math.pi * var)
                prob = max(coeff * math.exp(-((float(sample[col]) - mean) ** 2) / (2 * var)), 1e-15)
                lp += math.log(prob)
            log_scores[class_label] = lp
            
        return preds[0], log_scores


def calculate_metrics(y_true, y_pred):
    # Menghitung metrik klasifikasi (Akurasi, Presisi, Recall, F1-Score, Confusion Matrix) secara vectorized.
    y_t = np.array(y_true)
    y_p = np.array(y_pred)
    
    tp = int(np.sum((y_t == 1) & (y_p == 1)))
    fp = int(np.sum((y_t == 0) & (y_p == 1)))
    tn = int(np.sum((y_t == 0) & (y_p == 0)))
    fn = int(np.sum((y_t == 1) & (y_p == 0)))
    total = len(y_true)
    
    acc = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        "accuracy": acc, "error_rate": 1 - acc, "precision": precision,
        "recall": recall, "specificity": specificity,
        "fpr": 1 - specificity, "fnr": 1 - recall,
        "f1_score": 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn
    }


def print_evaluation_report(name, metrics):
    print("\n" + "=" * 65)
    print(f" LAPORAN EVALUASI: DATA {name.upper()}")
    print("=" * 65)
    for k, v in metrics.items():
        if k not in ["tp", "fp", "tn", "fn"]:
            print(f"  {k.replace('_', ' ').title():<22} : {v * 100:.2f}%")
    print("-" * 65)
    print("  Confusion Matrix:")
    print("                      Pred: Rendah (0)    Pred: Tinggi (1)")
    print(f"  Aktual: Rendah (0)         {metrics['tn']:^10} (TN)         {metrics['fp']:^10} (FP)")
    print(f"  Aktual: Tinggi (1)         {metrics['fn']:^10} (FN)         {metrics['tp']:^10} (TP)")
    print("=" * 65)


def run_demo_predictions(model):
    print("\n" + "=" * 70 + "\n DEMO PREDIKSI KASUS\n" + "=" * 70)
    uji_kasus = [
        {"Student_Type": "college", "Sleep_Hours": 4.5, "Study_Hours": 8.0, "Social_Media_Hours": 1.5, "Attendance": 65.0, "Exam_Pressure": 9.0, "Family_Support": 2.0, "Month": 5.0},
        {"Student_Type": "school", "Sleep_Hours": 8.5, "Study_Hours": 3.0, "Social_Media_Hours": 2.0, "Attendance": 95.0, "Exam_Pressure": 2.0, "Family_Support": 9.0, "Month": 3.0}
    ]
    for i, sample in enumerate(uji_kasus, 1):
        pred, _ = model.predict_single(sample, threshold_offset=THRESHOLD_OFFSET)
        lbl = "TINGKAT STRES TINGGI (1)" if pred == 1 else "TINGKAT STRES RENDAH (0)"
        print(f"  Kasus Uji {i}: Sleep={sample['Sleep_Hours']}h, Study={sample['Study_Hours']}h -> Hasil: ** {lbl} **")


def main():
    df = load_and_preprocess(FILEPATH)
    run_eda(df)
    train_df, val_df, test_df = train_val_test_split(df)
    
    cat_features = ["Student_Type"]
    num_features = ["Sleep_Hours", "Study_Hours", "Social_Media_Hours", "Attendance", "Exam_Pressure", "Family_Support", "Month"]
    
    X_train = train_df[cat_features + num_features]
    y_train = train_df[TARGET_COL]
    
    print("\n[STEP 4] Melatih model Naive Bayes...")
    model = NaiveBayes(cat_cols=cat_features, num_cols=num_features)
    model.fit(X_train, y_train)
    print("  -> Model kelar dilatih!")
    
    print("\n[STEP 5] Evaluasi pada Data Validation...")
    val_preds = model.predict(val_df[cat_features + num_features], threshold_offset=THRESHOLD_OFFSET)
    print_evaluation_report("Validation", calculate_metrics(val_df[TARGET_COL], val_preds))
    
    print("\n[STEP 6] Evaluasi pada Data Testing...")
    test_preds = model.predict(test_df[cat_features + num_features], threshold_offset=THRESHOLD_OFFSET)
    print_evaluation_report("Testing", calculate_metrics(test_df[TARGET_COL], test_preds))
    
    run_demo_predictions(model)


if __name__ == "__main__":
    main()
