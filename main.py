import os
import math
import sys
import pandas as pd

FILEPATH = "student_lifestyle_dataset.csv"
TARGET_COL = "Stress_Level"
RANDOM_SEED = 42
THRESHOLD_OFFSET = -0.5


def load_and_preprocess(filepath):
    """Membaca dataset dan melakukan imputasi missing values."""
    print(f"\n[STEP 1] Memuat dataset dari '{filepath}'...")
    
    if not os.path.exists(filepath):
        print(f"  [INFO] File '{filepath}' ga ada di lokal.")
        print("  [INFO] Coba download dataset dari Kaggle pakai kagglehub...")
        try:
            import kagglehub
            import shutil
            download_path = kagglehub.dataset_download(
                "sridevilavanyacse/student-lifestyle-and-stress-prediction-dataset"
            )
            src_file = os.path.join(download_path, "student-lifestyle-and-stress-dataset.csv")
            shutil.copy(src_file, filepath)
            print(f"  [INFO] Dataset berhasil diunduh dan disalin ke '{filepath}'.")
        except Exception as e:
            print(f"  [ERROR] Gagal download dataset: {e}")
            print("  Pastiin koneksi internet jalan atau taruh file csv manual di folder kerja.")
            sys.exit(1)

    df = pd.read_csv(filepath)
    total_rows = len(df)
    print(f"  -> Total data ke-load: {total_rows} baris, {len(df.columns)} kolom.")

    print("\n[STEP 2] Preprocessing & Imputasi Missing Values...")
    
    # Catat status missing values
    missing_before = df.isnull().sum().to_dict()
    missing_after = {}
    imputation_methods = {}
    
    # Imputasi kolom kategorikal
    if "Student_Type" in df.columns:
        missing_cat = missing_before.get("Student_Type", 0)
        if missing_cat > 0:
            mode_val = df["Student_Type"].mode()[0]
            df["Student_Type"] = df["Student_Type"].fillna(mode_val)
            imputation_methods["Student_Type"] = f"Modus ({mode_val})"
        else:
            imputation_methods["Student_Type"] = "-"
        missing_after["Student_Type"] = df["Student_Type"].isnull().sum()

    # Imputasi kolom numerik
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if TARGET_COL in num_cols:
        num_cols.remove(TARGET_COL)
        missing_after[TARGET_COL] = df[TARGET_COL].isnull().sum()
        imputation_methods[TARGET_COL] = "-"
        
    for col in num_cols:
        missing_num = missing_before.get(col, 0)
        if missing_num > 0:
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)
            imputation_methods[col] = f"Mean ({mean_val:.2f})"
        else:
            imputation_methods[col] = "-"
        missing_after[col] = df[col].isnull().sum()
        
    for col in df.columns:
        if col not in missing_after:
            missing_after[col] = df[col].isnull().sum()
            imputation_methods[col] = "-"

    # Tampilkan tabel preprocessing
    print("  +-----------------------------------------------------------------------------------+")
    print("  |                   TABEL VISUALISASI LAPORAN PREPROCESSING DATA                    |")
    print("  +----------------------+--------------------+--------------------+------------------+")
    print("  | Nama Kolom           | Nilai Kosong Awal  | Nilai Kosong Akhir | Metode Imputasi  |")
    print("  +----------------------+--------------------+--------------------+------------------+")
    for col in df.columns:
        pct_before = (missing_before.get(col, 0) / total_rows) * 100
        before_str = f"{missing_before.get(col, 0)} ({pct_before:.1f}%)"
        pct_after = (missing_after[col] / total_rows) * 100
        after_str = f"{missing_after[col]} ({pct_after:.1f}%)"
        method_str = imputation_methods.get(col, "-")
        print(f"  | {col:<20} | {before_str:<18} | {after_str:<18} | {method_str:<16} |")
    print("  +----------------------+--------------------+--------------------+------------------+")
    print("  -> Imputasi data selesai dengan sukses. Data bersih siap digunakan.")
    
    return df


def run_eda(df):
    """Analisis data eksploratif (EDA) sederhana."""
    print("\n" + "=" * 60)
    print(" EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 60)
    print(f"Dimensi Data: {df.shape[0]} baris, {df.shape[1]} kolom")
    
    print("\n1. Distribusi Kelas Target (Stress_Level):")
    counts = df[TARGET_COL].value_counts()
    pcts = df[TARGET_COL].value_counts(normalize=True) * 100
    for cls in sorted(counts.index):
        label = "Rendah (0)" if cls == 0 else "Tinggi (1)"
        print(f"   Kelas {cls} [{label:^10}]: {counts[cls]:>5} baris ({pcts[cls]:.2f}%)")
        
    print("\n2. Distribusi Fitur Kategorikal (Student_Type):")
    type_counts = df["Student_Type"].value_counts()
    for t, count in type_counts.items():
        print(f"   - {t:<15}: {count:>5} mahasiswa")
        
    print("\n3. Rata-rata Nilai Fitur Numerik per Kelas Stres:")
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if TARGET_COL in num_cols:
        grouped = df.groupby(TARGET_COL)[num_cols].mean()
        grouped.index = ["Stres Rendah (0)", "Stres Tinggi (1)"]
        print(grouped.to_string(float_format=lambda x: f"{x:.2f}"))
        
    print("=" * 60 + "\n")


def train_val_test_split(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=RANDOM_SEED):
    """Membagi dataset menjadi train, validation, dan test set."""
    print(f"\n[STEP 3] Membagi data (Train: {train_ratio*100:.0f}%, Val: {val_ratio*100:.0f}%, Test: {test_ratio*100:.0f}%)...")
    
    shuffled_df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    n = len(shuffled_df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = shuffled_df.iloc[:train_end]
    val_df = shuffled_df.iloc[train_end:val_end]
    test_df = shuffled_df.iloc[val_end:]
    
    print(f"  -> Data Training   : {len(train_df)} baris")
    print(f"  -> Data Validation : {len(val_df)} baris")
    print(f"  -> Data Testing    : {len(test_df)} baris")
    
    return train_df, val_df, test_df


class NaiveBayes:
    """Mixed Naive Bayes Classifier untuk fitur numerik dan kategorikal."""
    def __init__(self, cat_cols=None, num_cols=None):
        self.cat_cols = cat_cols if cat_cols is not None else []
        self.num_cols = num_cols if num_cols is not None else []
        self.classes = []
        self.priors = {}
        self.cat_likelihoods = {}
        self.num_params = {}
        
        # Daftar nilai unik fitur kategorikal untuk Laplace Smoothing
        self.cat_unique_vals = {}

    def fit(self, X, y):
        """Melatih model Naive Bayes."""
        # Identifikasi kelas unik
        self.classes = sorted(y.unique().tolist())
        total_samples = len(y)
        
        # Prior Probability: P(C) = Sampel kelas C / Total sampel
        for class_label in self.classes:
            class_count = sum(y == class_label)
            self.priors[class_label] = class_count / total_samples
            
        # Simpan nilai unik fitur kategorikal untuk Laplace Smoothing
        for feature_name in self.cat_cols:
            self.cat_unique_vals[feature_name] = sorted(X[feature_name].unique().tolist())
            
        # Hitung likelihood dan parameter
        for class_label in self.classes:
            self.cat_likelihoods[class_label] = {}
            self.num_params[class_label] = {}
            
            X_class = X[y == class_label]
            class_sample_count = len(X_class)
            
            # Likelihood Kategorikal dengan Laplace Smoothing
            for feature_name in self.cat_cols:
                self.cat_likelihoods[class_label][feature_name] = {}
                counts = X_class[feature_name].value_counts()
                unique_vals = self.cat_unique_vals[feature_name]
                category_count = len(unique_vals)
                
                for feature_value in unique_vals:
                    val_count = counts.get(feature_value, 0)
                    probability = (val_count + 1) / (class_sample_count + category_count)
                    self.cat_likelihoods[class_label][feature_name][feature_value] = probability
            
            # Likelihood Numerik dengan Parameter Gaussian
            for feature_name in self.num_cols:
                mean_val = X_class[feature_name].mean()
                variance = X_class[feature_name].var()
                
                if variance == 0 or pd.isna(variance):
                    variance = 1e-9
                self.num_params[class_label][feature_name] = (mean_val, variance)

    def _gaussian_pdf(self, x, mean_val, variance):
        """Menghitung Gaussian Probability Density Function (PDF)."""
        coefficient = 1.0 / math.sqrt(2 * math.pi * variance)
        exponent = math.exp(-((x - mean_val) ** 2) / (2 * variance))
        return coefficient * exponent

    def predict_single(self, sample, threshold_offset=0.0):
        """Memprediksi kelas untuk satu baris sampel data."""
        log_scores = {}
        
        for class_label in self.classes:
            # Mulai dari log Prior Probability
            log_prob = math.log(self.priors[class_label])
            
            # Tambahkan log likelihood kategorikal
            for feature_name in self.cat_cols:
                feature_value = sample[feature_name]
                
                if feature_value in self.cat_likelihoods[class_label][feature_name]:
                    probability = self.cat_likelihoods[class_label][feature_name][feature_value]
                else:
                    # Fallback kategori baru dengan Laplace smoothing
                    category_count = len(self.cat_unique_vals[feature_name])
                    probability = 1.0 / (category_count + 1)
                    
                log_prob += math.log(probability)
                
            # Tambahkan log likelihood numerik via Gaussian PDF
            for feature_name in self.num_cols:
                feature_value = float(sample[feature_name])
                mean_val, variance = self.num_params[class_label][feature_name]
                
                probability = self._gaussian_pdf(feature_value, mean_val, variance)
                
                # Batasi probabilitas agar tidak 0 (hindari error log(0))
                if probability < 1e-15:
                    probability = 1e-15
                    
                log_prob += math.log(probability)
                
            log_scores[class_label] = log_prob
            
        # Penentuan keputusan akhir (argmax) dengan modifikasi threshold_offset
        if len(self.classes) == 2 and self.classes == [0, 1]:
            log_difference = log_scores[1] - log_scores[0]
            if log_difference > threshold_offset:
                return 1, log_scores
            else:
                return 0, log_scores
        else:
            # Standar multi-kelas argmax
            best_class = max(log_scores, key=log_scores.get)
            return best_class, log_scores

    def predict(self, X, threshold_offset=0.0):
        """Memprediksi sekumpulan data (pandas DataFrame) baris demi baris."""
        predictions = []
        for _, row in X.iterrows():
            pred, _ = self.predict_single(row, threshold_offset)
            predictions.append(pred)
        return predictions


def calculate_metrics(y_true, y_pred):
    """Menghitung metrik evaluasi klasifikasi secara manual."""
    actual_labels = list(y_true)
    predicted_labels = list(y_pred)
    total_samples = len(actual_labels)
    
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    for actual_val, predicted_val in zip(actual_labels, predicted_labels):
        if actual_val == 1 and predicted_val == 1:
            true_positives += 1
        elif actual_val == 0 and predicted_val == 1:
            false_positives += 1
        elif actual_val == 0 and predicted_val == 0:
            true_negatives += 1
        elif actual_val == 1 and predicted_val == 0:
            false_negatives += 1
            
    accuracy = (true_positives + true_negatives) / total_samples if total_samples > 0 else 0
    error_rate = 1 - accuracy
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
    false_positive_rate = false_positives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
    false_negative_rate = false_negatives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "error_rate": error_rate,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "fpr": false_positive_rate,
        "fnr": false_negative_rate,
        "f1_score": f1_score,
        "tp": true_positives,
        "fp": false_positives,
        "tn": true_negatives,
        "fn": false_negatives
    }


def print_evaluation_report(name, metrics):
    """Menampilkan laporan evaluasi model ke konsol."""
    print("\n" + "=" * 65)
    print(f" LAPORAN EVALUASI: DATA {name.upper()}")
    print("=" * 65)
    print(f"  Akurasi (ACC)          : {metrics['accuracy'] * 100:.2f}%")
    print(f"  Error Rate (1-ACC)     : {metrics['error_rate'] * 100:.2f}%")
    print(f"  Presisi                : {metrics['precision'] * 100:.2f}%")
    print(f"  Recall / TPR           : {metrics['recall'] * 100:.2f}%")
    print(f"  Specificity / TNR      : {metrics['specificity'] * 100:.2f}%")
    print(f"  False Positive Rate    : {metrics['fpr'] * 100:.2f}%")
    print(f"  False Negative Rate    : {metrics['fnr'] * 100:.2f}%")
    print(f"  F1-Score               : {metrics['f1_score'] * 100:.2f}%")
    print("-" * 65)
    print("  Confusion Matrix:")
    print("                      Prediksi: Rendah (0)    Prediksi: Tinggi (1)")
    print(f"  Aktual: Rendah (0)         {metrics['tn']:^10} (TN)         {metrics['fp']:^10} (FP)")
    print(f"  Aktual: Tinggi (1)         {metrics['fn']:^10} (FN)         {metrics['tp']:^10} (TP)")
    print("-" * 65)
    print("  Keterangan Detail Confusion Matrix:")
    print(f"  - TN (True Negative)  : {metrics['tn']} data.")
    print("                          Aktual Stres Rendah (0) -> Diprediksi Rendah (0) [BENAR]")
    print(f"  - TP (True Positive)  : {metrics['tp']} data.")
    print("                          Aktual Stres Tinggi (1) -> Diprediksi Tinggi (1) [BENAR]")
    print(f"  - FP (False Positive) : {metrics['fp']} data.")
    print("                          Aktual Stres Rendah (0) -> Diprediksi Tinggi (1) [SALAH]")
    print(f"  - FN (False Negative) : {metrics['fn']} data.")
    print("                          Aktual Stres Tinggi (1) -> Diprediksi Rendah (0) [SALAH - BAHAYA!]")
    print("=" * 65)


def save_model_to_json(model, filename):
    """Menyimpan parameter model ke berkas JSON."""
    import json
    model_data = {
        "cat_cols": model.cat_cols,
        "num_cols": model.num_cols,
        "classes": model.classes,
        "priors": model.priors,
        "cat_likelihoods": model.cat_likelihoods,
        "num_params": model.num_params,
        "cat_unique_vals": model.cat_unique_vals
    }
    with open(filename, "w") as f:
        json.dump(model_data, f, indent=4)
    print(f"  -> Model berhasil diekspor ke '{filename}'")


def load_model_from_json(filename):
    """Memuat model Naive Bayes dari berkas JSON."""
    import json
    with open(filename, "r") as f:
        data = json.load(f)
        
    model = NaiveBayes(cat_cols=data["cat_cols"], num_cols=data["num_cols"])
    model.classes = data["classes"]
    
    # Konversi string key ke tipe aslinya
    def convert_key(k):
        try:
            if float(k).is_integer():
                return int(k)
            return float(k)
        except ValueError:
            return k
            
    model.priors = {convert_key(k): v for k, v in data["priors"].items()}
    
    model.cat_likelihoods = {}
    for cls_str, cols_data in data["cat_likelihoods"].items():
        cls = convert_key(cls_str)
        model.cat_likelihoods[cls] = cols_data
        
    model.num_params = {}
    for cls_str, cols_data in data["num_params"].items():
        cls = convert_key(cls_str)
        model.num_params[cls] = {}
        for col, params in cols_data.items():
            model.num_params[cls][col] = (params[0], params[1])
            
    model.cat_unique_vals = data["cat_unique_vals"]
    print(f"  -> Model berhasil dimuat dari '{filename}'")
    return model


def run_demo_predictions(model):
    """Menjalankan demo prediksi sampel kasus."""
    print("\n" + "=" * 70)
    print(" DEMO PREDIKSI KASUS KHUSUS (UJI)")
    print("=" * 70)
    
    uji_kasus = [
        {
            # Kasus 1
            "Student_Type": "college",
            "Sleep_Hours": 4.5,
            "Study_Hours": 8.0,
            "Social_Media_Hours": 1.5,
            "Attendance": 65.0,
            "Exam_Pressure": 9.0,
            "Family_Support": 2.0,
            "Month": 5.0
        },
        {
            # Kasus 2
            "Student_Type": "school",
            "Sleep_Hours": 8.5,
            "Study_Hours": 3.0,
            "Social_Media_Hours": 2.0,
            "Attendance": 95.0,
            "Exam_Pressure": 2.0,
            "Family_Support": 9.0,
            "Month": 3.0
        }
    ]

    for i, sample in enumerate(uji_kasus, 1):
        pred, log_scores = model.predict_single(sample, threshold_offset=THRESHOLD_OFFSET)
        label_pred = "TINGKAT STRES TINGGI (1)" if pred == 1 else "TINGKAT STRES RENDAH (0)"
        print(f"\n  Kasus Uji {i}:")
        print(f"    Tipe Mahasiswa      : {sample['Student_Type']}")
        print(f"    Jam Tidur / Hari    : {sample['Sleep_Hours']} jam")
        print(f"    Jam Belajar / Hari  : {sample['Study_Hours']} jam")
        print(f"    Tekanan Ujian (1-10): {sample['Exam_Pressure']}")
        print(f"    Dukungan Keluarga   : {sample['Family_Support']}")
        print(f"    Persentase Kehadiran: {sample['Attendance']}%")
        print(f"    -> Hasil Prediksi   : ** {label_pred} **")


def mode_train_from_scratch():
    """Mode training model dari awal."""
    df = load_and_preprocess(FILEPATH)
    train_df, _, _ = train_val_test_split(df)
    
    cat_features = ["Student_Type"]
    num_features = [
        "Sleep_Hours",
        "Study_Hours",
        "Social_Media_Hours",
        "Attendance",
        "Exam_Pressure",
        "Family_Support",
        "Month"
    ]
    
    X_train = train_df[cat_features + num_features]
    y_train = train_df[TARGET_COL]

    print("\n[STEP 4] Melatih model Naive Bayes...")
    model = NaiveBayes(cat_cols=cat_features, num_cols=num_features)
    model.fit(X_train, y_train)
    print("  -> Model kelar dilatih di data training!")

    print("\n[STEP 5] Mengekspor model hasil latihan ke JSON...")
    save_model_to_json(model, "model_naive_bayes.json")

    print("\nPrior Probability hasil training:")
    for cls in model.classes:
        label = "Rendah (0)" if cls == 0 else "Tinggi (1)"
        print(f"  P({label}) = {model.priors[cls]:.4f}")


def mode_load_from_json():
    """Mode evaluasi model dari berkas JSON."""
    json_file = "model_naive_bayes.json"
    
    if not os.path.exists(json_file):
        print(f"\n  [ERROR] File model '{json_file}' tidak ditemukan!")
        print("  Jalankan Mode 1 (Training & Ekspor Model) dulu untuk membuat file model.")
        return
    
    print(f"\n[STEP 1] Memuat model dari berkas '{json_file}'...")
    model = load_model_from_json(json_file)
    
    print("\nPrior Probability dari model JSON:")
    for cls in model.classes:
        label = "Rendah (0)" if cls == 0 else "Tinggi (1)"
        print(f"  P({label}) = {model.priors[cls]:.4f}")
    
    print(f"\n[STEP 2] Memuat dataset untuk evaluasi...")
    df = load_and_preprocess(FILEPATH)
    
    run_eda(df)
    _, val_df, test_df = train_val_test_split(df)
    
    cat_features = ["Student_Type"]
    num_features = [
        "Sleep_Hours",
        "Study_Hours",
        "Social_Media_Hours",
        "Attendance",
        "Exam_Pressure",
        "Family_Support",
        "Month"
    ]
    
    X_val = val_df[cat_features + num_features]
    y_val = val_df[TARGET_COL]
    
    X_test = test_df[cat_features + num_features]
    y_test = test_df[TARGET_COL]
    
    print("\n[STEP 3] Evaluasi model JSON pada data Validation...")
    val_preds = model.predict(X_val, threshold_offset=THRESHOLD_OFFSET)
    val_metrics = calculate_metrics(y_val, val_preds)
    print_evaluation_report("Validation (Model JSON)", val_metrics)
    
    print("\n[STEP 4] Evaluasi model JSON pada data Testing...")
    test_preds = model.predict(X_test, threshold_offset=THRESHOLD_OFFSET)
    test_metrics = calculate_metrics(y_test, test_preds)
    print_evaluation_report("Testing (Model JSON)", test_metrics)
    
    run_demo_predictions(model)


def main():
    print("=" * 70)
    print("      SISTEM PREDIKSI TINGKAT STRES MAHASISWA")
    print("    Metode: Naive Bayes")
    print("=" * 70)
    
    print("\n  Pilih mode yang ingin dijalankan:")
    print("  [1] Training & Ekspor Model ke JSON")
    print("  [2] Muat Model JSON & Evaluasi (Gunakan Model)")
    
    try:
        if not sys.stdin.isatty():
            print("\n  [INFO] Non-interactive environment. Otomatis jalankan Mode 1.")
            pilihan = "1"
        else:
            pilihan = input("\n  Masukkan pilihan (1/2): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n  Input dibatalkan.")
        return
    
    if pilihan == "1":
        print("\n" + "-" * 70)
        print("  >> Mode 1: TRAINING & EKSPOR MODEL")
        print("-" * 70)
        mode_train_from_scratch()
        
        try:
            if sys.stdin.isatty():
                print("\n" + "-" * 70)
                tanya_lanjut = input("Mau langsung lanjut ke Mode 2 (Muat Model & Evaluasi)? (y/n): ").strip().lower()
                if tanya_lanjut == 'y':
                     print("\n" + "-" * 70)
                     print("  >> Mode 2: MUAT MODEL & EVALUASI")
                     print("-" * 70)
                     mode_load_from_json()
        except (EOFError, KeyboardInterrupt):
            print("\nKeluar dari program.")
            return
    elif pilihan == "2":
        print("\n" + "-" * 70)
        print("  >> Mode 2: MUAT MODEL & EVALUASI")
        print("-" * 70)
        mode_load_from_json()
    else:
        print("\n  [ERROR] Pilihan tidak valid. Masukkan 1 atau 2.")
        return

    print("\n" + "=" * 70)
    print("  Program Selesai. Makasih!")
    print("=" * 70)


if __name__ == "__main__":
    main()
