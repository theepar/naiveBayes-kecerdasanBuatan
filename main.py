import os
import math
import sys
import pandas as pd

FILEPATH = "student_lifestyle_dataset.csv"
TARGET_COL = "Stress_Level"
RANDOM_SEED = 42
THRESHOLD_OFFSET = -0.5


# 1. Load & Preprocess Data
def load_and_preprocess(filepath):
    """
    Fungsi buat baca dataset dan beresin missing values (imputasi).
    """
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
    print(f"  -> Total data ke-load: {len(df)} baris, {len(df.columns)} kolom.")

    print("\n[STEP 2] Preprocessing data (isi missing values)...")
    
    # Isi kolom kategorik (Student_Type) pake Modus (nilai tersering)
    if "Student_Type" in df.columns:
        missing_cat = df["Student_Type"].isnull().sum()
        if missing_cat > 0:
            mode_val = df["Student_Type"].mode()[0]
            df["Student_Type"] = df["Student_Type"].fillna(mode_val)
            print(f"  -> Ada {missing_cat} data kosong di 'Student_Type' diisi modus: '{mode_val}'")
        else:
            print("  -> Kolom 'Student_Type' aman, ga ada yang kosong.")

    # Isi kolom numerik yang kosong pake rata-rata (Mean)
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if TARGET_COL in num_cols:
        num_cols.remove(TARGET_COL) 
    
    for col in num_cols:
        missing_num = df[col].isnull().sum()
        if missing_num > 0:
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)
            print(f"  -> Ada {missing_num} data kosong di '{col}' diisi rata-rata: {mean_val:.4f}")
            
    print("  -> Preprocessing beres. Data siap dipakai.")
    return df


# 2. Exploratory Data Analysis (EDA)
def run_eda(df):
    """
    Analisis data simpel buat ngeliat gambaran dataset.
    """
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


# 3. Split Data
def train_val_test_split(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=RANDOM_SEED):
    """
    Bagi dataset acak jadi training, validation, dan testing set.
    """
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


# 4. Naive Bayes Classifier
class NaiveBayes:
    """
    Model Naive Bayes bisa nanganin data campuran:
    - Kategorikal pakai frekuensi tabel dengan Laplace Smoothing.
    - Numerik kontinu pakai rumus fungsi peluang Gauss (normal).
    """
    def __init__(self, cat_cols=None, num_cols=None):
        self.cat_cols = cat_cols if cat_cols is not None else []
        self.num_cols = num_cols if num_cols is not None else []
        self.classes = []
        self.priors = {}
        # Likelihood kategorik: self.cat_likelihoods[kelas][kolom][nilai] = peluang
        self.cat_likelihoods = {}
        # Parameter numerik: self.num_params[kelas][kolom] = (mean, variance)
        self.num_params = {}
        # List nilai unik kategori untuk Laplace smoothing
        self.cat_unique_vals = {}

    def fit(self, X, y):
        self.classes = sorted(y.unique().tolist())
        total_samples = len(y)
        
        # Hitung peluang awal Prior P(C) untuk tiap kelas
        for class_label in self.classes:
            self.priors[class_label] = sum(y == class_label) / total_samples
            
        # Catat semua nilai kategori unik yang ada di data training
        for feature_name in self.cat_cols:
            self.cat_unique_vals[feature_name] = sorted(X[feature_name].unique().tolist())
            
        # Hitung parameter likelihood per kelas
        for class_label in self.classes:
            self.cat_likelihoods[class_label] = {}
            self.num_params[class_label] = {}
            
            # Filter baris yang sesuai kelas saat ini
            X_class = X[y == class_label]
            class_sample_count = len(X_class)
            
            # Fitur Kategorikal: Hitung pakai Laplace Smoothing
            for feature_name in self.cat_cols:
                self.cat_likelihoods[class_label][feature_name] = {}
                counts = X_class[feature_name].value_counts()
                unique_vals = self.cat_unique_vals[feature_name]
                category_count = len(unique_vals)  # Jumlah kategori unik
                
                for feature_value in unique_vals:
                    val_count = counts.get(feature_value, 0)
                    # Rumus Laplace: (count + 1) / (class_sample_count + category_count)
                    self.cat_likelihoods[class_label][feature_name][feature_value] = (val_count + 1) / (class_sample_count + category_count)
            
            # Fitur Numerik: Hitung nilai rata-rata (mean) & variansi (variance)
            for feature_name in self.num_cols:
                mean_val = X_class[feature_name].mean()
                variance = X_class[feature_name].var()
                # Kalau variansi 0, ganti angka super kecil biar ga pembagian nol
                if variance == 0 or pd.isna(variance):
                    variance = 1e-9
                self.num_params[class_label][feature_name] = (mean_val, variance)

    def _gaussian_pdf(self, x, mean_val, variance):
        # Hitung rumus peluang distribusi normal Gauss
        exponent = math.exp(-((x - mean_val) ** 2) / (2 * variance))
        return (1.0 / math.sqrt(2 * math.pi * variance)) * exponent

    def predict_single(self, sample, threshold_offset=0.0):
        # Prediksi satu baris data
        log_scores = {}
        
        for class_label in self.classes:
            # Mulai dari log prior P(C)
            log_prob = math.log(self.priors[class_label])
            
            # Tambah probabilitas log dari fitur kategorikal
            for feature_name in self.cat_cols:
                feature_value = sample[feature_name]
                if feature_value in self.cat_likelihoods[class_label][feature_name]:
                    probability = self.cat_likelihoods[class_label][feature_name][feature_value]
                else:
                    # Kalau ada nilai baru yang ga ada pas training, pakai fallback Laplace
                    category_count = len(self.cat_unique_vals[feature_name])
                    probability = 1.0 / (category_count + 1)
                log_prob += math.log(probability)
                
            # Tambah probabilitas log dari fitur numerik (Gaussian)
            for feature_name in self.num_cols:
                feature_value = float(sample[feature_name])
                mean_val, variance = self.num_params[class_label][feature_name]
                probability = self._gaussian_pdf(feature_value, mean_val, variance)
                # Batasin biar ga log(0)
                if probability < 1e-15:
                    probability = 1e-15
                log_prob += math.log(probability)
                
            log_scores[class_label] = log_prob
            
        # Jika klasifikasi biner 0 dan 1, gunakan threshold_offset untuk meminimalkan False Negative
        if len(self.classes) == 2 and self.classes == [0, 1]:
            if log_scores[1] - log_scores[0] > threshold_offset:
                return 1, log_scores
            else:
                return 0, log_scores
        else:
            best_class = max(log_scores, key=log_scores.get)
            return best_class, log_scores

    def predict(self, X, threshold_offset=0.0):
        predictions = []
        for _, row in X.iterrows():
            pred, _ = self.predict_single(row, threshold_offset)
            predictions.append(pred)
        return predictions


# 5. Evaluasi Metriks
def calculate_metrics(y_true, y_pred):
    """
    Hitung metrik performa model (Akurasi, Presisi, Recall, F1) manual dari nol.
    """
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
    print("\n" + "=" * 60)
    print(f" LAPORAN EVALUASI: DATA {name.upper()}")
    print("=" * 60)
    print(f"  Akurasi (ACC)          : {metrics['accuracy'] * 100:.2f}%")
    print(f"  Error Rate (1-ACC)     : {metrics['error_rate'] * 100:.2f}%")
    print(f"  Presisi                : {metrics['precision'] * 100:.2f}%")
    print(f"  Recall / TPR           : {metrics['recall'] * 100:.2f}%")
    print(f"  Specificity / TNR      : {metrics['specificity'] * 100:.2f}%")
    print(f"  False Positive Rate    : {metrics['fpr'] * 100:.2f}%")
    print(f"  False Negative Rate    : {metrics['fnr'] * 100:.2f}%")
    print(f"  F1-Score               : {metrics['f1_score'] * 100:.2f}%")
    print("-" * 60)
    print("  Confusion Matrix:")
    print("                      Prediksi: Rendah (0)    Prediksi: Tinggi (1)")
    print(f"  Aktual: Rendah (0)         {metrics['tn']:^10}               {metrics['fp']:^10}")
    print(f"  Aktual: Tinggi (1)         {metrics['fn']:^10}               {metrics['tp']:^10}")
    print("=" * 60)


# 6. Ekspor & Impor Model ke berkas JSON
def save_model_to_json(model, filename):
    """
    Menyimpan parameter model hasil training ke berkas JSON.
    """
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
    """
    Memuat kembali model Naive Bayes yang sudah dilatih dari berkas JSON.
    """
    import json
    with open(filename, "r") as f:
        data = json.load(f)
        
    model = NaiveBayes(cat_cols=data["cat_cols"], num_cols=data["num_cols"])
    model.classes = data["classes"]
    
    # Fungsi pembantu untuk mengembalikan kunci string dari JSON ke tipe aslinya (int/float)
    def convert_key(k):
        try:
            if float(k).is_integer():
                return int(k)
            return float(k)
        except ValueError:
            return k
            
    # Kembalikan tipe data kunci P(C)
    model.priors = {convert_key(k): v for k, v in data["priors"].items()}
    
    # Kembalikan tipe data kunci Likelihood kategorikal
    model.cat_likelihoods = {}
    for cls_str, cols_data in data["cat_likelihoods"].items():
        cls = convert_key(cls_str)
        model.cat_likelihoods[cls] = cols_data
        
    # Kembalikan tipe data kunci Parameter numerik (mean, variance)
    model.num_params = {}
    for cls_str, cols_data in data["num_params"].items():
        cls = convert_key(cls_str)
        model.num_params[cls] = {}
        for col, params in cols_data.items():
            model.num_params[cls][col] = (params[0], params[1])
            
    model.cat_unique_vals = data["cat_unique_vals"]
    print(f"  -> Model berhasil dimuat dari '{filename}'")
    return model


# 7. Demo Prediksi Kasus Khusus
def run_demo_predictions(model):
    """
    Menjalankan demo prediksi dengan data sampel yang sudah disiapkan.
    """
    print("\n" + "=" * 70)
    print(" DEMO PREDIKSI KASUS KHUSUS (UJI)")
    print("=" * 70)
    
    uji_kasus = [
        {
            # Kasus A: Kuliah berat, kurang tidur, jarang disupport
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
            # Kasus B: Sekolah nyantai, tidur cukup, support keluarga oke
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



# 9. Mode 1: Training dari Awal
def mode_train_from_scratch():
    """
    Alur: Load data -> Preprocess -> Split -> Training -> Ekspor JSON.
    """
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


# 10. Mode 2: Muat Model dari Berkas JSON
def mode_load_from_json():
    """
    Alur cepat: Langsung muat model yang sudah dilatih dari file JSON, lalu evaluasi & prediksi.
    """
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


# 11. Main Program (Menu Utama)
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
