import os
import math
import random
import sys
import pandas as pd

# Konfigurasi dasar
FILEPATH = "student_lifestyle_dataset.csv"
TARGET_COL = "Stress_Level"
RANDOM_SEED = 42


# 1. Load & Preprocess Data
def load_and_preprocess(filepath):
    """
    Fungsi buat baca dataset dan beresin missing values (imputasi).
    """
    print(f"\n[STEP 1] Memuat dataset dari '{filepath}'...")
    
    # Kalau file csv lokal ga ada, download dulu dari kagglehub
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
        num_cols.remove(TARGET_COL)  # Skip target biar ga ke-impute
    
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
    
    # Shuffle data biar acak merata
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


# 4. Mixed Naive Bayes Classifier
class MixedNaiveBayes:
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
        for cls in self.classes:
            self.priors[cls] = sum(y == cls) / total_samples
            
        # Catat semua nilai kategori unik yang ada di data training
        for col in self.cat_cols:
            self.cat_unique_vals[col] = sorted(X[col].unique().tolist())
            
        # Hitung parameter likelihood per kelas
        for cls in self.classes:
            self.cat_likelihoods[cls] = {}
            self.num_params[cls] = {}
            
            # Filter baris yang sesuai kelas saat ini
            X_cls = X[y == cls]
            n_cls = len(X_cls)
            
            # Fitur Kategorikal: Hitung pakai Laplace Smoothing
            for col in self.cat_cols:
                self.cat_likelihoods[cls][col] = {}
                counts = X_cls[col].value_counts()
                unique_vals = self.cat_unique_vals[col]
                k = len(unique_vals)  # Jumlah kategori unik
                
                for val in unique_vals:
                    val_count = counts.get(val, 0)
                    # Rumus Laplace: (count + 1) / (n_cls + k)
                    self.cat_likelihoods[cls][col][val] = (val_count + 1) / (n_cls + k)
            
            # Fitur Numerik: Hitung nilai rata-rata (mean) & variansi (variance)
            for col in self.num_cols:
                mean = X_cls[col].mean()
                var = X_cls[col].var()
                # Kalau variansi 0, ganti angka super kecil biar ga pembagian nol
                if var == 0 or pd.isna(var):
                    var = 1e-9
                self.num_params[cls][col] = (mean, var)

    def _gaussian_pdf(self, x, mean, var):
        # Hitung rumus peluang distribusi normal Gauss
        exponent = math.exp(-((x - mean) ** 2) / (2 * var))
        return (1.0 / math.sqrt(2 * math.pi * var)) * exponent

    def predict_single(self, sample):
        # Prediksi satu baris data
        log_scores = {}
        
        for cls in self.classes:
            # Mulai dari log prior P(C)
            log_prob = math.log(self.priors[cls])
            
            # Tambah probabilitas log dari fitur kategorikal
            for col in self.cat_cols:
                val = sample[col]
                if val in self.cat_likelihoods[cls][col]:
                    prob = self.cat_likelihoods[cls][col][val]
                else:
                    # Kalau ada nilai baru yang ga ada pas training, pakai fallback Laplace
                    k = len(self.cat_unique_vals[col])
                    prob = 1.0 / (k + 1)
                log_prob += math.log(prob)
                
            # Tambah probabilitas log dari fitur numerik (Gaussian)
            for col in self.num_cols:
                val = float(sample[col])
                mean, var = self.num_params[cls][col]
                prob = self._gaussian_pdf(val, mean, var)
                # Batasin biar ga log(0)
                if prob < 1e-15:
                    prob = 1e-15
                log_prob += math.log(prob)
                
            log_scores[cls] = log_prob
            
        # Cari kelas dengan skor peluang paling gede (argmax)
        best_class = max(log_scores, key=log_scores.get)
        return best_class, log_scores

    def predict(self, X):
        # Prediksi banyak data sekaligus
        predictions = []
        for _, row in X.iterrows():
            pred, _ = self.predict_single(row)
            predictions.append(pred)
        return predictions


# 5. Evaluasi Metriks
def calculate_metrics(y_true, y_pred):
    """
    Hitung metrik performa model (Akurasi, Presisi, Recall, F1) manual dari nol.
    """
    y_true = list(y_true)
    y_pred = list(y_pred)
    n = len(y_true)
    
    tp = 0  # Tebak stres tinggi (1) dan aslinya stres tinggi (1)
    fp = 0  # Tebak stres tinggi (1) padahal aslinya stres rendah (0)
    tn = 0  # Tebak stres rendah (0) dan aslinya stres rendah (0)
    fn = 0  # Tebak stres rendah (0) padahal aslinya stres tinggi (1)
    
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 0 and p == 0:
            tn += 1
        elif t == 1 and p == 0:
            fn += 1
            
    accuracy = (tp + tn) / n if n > 0 else 0
    error_rate = 1 - accuracy  # Tingkat Kesalahan = 1 - Akurasi
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # TPR / Sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # TNR
    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0  # False Positive Rate
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0  # False Negative Rate
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "error_rate": error_rate,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "fpr": fpr,
        "fnr": fnr,
        "f1_score": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn
    }


def print_evaluation_report(name, metrics):
    # Cetak laporan metrik model biar keliatan rapi
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
        
    model = MixedNaiveBayes(cat_cols=data["cat_cols"], num_cols=data["num_cols"])
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
        pred, log_scores = model.predict_single(sample)
        label_pred = "TINGKAT STRES TINGGI (1)" if pred == 1 else "TINGKAT STRES RENDAH (0)"
        print(f"\n  Kasus Uji {i}:")
        print(f"    Tipe Mahasiswa      : {sample['Student_Type']}")
        print(f"    Jam Tidur / Hari    : {sample['Sleep_Hours']} jam")
        print(f"    Jam Belajar / Hari  : {sample['Study_Hours']} jam")
        print(f"    Tekanan Ujian (1-10): {sample['Exam_Pressure']}")
        print(f"    Dukungan Keluarga   : {sample['Family_Support']}")
        print(f"    Persentase Kehadiran: {sample['Attendance']}%")
        print(f"    -> Hasil Prediksi   : ** {label_pred} **")


# 8. Input Manual Interaktif
def run_interactive_input(model):
    """
    Memungkinkan user mengetik data sendiri lewat konsol untuk diprediksi.
    """
    try:
        if not sys.stdin.isatty():
            print("\n[INFO] Non-interactive environment. Skip ketik manual.")
            return
            
        print("\n" + "=" * 70)
        print(" INPUT INTERAKTIF (UJI DATA BARU)")
        print("=" * 70)
        tanya_user = input("Mau coba ketik data kamu sendiri? (y/n): ").strip().lower()
        if tanya_user == 'y':
            while True:
                print("\nKetik data kamu di bawah:")
                try:
                    tipe = input("- Tipe Mahasiswa (college/school/working_student): ").strip().lower()
                    if tipe not in ['college', 'school', 'working_student']:
                        tipe = 'college'
                        print("  (Input salah, otomatis diganti ke 'college')")
                        
                    sleep = float(input("- Jam Tidur per Hari (misal 6.5): "))
                    study = float(input("- Jam Belajar per Hari (misal 4.0): "))
                    socmed = float(input("- Jam Medsos per Hari (misal 2.0): "))
                    attend = float(input("- Kehadiran Kelas (0 - 100): "))
                    pressure = float(input("- Skala Tekanan Ujian (1 - 10): "))
                    support = float(input("- Skala Dukung Keluarga (1 - 10): "))
                    month = float(input("- Bulan Akademik (1 - 12): "))
                    
                    user_sample = {
                        "Student_Type": tipe,
                        "Sleep_Hours": sleep,
                        "Study_Hours": study,
                        "Social_Media_Hours": socmed,
                        "Attendance": attend,
                        "Exam_Pressure": pressure,
                        "Family_Support": support,
                        "Month": month
                    }
                    
                    pred, _ = model.predict_single(user_sample)
                    label_pred = "TINGKAT STRES TINGGI (1)" if pred == 1 else "TINGKAT STRES RENDAH (0)"
                    print(f"\n==========================================")
                    print(f"HASIL PREDIKSI: {label_pred}")
                    print(f"==========================================")
                    
                except ValueError:
                    print("  [ERROR] Masukan angka salah. Ulangi lagi.")
                    
                lagi = input("\nMau coba data lain? (y/n): ").strip().lower()
                if lagi != 'y':
                    break
    except (EOFError, KeyboardInterrupt):
        print("\nInput manual dihentikan.")


# 9. Mode 1: Training dari Awal
def mode_train_from_scratch():
    """
    Alur lengkap: Load data -> Preprocess -> EDA -> Split -> Training -> Evaluasi -> Ekspor JSON.
    """
    # Load dan preprocess data
    df = load_and_preprocess(FILEPATH)
    
    # Jalankan visualisasi info data (EDA)
    run_eda(df)
    
    # Bagi data jadi Train, Val, Test
    train_df, val_df, test_df = train_val_test_split(df)
    
    # Pisahin tipe fitur kategorik dan numerik
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
    
    X_val = val_df[cat_features + num_features]
    y_val = val_df[TARGET_COL]
    
    X_test = test_df[cat_features + num_features]
    y_test = test_df[TARGET_COL]

    # Training model
    print("\n[STEP 4] Melatih model Naive Bayes...")
    model = MixedNaiveBayes(cat_cols=cat_features, num_cols=num_features)
    model.fit(X_train, y_train)
    print("  -> Model kelar dilatih di data training!")

    # Ekspor model ke file JSON
    print("\n[STEP 5] Mengekspor model hasil latihan ke JSON...")
    save_model_to_json(model, "model_naive_bayes.json")

    # Cek bobot awal prior kelas
    print("\nPrior Probability hasil training:")
    for cls in model.classes:
        label = "Rendah (0)" if cls == 0 else "Tinggi (1)"
        print(f"  P({label}) = {model.priors[cls]:.4f}")

    # Tes performa ke data Validation
    print("\n[STEP 6] Evaluasi pada data Validation...")
    val_preds = model.predict(X_val)
    val_metrics = calculate_metrics(y_val, val_preds)
    print_evaluation_report("Validation", val_metrics)

    # Tes performa ke data Testing
    print("\n[STEP 7] Evaluasi pada data Testing...")
    test_preds = model.predict(X_test)
    test_metrics = calculate_metrics(y_test, test_preds)
    print_evaluation_report("Testing", test_metrics)

    # Demo prediksi & input interaktif
    run_demo_predictions(model)
    run_interactive_input(model)


# 10. Mode 2: Muat Model dari Berkas JSON
def mode_load_from_json():
    """
    Alur cepat: Langsung muat model yang sudah dilatih dari file JSON, lalu evaluasi & prediksi.
    """
    json_file = "model_naive_bayes.json"
    
    if not os.path.exists(json_file):
        print(f"\n  [ERROR] File model '{json_file}' tidak ditemukan!")
        print("  Jalankan Mode 1 (Training dari Awal) dulu untuk membuat file model.")
        return
    
    print(f"\n[STEP 1] Memuat model dari berkas '{json_file}'...")
    model = load_model_from_json(json_file)
    
    # Cek bobot awal prior kelas
    print("\nPrior Probability dari model JSON:")
    for cls in model.classes:
        label = "Rendah (0)" if cls == 0 else "Tinggi (1)"
        print(f"  P({label}) = {model.priors[cls]:.4f}")
    
    # Load data untuk evaluasi
    print(f"\n[STEP 2] Memuat dataset untuk evaluasi...")
    df = load_and_preprocess(FILEPATH)
    
    # Bagi data (pakai seed yang sama supaya split konsisten)
    train_df, val_df, test_df = train_val_test_split(df)
    
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
    
    # Evaluasi pada data Validation
    print("\n[STEP 3] Evaluasi model JSON pada data Validation...")
    val_preds = model.predict(X_val)
    val_metrics = calculate_metrics(y_val, val_preds)
    print_evaluation_report("Validation (Model JSON)", val_metrics)
    
    # Evaluasi pada data Testing
    print("\n[STEP 4] Evaluasi model JSON pada data Testing...")
    test_preds = model.predict(X_test)
    test_metrics = calculate_metrics(y_test, test_preds)
    print_evaluation_report("Testing (Model JSON)", test_metrics)
    
    # Demo prediksi & input interaktif
    run_demo_predictions(model)
    run_interactive_input(model)


# 11. Main Program (Menu Utama)
def main():
    print("=" * 70)
    print("      SISTEM PREDIKSI TINGKAT STRES MAHASISWA")
    print("    Metode: Mixed Gaussian & Categorical Naive Bayes")
    print("=" * 70)
    
    print("\n  Pilih mode yang ingin dijalankan:")
    print("  [1] Training dari Awal (Load Data -> Training -> Evaluasi -> Ekspor JSON)")
    print("  [2] Muat Model dari JSON (Langsung pakai model yang sudah dilatih)")
    
    try:
        if not sys.stdin.isatty():
            # Non-interactive: default ke Mode 1
            print("\n  [INFO] Non-interactive environment. Otomatis jalankan Mode 1.")
            pilihan = "1"
        else:
            pilihan = input("\n  Masukkan pilihan (1/2): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n  Input dibatalkan.")
        return
    
    if pilihan == "1":
        print("\n" + "-" * 70)
        print("  >> Mode 1: TRAINING DARI AWAL")
        print("-" * 70)
        mode_train_from_scratch()
    elif pilihan == "2":
        print("\n" + "-" * 70)
        print("  >> Mode 2: MUAT MODEL DARI JSON")
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
