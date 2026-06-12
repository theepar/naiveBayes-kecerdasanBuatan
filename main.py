import os
import math
import random
import sys
import pandas as pd

# ===================== KONFIGURASI =====================
FILEPATH = "student_lifestyle_dataset.csv"
TARGET_COL = "Stress_Level"
RANDOM_SEED = 42


# ======================== 1. LOAD & PREPROCESS DATA ========================
def load_and_preprocess(filepath):
    """
    Membaca dataset, menangani missing values (imputasi),
    dan mengembalikan pandas DataFrame yang bersih.
    """
    print(f"\n[STEP 1] Memuat dataset dari '{filepath}'...")
    
    # Jika file tidak ada secara lokal, unduh via kagglehub dan salin
    if not os.path.exists(filepath):
        print(f"  [INFO] File '{filepath}' tidak ditemukan secara lokal.")
        print("  [INFO] Mengunduh dataset dari Kaggle menggunakan kagglehub...")
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
            print(f"  [ERROR] Gagal mengunduh dataset: {e}")
            print("  Pastikan koneksi internet aktif atau file sudah diletakkan di folder kerja.")
            sys.exit(1)

    df = pd.read_csv(filepath)
    print(f"  -> Total data termuat: {len(df)} baris, {len(df.columns)} kolom.")

    print("\n[STEP 2] Preprocessing data (Imputasi Missing Values)...")
    
    # 1. Imputasi fitur kategorikal (Student_Type) dengan Modus
    if "Student_Type" in df.columns:
        missing_cat = df["Student_Type"].isnull().sum()
        if missing_cat > 0:
            mode_val = df["Student_Type"].mode()[0]
            df["Student_Type"] = df["Student_Type"].fillna(mode_val)
            print(f"  -> {missing_cat} missing values pada 'Student_Type' diisi dengan Modus: '{mode_val}'")
        else:
            print("  -> Tidak ada missing values pada 'Student_Type'.")

    # 2. Imputasi fitur numerik dengan Rata-rata (Mean)
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if TARGET_COL in num_cols:
        num_cols.remove(TARGET_COL)  # Target tidak perlu diimputasi (karena tidak ada yang null)
    
    for col in num_cols:
        missing_num = df[col].isnull().sum()
        if missing_num > 0:
            mean_val = df[col].mean()
            df[col] = df[col].fillna(mean_val)
            print(f"  -> {missing_num} missing values pada '{col}' diisi dengan Rata-rata: {mean_val:.4f}")
            
    print("  -> Preprocessing selesai. Data bersih siap digunakan.")
    return df


# ======================== 2. EXPLORATORY DATA ANALYSIS (EDA) ========================
def run_eda(df):
    """
    Menampilkan analisis data eksploratif sederhana untuk memahami karakteristik dataset.
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
        
    print("\n3. Rata-rata Nilai Fitur Numerik Berdasarkan Tingkat Stres:")
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if TARGET_COL in num_cols:
        grouped = df.groupby(TARGET_COL)[num_cols].mean()
        # Rename index untuk keterbacaan
        grouped.index = [f"Stres Rendah (0)", f"Stres Tinggi (1)"]
        print(grouped.to_string(float_format=lambda x: f"{x:.2f}"))
        
    print("=" * 60 + "\n")


# ======================== 3. SPLIT DATA ========================
def train_val_test_split(df, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=RANDOM_SEED):
    """
    Membagi dataset secara acak menjadi data Training, Validation, dan Testing.
    """
    print(f"\n[STEP 3] Membagi data (Train: {train_ratio*100:.0f}%, Val: {val_ratio*100:.0f}%, Test: {test_ratio*100:.0f}%)...")
    
    # Shuffle data
    shuffled_df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    n = len(shuffled_df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = shuffled_df.iloc[:train_end]
    val_df = shuffled_df.iloc[train_end:val_end]
    test_df = shuffled_df.iloc[val_end:]
    
    print(f"  -> Ukuran Data Training   : {len(train_df)} baris")
    print(f"  -> Ukuran Data Validation : {len(val_df)} baris")
    print(f"  -> Ukuran Data Testing    : {len(test_df)} baris")
    
    return train_df, val_df, test_df


# ======================== 4. MIXED NAIVE BAYES CLASS CLASS ========================
class MixedNaiveBayes:
    """
    Klasifikasi Naive Bayes Campuran (from scratch) yang menangani:
      - Fitur kategorikal menggunakan distribusi frekuensi dengan Laplace Smoothing.
      - Fitur numerik/kontinu menggunakan fungsi densitas probabilitas Gaussian (Normal).
    """
    def __init__(self, cat_cols=None, num_cols=None):
        self.cat_cols = cat_cols if cat_cols is not None else []
        self.num_cols = num_cols if num_cols is not None else []
        self.classes = []
        self.priors = {}
        # Likelihood kategorik: self.cat_likelihoods[kelas][kolom][nilai] = prob
        self.cat_likelihoods = {}
        # Parameter numerik: self.num_params[kelas][kolom] = (mean, variance)
        self.num_params = {}
        # Nilai unik kategorik untuk Laplace smoothing
        self.cat_unique_vals = {}

    def fit(self, X, y):
        self.classes = sorted(y.unique().tolist())
        total_samples = len(y)
        
        # 1. Hitung Prior Probability P(C) untuk setiap kelas
        for cls in self.classes:
            self.priors[cls] = sum(y == cls) / total_samples
            
        # Catat semua nilai unik pada fitur kategorikal dari data training
        for col in self.cat_cols:
            self.cat_unique_vals[col] = sorted(X[col].unique().tolist())
            
        # 2. Hitung Likelihood Parameter per Kelas
        for cls in self.classes:
            self.cat_likelihoods[cls] = {}
            self.num_params[cls] = {}
            
            # Filter baris yang sesuai dengan kelas ini
            X_cls = X[y == cls]
            n_cls = len(X_cls)
            
            # A. Fitur Kategorikal: Laplace Smoothing
            for col in self.cat_cols:
                self.cat_likelihoods[cls][col] = {}
                counts = X_cls[col].value_counts()
                unique_vals = self.cat_unique_vals[col]
                k = len(unique_vals)  # Jumlah nilai unik untuk smoothing
                
                for val in unique_vals:
                    val_count = counts.get(val, 0)
                    # Rumus Laplace: (count + 1) / (n_cls + k)
                    self.cat_likelihoods[cls][col][val] = (val_count + 1) / (n_cls + k)
            
            # B. Fitur Numerik: Estimasi Mean (rata-rata) & Variance (variansi)
            for col in self.num_cols:
                mean = X_cls[col].mean()
                var = X_cls[col].var()
                # Jika variansi 0 (atau NaN karena data tunggal), beri nilai sangat kecil (epsilon)
                if var == 0 or pd.isna(var):
                    var = 1e-9
                self.num_params[cls][col] = (mean, var)

    def _gaussian_pdf(self, x, mean, var):
        """Menghitung probabilitas Gauss P(x | mean, var)"""
        exponent = math.exp(-((x - mean) ** 2) / (2 * var))
        return (1.0 / math.sqrt(2 * math.pi * var)) * exponent

    def predict_single(self, sample):
        """Memprediksi kelas untuk satu sampel baris data"""
        log_scores = {}
        
        for cls in self.classes:
            # Mulai dengan log dari Prior P(C)
            log_prob = math.log(self.priors[cls])
            
            # Tambahkan log-likelihood fitur kategorikal
            for col in self.cat_cols:
                val = sample[col]
                # Jika nilai ada di training likelihood, gunakan nilainya
                if val in self.cat_likelihoods[cls][col]:
                    prob = self.cat_likelihoods[cls][col][val]
                else:
                    # Fallback Laplace jika ada kategori tak dikenal
                    k = len(self.cat_unique_vals[col])
                    prob = 1.0 / (k + 1)
                log_prob += math.log(prob)
                
            # Tambahkan log-likelihood fitur numerik (Gaussian)
            for col in self.num_cols:
                val = float(sample[col])
                mean, var = self.num_params[cls][col]
                prob = self._gaussian_pdf(val, mean, var)
                # Batasi probabilitas minimum untuk menghindari log(0)
                if prob < 1e-15:
                    prob = 1e-15
                log_prob += math.log(prob)
                
            log_scores[cls] = log_prob
            
        # Pilih kelas dengan log-probability tertinggi (argmax)
        best_class = max(log_scores, key=log_scores.get)
        return best_class, log_scores

    def predict(self, X):
        """Memprediksi kelas untuk seluruh baris data di DataFrame X"""
        predictions = []
        for _, row in X.iterrows():
            pred, _ = self.predict_single(row)
            predictions.append(pred)
        return predictions


# ======================== 5. EVALUASI METRIKS ========================
def calculate_metrics(y_true, y_pred):
    """
    Menghitung metrik evaluasi klasifikasi biner dari nol (from scratch):
    Akurasi, Presisi, Recall, F1-Score, dan Confusion Matrix.
    """
    y_true = list(y_true)
    y_pred = list(y_pred)
    n = len(y_true)
    
    tp = 0  # True Positive (Aktual 1, Prediksi 1)
    fp = 0  # False Positive (Aktual 0, Prediksi 1)
    tn = 0  # True Negative (Aktual 0, Prediksi 0)
    fn = 0  # False Negative (Aktual 1, Prediksi 0)
    
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
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn
    }


def print_evaluation_report(name, metrics):
    """Mencetak laporan hasil metrik evaluasi secara rapi"""
    print("\n" + "=" * 50)
    print(f" LAPORAN EVALUASI: DATA {name.upper()}")
    print("=" * 50)
    print(f"  Akurasi   : {metrics['accuracy'] * 100:.2f}%")
    print(f"  Presisi   : {metrics['precision'] * 100:.2f}%")
    print(f"  Recall    : {metrics['recall'] * 100:.2f}%")
    print(f"  F1-Score  : {metrics['f1_score'] * 100:.2f}%")
    print("-" * 50)
    print("  Confusion Matrix:")
    print("                      Prediksi: Rendah (0)    Prediksi: Tinggi (1)")
    print(f"  Aktual: Rendah (0)         {metrics['tn']:^10}               {metrics['fp']:^10}")
    print(f"  Aktual: Tinggi (1)         {metrics['fn']:^10}               {metrics['tp']:^10}")
    print("=" * 50)


# ======================== 6. MAIN ORCHESTRATOR ========================
def main():
    print("=" * 70)
    print("      SISTEM PREDIKSI TINGKAT STRES MAHASISWA")
    print("    Metode: Mixed Gaussian & Categorical Naive Bayes")
    print("=" * 70)

    # 1. Load dan Preprocess Data
    df = load_and_preprocess(FILEPATH)
    
    # 2. Jalankan EDA
    run_eda(df)
    
    # 3. Split Dataset (70% Train, 15% Val, 15% Test)
    train_df, val_df, test_df = train_val_test_split(df)
    
    # Kelompokkan Fitur berdasarkan tipe datanya
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
    
    # Pisahkan fitur dan label
    X_train = train_df[cat_features + num_features]
    y_train = train_df[TARGET_COL]
    
    X_val = val_df[cat_features + num_features]
    y_val = val_df[TARGET_COL]
    
    X_test = test_df[cat_features + num_features]
    y_test = test_df[TARGET_COL]

    # 4. Training (Fitting Model)
    print("\n[STEP 4] Melatih model Naive Bayes...")
    model = MixedNaiveBayes(cat_cols=cat_features, num_cols=num_features)
    model.fit(X_train, y_train)
    print("  -> Model berhasil dilatih pada data training!")

    # Tampilkan Prior Probability hasil training
    print("\nPrior Probability hasil training:")
    for cls in model.classes:
        label = "Rendah (0)" if cls == 0 else "Tinggi (1)"
        print(f"  P({label}) = {model.priors[cls]:.4f}")

    # 5. Prediksi dan Evaluasi data Validation
    print("\n[STEP 5] Evaluasi pada data Validation...")
    val_preds = model.predict(X_val)
    val_metrics = calculate_metrics(y_val, val_preds)
    print_evaluation_report("Validation", val_metrics)

    # 6. Prediksi dan Evaluasi data Testing
    print("\n[STEP 6] Evaluasi pada data Testing...")
    test_preds = model.predict(X_test)
    test_metrics = calculate_metrics(y_test, test_preds)
    print_evaluation_report("Testing", test_metrics)

    # 7. Demo Uji Kasus Spesifik
    print("\n" + "=" * 70)
    print(" DEMO PREDIKSI KASUS KHUSUS (UJI)")
    print("=" * 70)
    
    uji_kasus = [
        {
            # Kasus A: Beban kuliah tinggi, tidur kurang, support rendah
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
            # Kasus B: Kehidupan seimbang, tidur cukup, support tinggi
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

    # 8. Menu Input Interaktif (jika di terminal interaktif)
    try:
        if not sys.stdin.isatty():
            print("\n[INFO] Deteksi lingkungan non-interaktif. Melewati input interaktif.")
        else:
            print("\n" + "=" * 70)
            print(" INPUT INTERAKTIF (UJI DATA BARU)")
            print("=" * 70)
            tanya_user = input("Ingin mencoba prediksi data Anda sendiri? (y/n): ").strip().lower()
            if tanya_user == 'y':
                while True:
                    print("\nSilakan masukkan data Anda:")
                    try:
                        tipe = input("- Tipe Mahasiswa (college/school/working_student): ").strip().lower()
                        if tipe not in ['college', 'school', 'working_student']:
                            tipe = 'college'
                            print("  (Input tidak valid, otomatis diset ke 'college')")
                            
                        sleep = float(input("- Jam Tidur per Hari (misal 6.5): "))
                        study = float(input("- Jam Belajar per Hari (misal 4.0): "))
                        socmed = float(input("- Jam Sosial Media per Hari (misal 2.0): "))
                        attend = float(input("- Persentase Kehadiran Kelas (0 - 100): "))
                        pressure = float(input("- Skala Tekanan Ujian (1 - 10): "))
                        support = float(input("- Skala Dukungan Keluarga (1 - 10): "))
                        month = float(input("- Bulan Akademik Saat Ini (1 - 12): "))
                        
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
                        print("  [ERROR] Masukan angka tidak valid. Silakan coba lagi.")
                        
                    lagi = input("\nCoba data lain? (y/n): ").strip().lower()
                    if lagi != 'y':
                        break
    except (EOFError, KeyboardInterrupt):
        print("\nDemo interaktif dihentikan.")

    print("\n" + "=" * 70)
    print("  Program Selesai. Terima kasih!")
    print("=" * 70)


if __name__ == "__main__":
    main()
