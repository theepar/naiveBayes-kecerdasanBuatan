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
    Fungsi buat membaca dataset dan membersihkan missing values (imputasi).
    Menampilkan visualisasi tabel missing-values sebelum/sesudah, dan menggambar grafik perbandingan.
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
    total_rows = len(df)
    print(f"  -> Total data ke-load: {total_rows} baris, {len(df.columns)} kolom.")

    print("\n[STEP 2] Preprocessing & Imputasi Missing Values...")
    
    # 1. Catat status missing values sebelum imputasi untuk divisualisasikan
    missing_before = df.isnull().sum().to_dict()
    missing_after = {}
    imputation_methods = {}
    
    # 2. Proses Imputasi Kolom Kategorikal (Student_Type) menggunakan Modus
    if "Student_Type" in df.columns:
        missing_cat = missing_before.get("Student_Type", 0)
        if missing_cat > 0:
            mode_val = df["Student_Type"].mode()[0]
            df["Student_Type"] = df["Student_Type"].fillna(mode_val)
            imputation_methods["Student_Type"] = f"Modus ({mode_val})"
        else:
            imputation_methods["Student_Type"] = "-"
        missing_after["Student_Type"] = df["Student_Type"].isnull().sum()

    # 3. Proses Imputasi Kolom Numerik menggunakan Mean (Rata-rata)
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
        
    # Tambahkan kolom lain yang belum tercatat (jika ada)
    for col in df.columns:
        if col not in missing_after:
            missing_after[col] = df[col].isnull().sum()
            imputation_methods[col] = "-"

    # 4. Tampilkan Tabel Visualisasi Laporan Preprocessing
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
    
    # 5. Visualisasi Plot Grafis (Opsional jika matplotlib terpasang)
    try:
        import matplotlib.pyplot as plt
        print("\n  [INFO] Menampilkan grafik perbandingan missing values...")
        columns = list(missing_before.keys())
        before_vals = [missing_before[col] for col in columns]
        after_vals = [missing_after[col] for col in columns]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        y = range(len(columns))
        width = 0.35
        
        rects1 = ax.barh([i + width/2 for i in y], before_vals, width, label='Sebelum Imputasi', color='#E74C3C')
        rects2 = ax.barh([i - width/2 for i in y], after_vals, width, label='Setelah Imputasi (Bersih)', color='#2ECC71')
        
        # Tambahkan label angka di samping setiap bar agar nilai terlihat jelas
        for rect in rects1:
            val = rect.get_width()
            if val > 0:  # Hanya tampilkan label jika nilainya bukan 0
                ax.annotate(f'{int(val)}',
                            xy=(val, rect.get_y() + rect.get_height() / 2),
                            xytext=(5, 0),
                            textcoords="offset points",
                            ha='left', va='center', fontsize=9, color='black', fontweight='bold')
                        
        for rect in rects2:
            val = rect.get_width()
            if val > 0:  # Hanya tampilkan label jika nilainya bukan 0
                ax.annotate(f'{int(val)}',
                            xy=(val, rect.get_y() + rect.get_height() / 2),
                            xytext=(5, 0),
                            textcoords="offset points",
                            ha='left', va='center', fontsize=9, color='black', fontweight='bold')
        
        ax.set_xlabel('Jumlah Nilai Kosong')
        ax.set_title('Visualisasi Perbandingan Nilai Kosong Sebelum vs Setelah Preprocessing')
        ax.set_yticks(y)
        ax.set_yticklabels(columns)
        ax.invert_yaxis()  # Supaya urutan kolom sama seperti di tabel (dari atas ke bawah)
        
        # Set limit X agar label teks di samping bar tidak terpotong
        max_val = max(before_vals) if before_vals else 0
        ax.set_xlim(0, max_val * 1.15)
        
        ax.legend()
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("\n  [INFO] Jalankan 'pip install matplotlib' jika ingin melihat grafik visual secara lokal.")
        
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
    Model Klasifikasi Naive Bayes untuk Data Campuran (Mixed Naive Bayes).
    
    Model ini dirancang untuk dapat menangani dua jenis tipe data fitur sekaligus:
    1. Fitur Kategorikal: Dihitung menggunakan probabilitas tabel frekuensi dengan
       Laplace Smoothing untuk mencegah masalah zero-probability (peluang nol).
    2. Fitur Numerik Kontinu: Dihitung menggunakan Probability Density Function (PDF)
       dari Distribusi Gaussian (Normal). Rata-rata (mean) dan variansi (variance)
       dihitung untuk setiap kelas.
    """
    def __init__(self, cat_cols=None, num_cols=None):
        self.cat_cols = cat_cols if cat_cols is not None else []
        self.num_cols = num_cols if num_cols is not None else []
        self.classes = []
        self.priors = {}
        
        # Likelihood kategorik: self.cat_likelihoods[kelas][fitur][nilai] = P(nilai_fitur | kelas)
        self.cat_likelihoods = {}
        
        # Parameter numerik: self.num_params[kelas][fitur] = (mean, variance)
        self.num_params = {}
        
        # Menyimpan daftar nilai unik dari fitur kategorikal yang ditemukan saat training
        # Penting untuk menentukan ukuran vocabulary (V) dalam Laplace Smoothing
        self.cat_unique_vals = {}

    def fit(self, X, y):
        """
        Melatih model Naive Bayes dengan menghitung probabilitas Prior dan Parameter Likelihood.
        
        Langkah-langkah Training:
        1. Identifikasi kelas target yang unik (misal: 0 = Stres Rendah, 1 = Stres Tinggi).
        2. Hitung Prior Probability P(C) untuk tiap kelas.
        3. Catat nilai unik dari fitur kategorikal untuk Laplace Smoothing.
        4. Hitung Parameter Likelihood untuk setiap kelas:
           - Kategorikal: Tabel probabilitas dengan Laplace Smoothing.
           - Numerik: Mean (rata-rata) dan Variansi (variance) untuk PDF Gaussian.
        """
        # --- Langkah 1: Identifikasi Kelas Unik ---
        self.classes = sorted(y.unique().tolist())
        total_samples = len(y)
        
        # --- Langkah 2: Hitung Prior Probability P(C) ---
        # Rumus: P(C) = Jumlah sampel kelas C / Total seluruh sampel data
        for class_label in self.classes:
            class_count = sum(y == class_label)
            self.priors[class_label] = class_count / total_samples
            
        # --- Langkah 3: Catat Nilai Unik Fitur Kategorikal ---
        # Diperlukan untuk menghitung jumlah kategori unik (V) pada Laplace Smoothing
        for feature_name in self.cat_cols:
            self.cat_unique_vals[feature_name] = sorted(X[feature_name].unique().tolist())
            
        # --- Langkah 4: Hitung Likelihood dan Parameter Distribusi ---
        for class_label in self.classes:
            self.cat_likelihoods[class_label] = {}
            self.num_params[class_label] = {}
            
            # Filter baris data yang sesuai dengan kelas saat ini
            X_class = X[y == class_label]
            class_sample_count = len(X_class)
            
            # A. Pemodelan Fitur Kategorikal (Laplace Smoothing)
            # Rumus: P(x_i | C) = (Jumlah kemunculan x_i di kelas C + 1) / (N_c + V_i)
            # N_c = jumlah sampel kelas C, V_i = jumlah kategori unik pada fitur i
            for feature_name in self.cat_cols:
                self.cat_likelihoods[class_label][feature_name] = {}
                counts = X_class[feature_name].value_counts()
                unique_vals = self.cat_unique_vals[feature_name]
                category_count = len(unique_vals)  # Nilai V_i (Laplace parameter)
                
                for feature_value in unique_vals:
                    val_count = counts.get(feature_value, 0)
                    # Rumus Laplace Smoothing untuk menghindari peluang nol jika kategori tak muncul
                    probability = (val_count + 1) / (class_sample_count + category_count)
                    self.cat_likelihoods[class_label][feature_name][feature_value] = probability
            
            # B. Pemodelan Fitur Numerik Kontinu (Gaussian Parameter)
            # Mengasumsikan data terdistribusi normal. Kita hitung Mean (mu) dan Variansi (sigma^2).
            for feature_name in self.num_cols:
                mean_val = X_class[feature_name].mean()
                variance = X_class[feature_name].var()
                
                # Penanganan khusus jika variansi bernilai 0 atau NaN (untuk mencegah pembagian nol di PDF)
                if variance == 0 or pd.isna(variance):
                    variance = 1e-9
                self.num_params[class_label][feature_name] = (mean_val, variance)

    def _gaussian_pdf(self, x, mean_val, variance):
        """
        Menghitung Probability Density Function (PDF) dari Distribusi Gaussian (Normal).
        Digunakan untuk memperkirakan probabilitas kontinu: P(x | kelas)
        
        Rumus Gaussian PDF:
        f(x; mu, sigma^2) = [ 1 / sqrt(2 * pi * variance) ] * e^( - (x - mu)^2 / (2 * variance) )
        
        Dimana:
        - mu (mean_val)     : rata-rata nilai fitur pada kelas tersebut.
        - sigma^2 (variance): variansi nilai fitur pada kelas tersebut.
        """
        # 1. Hitung bagian pembagi (koefisien normalisasi): 1 / sqrt(2 * pi * variance)
        coefficient = 1.0 / math.sqrt(2 * math.pi * variance)
        
        # 2. Hitung nilai eksponen: -((x - mean)^2) / (2 * variance)
        exponent = math.exp(-((x - mean_val) ** 2) / (2 * variance))
        
        # 3. Kembalikan hasil perkalian keduanya
        return coefficient * exponent

    def predict_single(self, sample, threshold_offset=0.0):
        """
        Memprediksi kelas untuk satu baris sampel data.
        
        Penting untuk Menghindari Underflow:
        Karena nilai peluang berkisar antara 0 dan 1, mengalikan banyak peluang akan menghasilkan
        nilai yang sangat kecil mendekati nol (numerical underflow).
        Oleh karena itu, kita ubah perkalian menjadi penjumlahan dengan logaritma natural:
        P(C | X) = P(C) * P(x_1|C) * P(x_2|C) ...
        Menjadi:
        log P(C | X) = log P(C) + log P(x_1|C) + log P(x_2|C) ...
        """
        log_scores = {}
        
        for class_label in self.classes:
            # 1. Mulai dari nilai log dari Prior Probability: log P(C)
            log_prob = math.log(self.priors[class_label])
            
            # 2. Tambahkan log likelihood fitur kategorikal
            for feature_name in self.cat_cols:
                feature_value = sample[feature_name]
                
                # Cek jika nilai kategori ada di kamus pelatihan
                if feature_value in self.cat_likelihoods[class_label][feature_name]:
                    probability = self.cat_likelihoods[class_label][feature_name][feature_value]
                else:
                    # Fallback jika ada kategori baru di data uji: gunakan Laplace smoothing dasar
                    category_count = len(self.cat_unique_vals[feature_name])
                    probability = 1.0 / (category_count + 1)
                    
                log_prob += math.log(probability)
                
            # 3. Tambahkan log likelihood fitur numerik menggunakan Gaussian PDF
            for feature_name in self.num_cols:
                feature_value = float(sample[feature_name])
                mean_val, variance = self.num_params[class_label][feature_name]
                
                probability = self._gaussian_pdf(feature_value, mean_val, variance)
                
                # Batasi nilai probabilitas agar tidak 0 (untuk mencegah error log(0))
                if probability < 1e-15:
                    probability = 1e-15
                    
                log_prob += math.log(probability)
                
            log_scores[class_label] = log_prob
            
        # 4. Penentuan Keputusan Akhir (Argmax dengan Modifikasi Threshold)
        # Jika klasifikasi biner 0 (Rendah) dan 1 (Tinggi), kita gunakan threshold_offset.
        # Secara matematis:
        # Prediksi 1 jika: log P(C=1|X) - log P(C=0|X) > threshold_offset
        # Yang ekuivalen dengan: log P(C=1|X) > log P(C=0|X) + threshold_offset
        # Jika threshold_offset bernilai negatif (misal -0.5), model akan lebih cenderung memilih kelas 1.
        # Hal ini sangat berguna jika kita ingin meminimalkan False Negative (stres tinggi terlewat diprediksi rendah).
        if len(self.classes) == 2 and self.classes == [0, 1]:
            log_difference = log_scores[1] - log_scores[0]
            if log_difference > threshold_offset:
                return 1, log_scores
            else:
                return 0, log_scores
        else:
            # Standar multi-kelas: pilih kelas dengan log probability tertinggi (argmax)
            best_class = max(log_scores, key=log_scores.get)
            return best_class, log_scores

    def predict(self, X, threshold_offset=0.0):
        """
        Memprediksi sekumpulan data (pandas DataFrame) baris demi baris.
        
        Parameter:
        X                : pandas.DataFrame -> Fitur data uji
        threshold_offset : float            -> Nilai pergeseran batas keputusan biner
        
        Mengembalikan:
        list -> Berisi hasil prediksi kelas (0 atau 1) untuk setiap baris
        """
        predictions = []
        for _, row in X.iterrows():
            pred, _ = self.predict_single(row, threshold_offset)
            predictions.append(pred)
        return predictions


# 5. Evaluasi Metriks
def calculate_metrics(y_true, y_pred):
    """
    Menghitung metrik performa klasifikasi secara manual dari awal tanpa library eksternal.
    
    Langkah:
    1. Hitung Confusion Matrix: True Positive (TP), False Positive (FP), 
       True Negative (TN), dan False Negative (FN).
    2. Hitung metrik turunan: Akurasi, Error Rate, Presisi, Recall, Specificity,
       False Positive Rate (FPR), False Negative Rate (FNR), dan F1-Score.
       
    Definisi Istilah:
    - TP (True Positive)  : Kelas Aktual 1 (Tinggi), diprediksi 1 (Tinggi)
    - FP (False Positive) : Kelas Aktual 0 (Rendah), diprediksi 1 (Tinggi)
    - TN (True Negative)  : Kelas Aktual 0 (Rendah), diprediksi 0 (Rendah)
    - FN (False Negative) : Kelas Aktual 1 (Tinggi), diprediksi 0 (Rendah)
    """
    actual_labels = list(y_true)
    predicted_labels = list(y_pred)
    total_samples = len(actual_labels)
    
    # Inisialisasi elemen Confusion Matrix
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    # 1. Akumulasi Confusion Matrix
    for actual_val, predicted_val in zip(actual_labels, predicted_labels):
        if actual_val == 1 and predicted_val == 1:
            true_positives += 1
        elif actual_val == 0 and predicted_val == 1:
            false_positives += 1
        elif actual_val == 0 and predicted_val == 0:
            true_negatives += 1
        elif actual_val == 1 and predicted_val == 0:
            false_negatives += 1
            
    # 2. Perhitungan Metrik Turunan (disertai penanganan pembagian nol dengan fallback 0)
    # Akurasi: Proporsi prediksi benar dari seluruh data
    accuracy = (true_positives + true_negatives) / total_samples if total_samples > 0 else 0
    
    # Error Rate: Proporsi prediksi yang salah
    error_rate = 1 - accuracy
    
    # Presisi: Dari semua yang diprediksi positif, berapa yang sebenarnya positif
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    
    # Recall (Sensitivity/TPR): Dari semua yang sebenarnya positif, berapa yang berhasil dideteksi
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    # Specificity (TNR): Dari semua yang sebenarnya negatif, berapa yang berhasil dideteksi
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
    
    # FPR: Rasio kelas negatif yang salah dideteksi sebagai positif
    false_positive_rate = false_positives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
    
    # FNR: Rasio kelas positif yang salah dideteksi sebagai negatif
    false_negative_rate = false_negatives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    # F1-Score: Rata-rata harmonik antara Presisi dan Recall (keseimbangan keduanya)
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
    """
    Menampilkan laporan evaluasi model secara detail ke konsol,
    termasuk semua metrik performa utama dan tabel Confusion Matrix.
    """
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
