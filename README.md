# Student Stress Prediction using Mixed Naive Bayes

Proyek ini adalah implementasi sistem prediksi tingkat stres mahasiswa (`Stress_Level`: 0 untuk Rendah, 1 untuk Tinggi) menggunakan metode **Mixed Naive Bayes Classifier** yang dibangun dari nol (_from scratch_) tanpa menggunakan pustaka _machine learning_ eksternal seperti `scikit-learn`.

Model ini dirancang khusus untuk menangani tipe data campuran:

1. **Fitur Kategorikal (`Student_Type`)** menggunakan probabilitas tabel frekuensi dengan _Laplace Smoothing_.
2. **Fitur Numerik Kontinu (Jam Belajar, Tidur, Kehadiran, dll)** menggunakan rumus fungsi peluang **Gaussian (Normal) Distribution**.

---

## Struktur File Proyek

- **`main.py`**: Program utama proyek dengan sistem menu interaktif. Menyediakan dua mode operasi:
  - **Mode 1** — Training dari awal (load data → preprocessing → EDA → training → evaluasi → ekspor JSON).
  - **Mode 2** — Muat model dari berkas JSON yang sudah dilatih sebelumnya, lalu langsung evaluasi & prediksi tanpa training ulang.
- **`model_naive_bayes.json`**: Berkas JSON hasil ekspor parameter model (bobot prior, mean, variansi, dan tabel peluang kategorikal).
- **`student_lifestyle_dataset.csv`**: File dataset lokal yang berisi data riwayat aktivitas mahasiswa dan tingkat stresnya.

---

## Persiapan Lingkungan

Proyek ini membutuhkan Python 3.x dan pustaka `pandas` untuk pemrosesan tabel data awal, serta `kagglehub` apabila ingin mengunduh dataset secara otomatis.

Instal dependensi menggunakan perintah berikut:

```bash
pip install pandas kagglehub
```

---

## Cara Menjalankan Program

Jalankan perintah berikut untuk memulai program:

```bash
python main.py
```

Setelah dijalankan, akan muncul menu pilihan:

```
  Pilih mode yang ingin dijalankan:
  [1] Training dari Awal (Load Data -> Training -> Evaluasi -> Ekspor JSON)
  [2] Muat Model dari JSON (Langsung pakai model yang sudah dilatih)
```

### Mode 1: Training dari Awal

Memproses seluruh pipeline dari awal: memuat dataset, membersihkan data kosong (_imputasi_), menampilkan analisis data (EDA), membagi dataset (70% Training, 15% Validation, 15% Testing), melatih model Naive Bayes, mengekspor model ke JSON, dan mengevaluasi performa model. Di akhir tersedia demo prediksi dan input data manual secara interaktif.

### Mode 2: Muat Model dari JSON

Memuat model yang sudah dilatih langsung dari berkas `model_naive_bayes.json` tanpa proses training ulang, lalu menjalankan evaluasi pada data Validation & Testing untuk membuktikan bahwa ekspor/impor model berjalan sempurna. Fitur demo prediksi dan input interaktif juga tersedia.

---

## Metrik Evaluasi Model

Laporan evaluasi model mencakup metrik-metrik berikut:

| Metrik | Rumus | Keterangan |
|---|---|---|
| **Accuracy** | `(TP + TN) / (TP + TN + FP + FN)` | Rasio prediksi benar dari total data |
| **Error Rate** | `1 - Accuracy` | Tingkat kesalahan prediksi |
| **Precision** | `TP / (TP + FP)` | Ketepatan prediksi positif |
| **Recall / TPR** | `TP / (TP + FN)` | Sensitivitas mendeteksi kelas positif |
| **Specificity / TNR** | `TN / (TN + FP)` | Kemampuan mendeteksi kelas negatif |
| **False Positive Rate** | `FP / (TN + FP)` | Rasio kesalahan prediksi positif |
| **False Negative Rate** | `FN / (TP + FN)` | Rasio kesalahan prediksi negatif |
| **F1-Score** | `2 × (Precision × Recall) / (Precision + Recall)` | Keseimbangan Precision dan Recall |

---

## Performa Model

Evaluasi model pada data **Testing (15%)** yang belum pernah dilihat menghasilkan metrik performa sebagai berikut:

| Metrik | Skor |
|---|---|
| Akurasi (ACC) | `81.54%` |
| Error Rate | `18.46%` |
| Presisi | `73.75%` |
| Recall / TPR | `60.93%` |
| Specificity / TNR | `90.54%` |
| False Positive Rate | `9.46%` |
| False Negative Rate | `39.07%` |
| F1-Score | `66.73%` |

### Confusion Matrix (Data Testing):

```text
                      Prediksi: Rendah (0)    Prediksi: Tinggi (1)
  Aktual: Rendah (0)            2411                     252
  Aktual: Tinggi (1)            454                      708
```
