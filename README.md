# Student Stress Prediction using Naive Bayes

Proyek ini adalah implementasi sistem prediksi tingkat stres mahasiswa (`Stress_Level`: 0 untuk Rendah, 1 untuk Tinggi) menggunakan metode **Naive Bayes Classifier** yang dibangun dari nol (_from scratch_) tanpa menggunakan pustaka _machine learning_ eksternal seperti `scikit-learn`.

Model ini dirancang khusus untuk menangani tipe data campuran:

1. **Fitur Kategorikal (`Student_Type`)** menggunakan probabilitas tabel frekuensi dengan _Laplace Smoothing_.
2. **Fitur Numerik Kontinu (Jam Belajar, Tidur, Kehadiran, dll)** menggunakan rumus fungsi peluang **Gaussian (Normal) Distribution**.

---

## Struktur File Proyek

- **`main.py`**: Program utama proyek dengan sistem menu interaktif. Menyediakan dua mode operasi:
  - **Mode 1** — Training & Ekspor Model (load data → preprocessing → split → training → ekspor JSON).
  - **Mode 2** — Muat Model & Evaluasi (muat JSON → load data → EDA → split → evaluasi Validation & Testing → demo prediksi).
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
  [1] Training & Ekspor Model ke JSON
  [2] Muat Model JSON & Evaluasi (Gunakan Model)
```

### Mode 1: Training & Ekspor Model

Melatih model Naive Bayes menggunakan data training hasil pemisahan (70%) dari dataset, lalu menyimpannya langsung ke berkas `model_naive_bayes.json`. Tidak ada evaluasi performa atau demo prediksi di mode ini. Setelah alur selesai, program akan menanyakan secara interaktif apakah Anda ingin langsung melanjutkan ke **Mode 2** (`y/n`).

### Mode 2: Muat Model & Evaluasi

Memuat model yang sudah dilatih langsung dari berkas `model_naive_bayes.json` tanpa proses training ulang. Selanjutnya, memuat seluruh dataset untuk menampilkan Analisis Eksplorasi Data (EDA), memisahkan data pengujian (Validation & Testing), menghitung seluruh metrik performa model pada data pengujian, serta menampilkan demo prediksi untuk kasus uji khusus.

---

## Metrik Evaluasi Model

Laporan evaluasi model mencakup metrik-metrik berikut:

| Metrik                  | Rumus                                             | Keterangan                            |
| ----------------------- | ------------------------------------------------- | ------------------------------------- |
| **Accuracy**            | `(TP + TN) / (TP + TN + FP + FN)`                 | Rasio prediksi benar dari total data  |
| **Error Rate**          | `1 - Accuracy`                                    | Tingkat kesalahan prediksi            |
| **Precision**           | `TP / (TP + FP)`                                  | Ketepatan prediksi positif            |
| **Recall / TPR**        | `TP / (TP + FN)`                                  | Sensitivitas mendeteksi kelas positif |
| **Specificity / TNR**   | `TN / (TN + FP)`                                  | Kemampuan mendeteksi kelas negatif    |
| **False Positive Rate** | `FP / (TN + FP)`                                  | Rasio kesalahan prediksi positif      |
| **False Negative Rate** | `FN / (TP + FN)`                                  | Rasio kesalahan prediksi negatif      |
| **F1-Score**            | `2 × (Precision × Recall) / (Precision + Recall)` | Keseimbangan Precision dan Recall     |

---

## Performa Model

Evaluasi model pada data **Testing (15%)** yang belum pernah dilihat dengan konfigurasi `THRESHOLD_OFFSET = -0.5` (untuk meminimalkan False Negative) menghasilkan metrik performa sebagai berikut:

| Metrik              | Skor     |
| ------------------- | -------- |
| Akurasi (ACC)       | `80.52%` |
| Error Rate          | `19.48%` |
| Presisi             | `65.69%` |
| Recall / TPR        | `75.13%` |
| Specificity / TNR   | `82.88%` |
| False Positive Rate | `17.12%` |
| False Negative Rate | `24.87%` |
| F1-Score            | `70.09%` |

### Confusion Matrix (Data Testing):

```text
                      Prediksi: Rendah (0)    Prediksi: Tinggi (1)
  Aktual: Rendah (0)            2207                     456
  Aktual: Tinggi (1)            289                      873
```

> [!TIP]
> Nilai `THRESHOLD_OFFSET = -0.5` dikonfigurasi di bagian atas berkas [main.py](file:///d:/Kodingan\python\naive-bayes\main.py). Pengaturan ini menggeser batas keputusan (_decision boundary_) log-posterior sebesar `-0.5` untuk membuat model lebih sensitif terhadap deteksi kelas stres tinggi (1), sehingga menekan **False Negative** dari **454** menjadi hanya **289** (penurunan sebesar ~36%) dan meningkatkan **F1-Score** secara keseluruhan dari **66.73%** menjadi **70.09%**.
