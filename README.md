# Student Stress Prediction using Mixed Naive Bayes

Proyek ini adalah implementasi sistem prediksi tingkat stres mahasiswa (`Stress_Level`: 0 untuk Rendah, 1 untuk Tinggi) menggunakan metode **Mixed Naive Bayes Classifier** yang dibangun dari nol (_from scratch_) tanpa menggunakan pustaka _machine learning_ eksternal seperti `scikit-learn`.

Model ini dirancang khusus untuk menangani tipe data campuran:

1. **Fitur Kategorikal (`Student_Type`)** menggunakan probabilitas tabel frekuensi dengan _Laplace Smoothing_.
2. **Fitur Numerik Kontinu (Jam Belajar, Tidur, Kehadiran, dll)** menggunakan rumus fungsi peluang **Gaussian (Normal) Distribution**.

---

## Struktur File Proyek

- **`main.py`**: Pipeline utama proyek. Meliputi proses pemuatan data, pembersihan (_imputasi_) nilai kosong, pemisahan dataset, proses latih (_training_), ekspor model ke format JSON, serta evaluasi otomatis dan demo prediksi interaktif.
- **`predict_from_json.py`**: Program pengujian terpisah yang memuat model langsung dari berkas JSON tanpa proses latihan ulang, membuktikan ekspor model berjalan dengan sempurna.
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

### 1. Proses Training & Evaluasi Utama

Jalankan perintah di bawah untuk memproses data, melatih model, melakukan ekspor model ke JSON, melihat hasil metrik performa, dan mencoba input manual secara interaktif:

```bash
python main.py
```

### 2. Pengujian Prediksi dari Berkas JSON

Jalankan perintah di bawah untuk menguji apakah model yang disimpan dalam berkas JSON dapat dimuat dan menghasilkan prediksi serta metrik evaluasi yang tepat:

```bash
python predict_from_json.py
```

---

## Performa Model

Evaluasi model pada data **Testing (15%)** yang belum pernah dilihat menghasilkan metrik performa sebagai berikut:

- **Akurasi**: `81.54%`
- **Presisi**: `73.75%`
- **Recall**: `60.93%`
- **F1-Score**: `66.73%`

### Confusion Matrix (Data Testing):

```text
                      Prediksi: Rendah (0)    Prediksi: Tinggi (1)
  Aktual: Rendah (0)            2411                     252
  Aktual: Tinggi (1)            454                      708
```
