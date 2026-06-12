import pandas as pd
from main import load_model_from_json, calculate_metrics, print_evaluation_report, train_val_test_split, FILEPATH

def main():
    print("=" * 70)
    print("    PENGUJIAN PREDIKSI MENGGUNAKAN MODEL DARI BERKAS JSON")
    print("         (Membuktikan Model Hasil Ekspor Berjalan Sukses)")
    print("=" * 70)
    
    # 1. Muat model langsung dari berkas JSON hasil ekspor
    model = load_model_from_json("model_naive_bayes.json")
    
    # 2. Muat dataset untuk mengambil data testing
    df = pd.read_csv(FILEPATH)
    
    # Lakukan preprocessing yang sama (imputasi data kosong)
    df["Student_Type"] = df["Student_Type"].fillna(df["Student_Type"].mode()[0])
    for col in df.select_dtypes(include=["number"]).columns:
        if col != "Stress_Level":
            df[col] = df[col].fillna(df[col].mean())
            
    # Split data menggunakan pembagian yang sama
    _, _, test_df = train_val_test_split(df)
    
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
    
    X_test = test_df[cat_features + num_features]
    y_test = test_df["Stress_Level"]
    
    # 3. Jalankan prediksi menggunakan model hasil muatan JSON
    print("\nMenjalankan prediksi pada data Testing...")
    test_preds = model.predict(X_test)
    
    # 4. Hitung metrik evaluasi
    test_metrics = calculate_metrics(y_test, test_preds)
    print_evaluation_report("Testing (Loaded from JSON)", test_metrics)
    
    # 5. Coba prediksi sampel manual untuk pembuktian
    print("\nUji coba satu sampel acak:")
    sample = {
        "Student_Type": "college",
        "Sleep_Hours": 4.5,
        "Study_Hours": 8.0,
        "Social_Media_Hours": 1.5,
        "Attendance": 65.0,
        "Exam_Pressure": 9.0,
        "Family_Support": 2.0,
        "Month": 5.0
    }
    pred, _ = model.predict_single(sample)
    label = "TINGKAT STRES TINGGI (1)" if pred == 1 else "TINGKAT STRES RENDAH (0)"
    print(f"  Input: {sample}")
    print(f"  -> Hasil Prediksi dari Model JSON: ** {label} **")
    print("=" * 70)

if __name__ == "__main__":
    main()
