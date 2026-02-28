# Implementasi Algoritma Support Vector Machine (SVM)
## Aspect-Based Sentiment Analysis (ABSA) pada Ulasan Aplikasi Streaming Film

Project ini merupakan implementasi algoritma Support Vector Machine (SVM) untuk melakukan
Aspect-Based Sentiment Analysis (ABSA) pada ulasan pengguna aplikasi streaming film.

Sistem dikembangkan sebagai bagian dari penelitian/skripsi dan terdiri dari dua komponen utama:
- Google Colab (pemodelan dan pengolahan data)
- Aplikasi Web berbasis Streamlit (visualisasi dan demonstrasi hasil)

---

## Komponen Sistem

### 1. Google Colab
Digunakan untuk:
- Scraping dan preprocessing data
- Pelabelan sentimen dan aspek
- Pelatihan model klasifikasi sentimen dan aspek menggunakan SVM
- Implementasi model pada dataset akhir
- Visualisasi hasil analisis
- Penyimpanan model (.pkl) dan evaluasi (.json)

### 2. Aplikasi Web Streamlit
Digunakan sebagai media demonstrasi hasil model secara interaktif, meliputi:
- Dashboard ringkasan analisis
- Analisis sentimen langsung (live test)
- Kontribusi kata (typo/slang dan keyword aspek)
- Manajemen saran melalui menu admin

---

## Cara Menjalankan Aplikasi Web
1. Clone repository ini
2. Buka folder project menggunakan Visual Studio Code
3. Buka terminal
4. Jalankan perintah:
   ```bash
   streamlit run app.py
