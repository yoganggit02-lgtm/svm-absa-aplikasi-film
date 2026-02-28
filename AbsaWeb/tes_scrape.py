from google_play_scraper import reviews, Sort

print("Mulai tes scrape WeTV...")

# ✅ Gunakan ID yang SAMA PERSIS dengan kode Colab kamu (tanpa titik)
app_id_wetv = 'com.tencent.qqlivei18n' 

result, _ = reviews(
    app_id_wetv,
    lang='id', 
    country='id', 
    count=10, 
    sort=Sort.NEWEST
)

print(f"Jumlah data ditemukan: {len(result)}")

if len(result) > 0:
    print("------------------------------------------------")
    print("Contoh ulasan:", result[0]['content'])
    print("------------------------------------------------")
    print("KESIMPULAN: Library Google Play Scraper SUKSES & AMAN! ✅")
else:
    print("Gagal, data masih kosong. Coba cek koneksi internet.")