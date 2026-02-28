import re
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# ==============================================================================
# 1. INISIALISASI SASTRAWI (Berat, jadi taruh di luar fungsi biar cuma load sekali)
# ==============================================================================
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Daftar stopwords & negasi (Bisa ditaruh global karena jarang berubah)
# Kita pakai set() biar pencariannya cepat
STOPWORDS = set("""
yang dan di ke dari untuk pada dengan adalah itu ini karena sebagai juga
agar supaya atau sehingga dalam sudah sangat lebih kurang saja hanya masih
bisa dapat harus akan jadi pun lah kah nya si sang para
""".split())

NEGASI = {"tidak","bukan","belum","jangan","tak","tdk","bkn","ga","gak","gk","nggak","ngga","enggak","kagak"}
KONJUNGSI_PEMISAH = {"tapi","namun","tetapi","walaupun","meskipun","padahal","sedangkan","cuma"}

# Update stopwords agar tidak menghapus kata penting
STOPWORDS = STOPWORDS - NEGASI - KONJUNGSI_PEMISAH

# ==============================================================================
# 2. FUNGSI UTAMA (SPLIT & CLEAN)
# ==============================================================================

def clean_basic(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text) # Hapus URL
    text = re.sub(r"\S+@\S+", " ", text) # Hapus Email
    text = re.sub(r"[^\x00-\x7F]+", " ", text) # Hapus Emoji/ASCII aneh
    text = re.sub(r"\b(wk+w*k+|ha+ha+|he+he+|xixixi+|wkwk+)\b", " ", text) # Hapus tawa
    text = re.sub(r'(.)\1{2,}', r'\1\1', text) # Hapus huruf berulang (baaaagus -> bagus)
    text = re.sub(r"[^a-z0-9\s]", " ", text) # Hapus simbol aneh
    text = re.sub(r"\s+", " ", text).strip() # Hapus spasi ganda
    return text

def handle_negation(text):
    """Menggabungkan kata negasi dengan kata setelahnya (tidak_suka)"""
    tokens = text.split()
    hasil = []
    i = 0
    while i < len(tokens):
        if tokens[i] in NEGASI and i+1 < len(tokens):
            hasil.append(tokens[i] + "_" + tokens[i+1])
            i += 2
        else:
            hasil.append(tokens[i])
            i += 1
    return " ".join(hasil)

def remove_stopwords(text):
    """Menghapus kata umum, TAPI menjaga kata yg sudah digabung (_)"""
    hasil = []
    for w in text.split():
        # Simpan kata jika: ada underscore (negasi), itu konjungsi, atau bukan stopword
        if "_" in w or w in KONJUNGSI_PEMISAH or w not in STOPWORDS:
            hasil.append(w)
    return " ".join(hasil)

def safe_stem(text):
    """Stemming hanya untuk kata biasa, jangan stem kata ber-underscore"""
    return " ".join([w if "_" in w else stemmer.stem(w) for w in text.split()])

def split_by_conjunction(text):
    """
    Memecah kalimat majemuk. Penting untuk analisis aspek.
    Contoh: 'aplikasi bagus tapi mahal' -> ['aplikasi bagus', 'mahal']
    """
    # Tambahkan spasi agar tidak memotong kata (misal: 'tetapi' tidak kena split 'tapi')
    text = " " + str(text).lower().strip() + " "
    
    # Ganti semua variasi konjungsi dengan tanda pipa |
    konjungsi_list = [" tapi ", " namun ", " tetapi ", " sedangkan ", " padahal ", " meskipun ", " walaupun ", " cuma "]
    
    for k in konjungsi_list:
        text = text.replace(k, " | ") 

    # Pecah dan bersihkan
    parts = [p.strip() for p in text.split("|") if p.strip() != ""]
    
    return parts if parts else [text.strip()]

# ==============================================================================
# 3. FUNGSI PREPROCESSING UTAMA (DENGAN LOAD KAMUS REAL-TIME)
# ==============================================================================

def preprocess_text(text):
    # 1. Cleaning Dasar
    text = clean_basic(text)
    
    # 2. NORMALISASI TYPO (Load Kamus di sini agar Update Real-time)
    try:
        # Load kedua kamus
        df_typo = pd.read_csv("kamus/kamus_typo.csv", header=None, names=["typo", "normal"])
        
        # Cek apakah kamus gaul ada, kalau tidak, pakai empty dataframe biar ga error
        try:
            df_gaul = pd.read_csv("kamus/kamus.csv")[["slang", "formal"]]
            df_gaul.columns = ["typo", "normal"]
        except:
            df_gaul = pd.DataFrame(columns=["typo", "normal"])

        # Gabung kamus
        kamus_all = pd.concat([df_typo, df_gaul], ignore_index=True)
        
        # Bersihkan spasi
        kamus_all["typo"] = kamus_all["typo"].astype(str).str.lower().str.strip()
        kamus_all["normal"] = kamus_all["normal"].astype(str).str.lower().str.strip()
        
        # Buat Dictionary Map
        normal_dict = dict(zip(kamus_all["typo"], kamus_all["normal"]))
        
    except Exception as e:
        # Fallback jika file error/hilang
        normal_dict = {}


    
    words = text.split()
    normalized_words = [normal_dict.get(w, w) for w in words]
    text = " ".join(normalized_words)

    # 3. Lanjut Pipeline
    text = handle_negation(text)
    text = remove_stopwords(text)
    text = safe_stem(text)
    
    return text

def split_by_conjunction(text):
    """
    Memecah kalimat majemuk.
    UPDATE: Sekarang mengenali singkatan konjungsi (tp, tpi, sdgkn, dll)
    """
    # Tambahkan spasi agar tidak memotong kata (safety)
    text = " " + str(text).lower().strip() + " "
    
    # Daftar konjungsi LENGKAP (Baku + Singkatan)
    # Pastikan pakai spasi di kiri kanan (" tp ") biar tidak memotong kata seperti 'htp'
    konjungsi_list = [
        " tapi ", " namun ", " tetapi ", " sedangkan ", " padahal ", " meskipun ", " walaupun ", " cuma ", " melainkan ",
        " tp ", " tpi ", " sdgkn ", " pdhl ", " mskpn ", " wlpn ", " cm " # <--- INI TAMBAHANNYA
    ]
    
    # Ganti semua variasi konjungsi dengan tanda pipa |
    for k in konjungsi_list:
        text = text.replace(k, " | ") 

    # Pecah dan bersihkan
    parts = [p.strip() for p in text.split("|") if p.strip() != ""]
    
    # Hapus konjungsi di awal kalimat hasil pecahan (opsional, biar bersih)
    # Contoh: "| tapi mahal" -> "mahal" (sudah terhandle strip di atas sebenarnya)
    
    return parts if parts else [text.strip()]