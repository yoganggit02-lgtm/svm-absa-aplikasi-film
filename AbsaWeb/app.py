import streamlit as st
import pandas as pd
import joblib
import time
import json
import matplotlib.pyplot as plt
import altair as alt 
from wordcloud import WordCloud
from streamlit_option_menu import option_menu
from google_play_scraper import search, Sort, reviews

# Import helper functions
from preprocessing import preprocess_text, split_by_conjunction

# ================= PAGE CONFIGURATION =================
st.set_page_config(
    page_title="Sistem Analisis Sentimen SVM",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= CUSTOM CSS =================
st.markdown("""
    <style>
        .stApp { background-color: #ffffff; color: #1f2937; }
        .stTextArea textarea { background-color: #f9fafb !important; color: #111827 !important; border: 1px solid #d1d5db !important; }
        .stTextInput input { background-color: #f9fafb !important; color: #111827 !important; border: 1px solid #d1d5db !important; }
        .main-header { font-size: 1.6rem; font-weight: 700; color: #111827; margin-bottom: 0.5rem; margin-top: 1rem; line-height: 1.4; text-align: center; }
        .sub-header { font-size: 1rem; color: #6b7280; margin-bottom: 2rem; text-align: center; }
        div[data-testid="stVerticalBlockBorderWrapper"] { border: 1px solid #e5e7eb; border-radius: 10px; padding: 20px; background-color: #ffffff; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); }
        section[data-testid="stSidebar"] { background-color: #f8fafc; border-right: 1px solid #e2e8f0; }
        div[data-testid="stMetricValue"] { font-size: 1.6rem !important; font-weight: 700; color: #111827; }
        div[data-testid="stMetricLabel"] { font-size: 0.85rem !important; color: #6b7280; }
        .block-container { padding-top: 4rem !important; padding-bottom: 2rem; }
        /* Tombol Merah */
        div[data-testid="column"] button:contains("Tolak") { background-color: #ef4444; color: white; border: 1px solid #dc2626; }
    </style>
""", unsafe_allow_html=True)

# ================= LOAD RESOURCES =================
@st.cache_resource
def load_resources():
    try:
        df_mentah   = pd.read_csv('datasetmentah.csv')
        df_bersih   = pd.read_csv('dataset_bersih.csv')
        df_sentimen = pd.read_csv('dataset_sentimen.csv')
        df_aspek    = pd.read_csv('dataset_aspek.csv')
        model_sent = joblib.load('model/svm_sentimen.pkl')
        tfidf_sent = joblib.load('model/tfidf_sentimen.pkl')
        model_aspek = joblib.load('model/svm_aspek.pkl')
        tfidf_aspek = joblib.load('model/tfidf_aspek.pkl')
        with open('model_metrics.json') as f:
            metrics = json.load(f)
        return df_mentah, df_bersih, df_sentimen, df_aspek, model_sent, tfidf_sent, model_aspek, tfidf_aspek, metrics
    except Exception as e:
        st.error(f"Gagal memuat resource: {e}")
        return None, None, None, None, None, None, None, None, None

resources = load_resources()
if resources[0] is None: st.stop()
df_mentah, df_bersih, df_sentimen, df_aspek, m_sent, t_sent, m_aspek, t_aspek, m_data = resources

# ================= SIDEBAR =================
with st.sidebar:
    st.markdown("### Navigasi Sistem")
    menu = option_menu(
        menu_title=None,
        options=["Ringkasan Dashboard", "Analisis Langsung", "Kontribusi Kata", "Kelola Kamus (Admin)"], 
        icons=["grid", "robot", "people", "shield-lock"], 
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#4b5563", "font-size": "14px"}, 
            "nav-link": {"font-size": "14px", "text-align": "left", "margin":"0px", "color": "#374151", "--hover-color": "#f3f4f6"},
            "nav-link-selected": {"background-color": "#e5e7eb", "color": "#111827", "font-weight": "600"},
        }
    )
    st.divider()
    if st.button("Reload Sistem & Kamus", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()
    st.divider()
    st.caption("Versi Sistem 1.0.0")

# ================= HELPER =================
def ambil_metric(data):
    if "accuracy" in data: return data["accuracy"]*100, data["precision"]*100, data["f1_score"]*100
    elif "acc" in data: return data.get("acc"), data.get("prec"), data.get("f1")
    return 0, 0, 0

# ================= DASHBOARD =================
if menu == "Ringkasan Dashboard":
    st.markdown('<div class="main-header">IMPLEMENTASI ALGORITMA SUPPORT VECTOR MACHINE (SVM) UNTUK ANALISIS SENTIMEN KOMPARATIF BERBASIS ASPEK PADA ULASAN PENGGUNA APLIKASI STREAMING FILM</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Dashboard Ringkasan Distribusi Dataset Modeling</div>', unsafe_allow_html=True)
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        jml_mentah, jml_bersih, jml_sentimen, jml_aspek = len(df_mentah), len(df_bersih), len(df_sentimen), len(df_aspek)
        c1.metric("Data Mentah", f"{jml_mentah:,}"); c2.metric("Data Bersih", f"{jml_bersih:,}", delta=f"{jml_bersih - jml_mentah}")
        c3.metric("Data Berlabel", f"{jml_sentimen:,}", delta=f"{jml_sentimen - jml_bersih}"); c4.metric("Total Aspek", f"{jml_aspek:,}", delta=f"{jml_aspek - jml_sentimen}", delta_color="off")
    col_left, col_right = st.columns([1, 1])
    with col_left:
        with st.container(border=True):
            st.subheader("Distribusi Data")
            t1, t2 = st.tabs(["Sentimen", "Aspek"])
            with t1: st.bar_chart(df_sentimen["Sentimen"].value_counts(), color="#2563eb")
            with t2: st.bar_chart(df_aspek["Aspek"].value_counts(), color="#2563eb")
    with col_right:
        with st.container(border=True):
            st.subheader("Evaluasi Model")
            acc_s, prec_s, f1_s = ambil_metric(m_data.get("sentimen", {}))
            acc_a, prec_a, f1_a = ambil_metric(m_data.get("aspek", {}))
            st.caption("SENTIMEN"); m1, m2, m3 = st.columns(3)
            m1.metric("Akurasi", f"{acc_s:.1f}%"); m2.metric("Presisi", f"{prec_s:.1f}%"); m3.metric("F1 Score", f"{f1_s:.1f}%")
            st.divider(); st.caption("ASPEK"); m4, m5, m6 = st.columns(3)
            m4.metric("Akurasi", f"{acc_a:.1f}%"); m5.metric("Presisi", f"{prec_a:.1f}%"); m6.metric("F1 Score", f"{f1_a:.1f}%")

# ================= ANALISIS LANGSUNG =================
elif menu == "Analisis Langsung":
    st.markdown('<div class="main-header">Analisis Real-time</div>', unsafe_allow_html=True)
    tab_manual, tab_scrape = st.tabs(["Input Manual", "Scraper Streaming Film"])
    with tab_manual:
        with st.container(border=True):
            col_in, col_res = st.columns([1, 1])
            with col_in:
                st.subheader("Input ulasan")
                teks_input = st.text_area("Masukkan ulasan...", height=150, value="film lnkp tp hrg mahal")
                analyze_btn = st.button("Analisis Teks", type="primary", use_container_width=True)
                show_debug = st.checkbox("Tampilkan Proses Cleaning (Debug)")
            with col_res:
                st.subheader("Hasil Analisis")
                if analyze_btn and teks_input:
                    with st.spinner('Memproses...'):
                        time.sleep(0.5) 
                        pecahan = split_by_conjunction(teks_input)
                        for i, p in enumerate(pecahan):
                            clean = preprocess_text(p)
                            if show_debug:
                                with st.expander(f"Debug: '{p}'"): st.text(f"Original: {p}\nCleaned : {clean}")
                            pred_s = m_sent.predict(t_sent.transform([clean]))[0]
                            pred_a = m_aspek.predict(t_aspek.transform([clean]))[0]
                            with st.container(border=True):
                                st.markdown(f"**Frasa {i+1}:** {p}")
                                r1, r2 = st.columns(2)
                                color_s = "#166534" if pred_s == "Positif" else "#991b1b" if pred_s == "Negatif" else "#374151"
                                r1.markdown(f"Sentimen: <span style='color:{color_s}; font-weight:bold'>{pred_s.upper()}</span>", unsafe_allow_html=True)
                                r2.markdown(f"Aspek: <span style='font-weight:bold; color:#1f2937'>{pred_a.upper()}</span>", unsafe_allow_html=True)
    with tab_scrape:
        @st.dialog("Cari Aplikasi Streaming") 
        def popup_pencarian():
            keyword = st.text_input("Nama Aplikasi", placeholder="Contoh: Netflix, WeTV").strip()
            jml = st.number_input("Batas Ambil Data", 10, 200, 20)
            st.divider()
            if keyword:
                try: hasil = search(keyword, lang="id", country="id")
                except: hasil = []
                if not hasil: st.warning("Aplikasi tidak ditemukan.")
                else:
                    streaming_keywords = ['streaming', 'movie', 'film', 'drama', 'tv', 'video', 'nonton', 'cinema', 'wetv', 'netflix', 'viu', 'iqiyi', 'vidio', 'disney', 'prime']
                    found_streaming_app = False
                    for app in hasil[:10]:
                        real_id = app['appId']
                        title_lower = app['title'].lower()
                        is_streaming = any(k in title_lower or k in app.get('summary','').lower() for k in streaming_keywords)
                        if not is_streaming: continue
                        found_streaming_app = True
                        if (real_id is None) and ("WeTV" in app['title']): real_id = "com.tencent.qqlivei18n"
                        if real_id is None: continue 
                        with st.container(border=True):
                            c1, c2, c3 = st.columns([1, 4, 2], vertical_alignment="center")
                            with c1: st.image(app['icon'], width=50)
                            with c2: st.markdown(f"**{app['title']}**"); st.caption(f"ID: {real_id}")
                            with c3:
                                if st.button("Pilih", key=f"btn_{real_id}", use_container_width=True):
                                    st.session_state.update({'trigger_scrape': True, 'target_id': real_id, 'target_title': app['title'], 'target_limit': jml})
                                    st.rerun()
        if st.button("Cari Aplikasi Streaming", type="primary", use_container_width=True): popup_pencarian()
        if st.session_state.get('trigger_scrape'):
            app_id, app_title, limit = st.session_state['target_id'], st.session_state['target_title'], st.session_state['target_limit']
            with st.status(f"Sedang bekerja... **{app_title}**", expanded=True) as status:
                res, _ = reviews(app_id, lang='id', country='id', count=limit, sort=Sort.NEWEST)
                if not res: res, _ = reviews(app_id, lang='en', country='us', count=limit, sort=Sort.NEWEST)
                if res:
                    df = pd.DataFrame(res)[['content', 'score', 'at']].head(limit)
                    df.columns = ['Komentar_Asli', 'Rating', 'Tanggal']
                    df['temp_split'] = df['Komentar_Asli'].apply(split_by_conjunction)
                    df_final = df.explode('temp_split').reset_index(drop=True)
                    df_final['clean'] = df_final['temp_split'].apply(preprocess_text)
                    df_final = df_final[df_final['clean'].str.strip() != ""]
                    df_final['sentimen_pred'] = m_sent.predict(t_sent.transform(df_final['clean']))
                    df_final['aspek_pred'] = m_aspek.predict(t_aspek.transform(df_final['clean']))
                    status.update(label="Selesai!", state="complete", expanded=False); time.sleep(1)
                    st.session_state['hasil_scrape'] = df_final; st.session_state['app_terpilih'] = app_title; st.session_state['trigger_scrape'] = False; st.rerun()
                else: status.update(label="Gagal", state="error"); st.session_state['trigger_scrape'] = False
        if 'hasil_scrape' in st.session_state:
            df = st.session_state['hasil_scrape']
            with st.container(border=True):
                st.subheader(f"Hasil Analisis: {st.session_state.get('app_terpilih', 'App')}")
                t1, t2 = st.tabs(["Visualisasi", "Tabel Data"])
                with t1:
                    df['sentimen_pred'] = df['sentimen_pred'].str.capitalize()
                    st.markdown("##### 📊 Analisis Komparatif: Sentimen per Aspek")
                    chart_df = df.groupby(['aspek_pred', 'sentimen_pred']).size().reset_index(name='Jumlah')
                    chart = alt.Chart(chart_df).mark_bar().encode(x=alt.X('aspek_pred:N', title='Aspek'), y=alt.Y('Jumlah:Q'), color=alt.Color('sentimen_pred:N', scale=alt.Scale(domain=['Positif', 'Negatif'], range=['#22c55e', '#ef4444'])), xOffset='sentimen_pred:N')
                    st.altair_chart(chart, use_container_width=True)
                    st.divider(); g1, g2 = st.columns(2)
                    with g1: st.markdown("###### Total Sentimen"); st.bar_chart(df['sentimen_pred'].value_counts(), color="#2563eb", horizontal=True)
                    with g2: st.markdown("###### Total Review per Aspek"); st.bar_chart(df['aspek_pred'].value_counts(), color="#2563eb", horizontal=True)
                    st.divider(); wc1, wc2 = st.columns(2)
                    with wc1:
                        st.caption("☁️ WordCloud Positif"); txt_pos = " ".join(df[df['sentimen_pred']=='Positif']['clean'])
                        if txt_pos: wc = WordCloud(width=400, height=200, background_color='white', colormap='Greens').generate(txt_pos); fig, ax = plt.subplots(figsize=(5,3)); ax.imshow(wc, interpolation='bilinear'); ax.axis("off"); st.pyplot(fig)
                    with wc2:
                        st.caption("☁️ WordCloud Negatif"); txt_neg = " ".join(df[df['sentimen_pred']=='Negatif']['clean'])
                        if txt_neg: wc = WordCloud(width=400, height=200, background_color='white', colormap='Reds').generate(txt_neg); fig, ax = plt.subplots(figsize=(5,3)); ax.imshow(wc, interpolation='bilinear'); ax.axis("off"); st.pyplot(fig)
                with t2:
                    st.download_button("Download CSV", df.to_csv(index=False).encode('utf-8'), "laporan.csv", "text/csv")
                    st.dataframe(df[['temp_split', 'sentimen_pred', 'aspek_pred', 'clean']], use_container_width=True)
            if st.button("Bersihkan Hasil"): del st.session_state['hasil_scrape']; st.rerun()

# ================= KONTRIBUSI KATA (USER) =================
elif menu == "Kontribusi Kata":
    st.markdown('<div class="main-header">Kontribusi Pengguna</div>', unsafe_allow_html=True)
    st.info("Bantu kami memperluas database. Pilih jenis kontribusi Anda.")
    with st.container(border=True):
        jenis_kata = st.selectbox("Jenis Kontribusi", ["Lapor Typo / Singkatan", "Saran Keyword Aspek Baru"])
        c1, c2 = st.columns(2)
        if jenis_kata == "Lapor Typo / Singkatan":
            input_kiri = c1.text_input("Kata Typo", placeholder="cth: mntp").strip().lower()
            input_kanan = c2.text_input("Kata Sebenarnya", placeholder="cth: mantap").strip().lower()
        elif jenis_kata == "Saran Keyword Aspek Baru":
            input_kiri = c1.text_input("Keyword Baru", placeholder="cth: buffering").strip().lower()
            pilihan_aspek = ["Kualitas Streaming", "Harga & Pembayaran", "Aplikasi & Fitur", "Konten", "Akun & Akses", "Umum"]
            input_kanan = c2.selectbox("Masuk Kategori Aspek Mana?", pilihan_aspek)
        if st.button("Kirim Usulan", type="primary", use_container_width=True):
            if input_kiri:
                try:
                    df_saran = pd.DataFrame([[input_kiri, input_kanan, jenis_kata]], columns=['typo', 'normal', 'jenis'])
                    df_saran.to_csv('kamus/kamus_saran.csv', mode='a', header=False, index=False)
                    st.success(f"Terima kasih! Saran dikirim ke Admin.")
                except:
                    df_saran = pd.DataFrame([[input_kiri, input_kanan, jenis_kata]], columns=['typo', 'normal', 'jenis'])
                    df_saran.to_csv('kamus/kamus_saran.csv', header=False, index=False)
                    st.success("Saran terkirim.")
            else: st.error("Mohon isi data.")

# ================= KELOLA KAMUS (ADMIN - TAB SEPARATED) =================
elif menu == "Kelola Kamus (Admin)":
    st.markdown('<div class="main-header">Manajemen Kamus (Admin)</div>', unsafe_allow_html=True)
    PASSWORD_ADMIN = "admin123" 
    with st.container(border=True):
        st.info("Area Terbatas untuk Administrator.")
        input_pass = st.text_input("Password Admin", type="password")

    if input_pass == PASSWORD_ADMIN:
        st.success("Akses Diterima.")
        st.divider()

        # --- PEMISAHAN TAB SARAN ---
        st.subheader("📥 Inbox Saran Pengguna")
        tab_saran_typo, tab_saran_aspek = st.tabs(["Saran Typo", "Saran Aspek"])

        try:
            df_saran_all = pd.read_csv('kamus/kamus_saran.csv', names=['typo', 'normal', 'jenis'], header=None)
            
            # --- TAB 1: SARAN TYPO ---
            with tab_saran_typo:
                df_typo_only = df_saran_all[df_saran_all['jenis'] == "Lapor Typo / Singkatan"].copy()
                if not df_typo_only.empty:
                    df_typo_only['Pilih'] = False
                    edit_typo = st.data_editor(df_typo_only[['Pilih', 'typo', 'normal']], column_config={"Pilih": st.column_config.CheckboxColumn("Pilih?"), "typo": "Typo", "normal": "Perbaikan"}, hide_index=True, use_container_width=True, key="ed_typo")
                    
                    c1, c2 = st.columns([1, 1])
                    if c1.button("✅ Terima Typo Terpilih", use_container_width=True):
                        diterima = edit_typo[edit_typo['Pilih'] == True]
                        if not diterima.empty:
                            diterima[['typo', 'normal']].to_csv('kamus/kamus_typo.csv', mode='a', header=False, index=False)
                            # Hapus dari file saran utama
                            df_saran_all = df_saran_all[~((df_saran_all['typo'].isin(diterima['typo'])) & (df_saran_all['jenis'] == "Lapor Typo / Singkatan"))]
                            df_saran_all.to_csv('kamus/kamus_saran.csv', index=False, header=False)
                            st.toast("Typo disimpan!"); time.sleep(1); st.cache_resource.clear(); st.rerun()
                    if c2.button("🗑️ Tolak Typo Terpilih", key="del_typo", use_container_width=True):
                        ditolak = edit_typo[edit_typo['Pilih'] == True]
                        if not ditolak.empty:
                            df_saran_all = df_saran_all[~((df_saran_all['typo'].isin(ditolak['typo'])) & (df_saran_all['jenis'] == "Lapor Typo / Singkatan"))]
                            df_saran_all.to_csv('kamus/kamus_saran.csv', index=False, header=False)
                            st.toast("Dihapus."); time.sleep(1); st.rerun()
                else: st.info("Tidak ada saran typo.")

            # --- TAB 2: SARAN ASPEK ---
            with tab_saran_aspek:
                df_aspek_only = df_saran_all[df_saran_all['jenis'] == "Saran Keyword Aspek Baru"].copy()
                if not df_aspek_only.empty:
                    df_aspek_only['Pilih'] = False
                    edit_aspek = st.data_editor(df_aspek_only[['Pilih', 'typo', 'normal']], column_config={"Pilih": st.column_config.CheckboxColumn("Pilih?"), "typo": "Keyword", "normal": "Kategori Aspek"}, hide_index=True, use_container_width=True, key="ed_aspek")
                    
                    c1, c2 = st.columns([1, 1])
                    if c1.button("✅ Terima Aspek Terpilih", use_container_width=True):
                        diterima = edit_aspek[edit_aspek['Pilih'] == True]
                        if not diterima.empty:
                            # Format file Abang: [Aspek, Keyword] -> [normal, typo]
                            diterima[['normal', 'typo']].to_csv('kamus/kamus_aspek.csv', mode='a', header=False, index=False)
                            df_saran_all = df_saran_all[~((df_saran_all['typo'].isin(diterima['typo'])) & (df_saran_all['jenis'] == "Saran Keyword Aspek Baru"))]
                            df_saran_all.to_csv('kamus/kamus_saran.csv', index=False, header=False)
                            st.toast("Aspek disimpan!"); time.sleep(1); st.rerun()
                    if c2.button("🗑️ Tolak Aspek Terpilih", key="del_aspek", use_container_width=True):
                        ditolak = edit_aspek[edit_aspek['Pilih'] == True]
                        if not ditolak.empty:
                            df_saran_all = df_saran_all[~((df_saran_all['typo'].isin(ditolak['typo'])) & (df_saran_all['jenis'] == "Saran Keyword Aspek Baru"))]
                            df_saran_all.to_csv('kamus/kamus_saran.csv', index=False, header=False)
                            st.toast("Dihapus."); time.sleep(1); st.rerun()
                else: st.info("Tidak ada saran aspek.")
        except: st.info("Inbox kosong.")

        st.divider()
        st.subheader("📊 Database Kamus Aktif")
        col_db1, col_db2 = st.columns(2)
        with col_db1:
            with st.expander("Lihat Database Typo"):
                try: st.dataframe(pd.read_csv('kamus/kamus_typo.csv', names=['typo', 'normal'], header=None), use_container_width=True)
                except: st.write("Kosong.")
        with col_db2:
            with st.expander("Lihat Database Aspek (Untuk Colab)"):
                try: 
                    df_db_a = pd.read_csv('kamus/kamus_aspek.csv', names=['Aspek', 'Keyword'], header=None)
                    st.dataframe(df_db_a, use_container_width=True)
                    st.download_button("📥 Download kamus_aspek.csv", df_db_a.to_csv(index=False, header=False).encode('utf-8'), "kamus_aspek.csv", "text/csv")
                except: st.write("Kosong.")

        st.divider()
        c_reset, c_info = st.columns([1, 2])
        with c_reset:
            if st.button("⚠️ Reset Kamus Typo ke Backup", type="secondary"):
                try: pd.read_csv('kamus/kamus_backup.csv', header=None).to_csv('kamus/kamus_typo.csv', index=False, header=False); st.toast("Reset Berhasil!"); time.sleep(1); st.cache_resource.clear(); st.rerun()
                except: st.error("Backup tidak ditemukan.")

    elif input_pass != "": st.error("Password Salah.")