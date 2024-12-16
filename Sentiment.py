import streamlit as st
import pandas as pd
import re
import string
import nltk
import os

# Tentukan lokasi direktori untuk NLTK di Streamlit
nltk_data_dir = '/home/appuser/nltk_data'  # Pastikan ini sesuai dengan lingkungan Streamlit
nltk.data.path.append(nltk_data_dir)

# Fungsi untuk mengunduh sumber daya jika belum ada
def download_if_not_exists(package):
    try:
        nltk.data.find(f'tokenizers/{package}')
        print(f"'{package}' sudah ada.")
    except LookupError:
        print(f"'{package}' tidak ditemukan. Mengunduh...")
        nltk.download(package, download_dir=nltk_data_dir)

# Pastikan sumber daya yang diperlukan ada
download_if_not_exists('punkt')
download_if_not_exists('vader_lexicon')

# Inisialisasi analisis sentimen VADER
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

# Fungsi preprocessing
def preprocess_text(kalimat):
    lower_case = kalimat.lower()
    hasil = re.sub(r'\d+', '', lower_case)  # Hapus angka
    hasil = hasil.translate(str.maketrans('', '', string.punctuation))  # Hapus tanda baca
    hasil = hasil.strip()  # Hapus spasi di awal dan akhir
    return hasil

# Fungsi tokenisasi
def tokenize_text(kalimat):
    try:
        return nltk.tokenize.word_tokenize(kalimat)
    except LookupError:
        st.error("Sumber daya NLTK 'punkt' tidak ditemukan. Silakan periksa pengaturan sumber daya.")
        return []

# Streamlit App
st.title("Analisis Sentimen dengan ANN, SMOTE, dan Visualisasi Sentimen")

# Upload dataset
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
if uploaded_file is not None:
    # Membaca dataset
    df = pd.read_csv(uploaded_file)
    original_filename = uploaded_file.name  # Ambil nama file asli
    st.subheader("Dataset Lengkap")
    st.write("Berikut adalah semua data dari dataset yang diunggah: ")
    st.dataframe(df)

    # Dropdown untuk memilih kolom yang akan dianalisis
    selected_column = st.selectbox("Pilih kolom untuk analisis sentimen:", df.columns)

    # Validasi kolom yang dipilih
    if selected_column:
        # Hapus baris dengan nilai kosong di kolom yang dipilih
        df = df.dropna(subset=[selected_column])
        st.write(f"Dataset setelah menghapus data kosong di kolom '{selected_column}': {df.shape[0]} baris")

        # Validasi dataset setelah pembersihan
        if df.empty:
            st.error("Dataset kosong setelah menghapus data yang tidak valid. Harap unggah dataset yang valid.")
        else:
            # Tambahkan tombol untuk memulai analisis
            if st.button("Mulai Analisis"):
                # Proses analisis hanya pada kolom yang dipilih
                df['processed_text'] = df[selected_column].apply(preprocess_text)
                df['tokens'] = df['processed_text'].apply(tokenize_text)
                # Proses pembersihan token, stemming, dan penambahan label sentimen
                from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
                from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
                factory = StopWordRemoverFactory()
                stopwords = factory.get_stop_words()
                df['tokens_cleaned'] = df['tokens'].apply(lambda tokens: [token for token in tokens if token not in stopwords])
                stem_factory = StemmerFactory()
                stemmer = stem_factory.create_stemmer()
                df['tokens_stemmed'] = df['tokens_cleaned'].apply(lambda tokens: [stemmer.stem(token) for token in tokens])
                df['processed_text'] = df['tokens_stemmed'].apply(lambda x: ' '.join(x))
                
                # Analisis sentimen otomatis
                def auto_label_sentiment(text):
                    if isinstance(text, int):  # Periksa apakah teks adalah angka
                        text = str(text)  # Ubah angka menjadi string
                    score = sid.polarity_scores(text)['compound']
                    return 'positive' if score > 0 else 'negative'

                df['label'] = df[selected_column].apply(auto_label_sentiment)

                st.subheader("Dataset Setelah Analisis")
                st.dataframe(df)  # Menampilkan dataset lengkap dengan kolom tambahan

                # Simpan hasil analisis di penyimpanan lokal dengan nama file yang bervariasi
                def save_to_local(dataframe, original_filename):
                    directory = "C:/Users/Miftahul Huda N/Komunitas Maribelajar Indonesia/CP7 - 04 - Harmoni - Documents/General/DataSet/"  # Pastikan path ini benar
                    if not os.path.exists(directory):  # Jika folder belum ada, buat folder tersebut
                        os.makedirs(directory)
                    base_filename = os.path.splitext(original_filename)[0]
                    filename = f"hasil_analisis_{base_filename}.csv"
                    path = os.path.join(directory, filename)
                    dataframe.to_csv(path, index=False)  # Simpan dataframe ke CSV
                    return path

                path = save_to_local(df, original_filename)
                st.success(f"Data berhasil disimpan di: {path}")

# Tambahkan tombol untuk mengarahkan ke link Power BI
st.markdown("[Klik di sini untuk membuka laporan Power BI](https://app.powerbi.com/groups/me/reports/a7f6761d-41fb-41cc-96cf-3ca2c89db735/c60fca462db0cce0a67b?experience=power-bi) di tab baru.")
st.markdown("Atau, **klik kanan** link tersebut dan pilih 'Open link in new tab' untuk membukanya di tab baru.")
