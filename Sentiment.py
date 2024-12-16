import streamlit as st
import pandas as pd
import re
import string
import nltk
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os

# Download resources
nltk.download('punkt')
nltk.download('vader_lexicon')

# Inisialisasi analisis sentimen VADER
sid = SentimentIntensityAnalyzer()

# Fungsi preprocessing
def preprocess_text(kalimat):
    lower_case = kalimat.lower()
    hasil = re.sub(r'\d+', '', lower_case)
    hasil = hasil.translate(str.maketrans('', '', string.punctuation))
    hasil = hasil.strip()
    return hasil

def tokenize_text(kalimat):
    return nltk.tokenize.word_tokenize(kalimat)

# Stopword removal
factory = StopWordRemoverFactory()
stopwords = factory.get_stop_words()

def stopwords_text(tokens):
    return [token for token in tokens if token not in stopwords]

# Stemming
stem_factory = StemmerFactory()
stemmer = stem_factory.create_stemmer()

def stemming_text(tokens):
    return [stemmer.stem(token) for token in tokens]

# Labeling otomatis dengan analisis sentimen (hanya positif dan negatif)
def auto_label_sentiment(text):
    if isinstance(text, int):  # Periksa apakah teks adalah angka
        text = str(text)  # Ubah angka menjadi string
    score = sid.polarity_scores(text)['compound']
    return 'positive' if score > 0 else 'negative'

# Fungsi untuk menyimpan data ke penyimpanan lokal
def save_to_local(dataframe, original_filename):
    directory = "C:/Users/Miftahul Huda N/Komunitas Maribelajar Indonesia/CP7 - 04 - Harmoni - Documents/General/DataSet/"  # Pastikan path ini benar
    if not os.path.exists(directory):  # Jika folder belum ada, buat folder tersebut
        os.makedirs(directory)
    # Buat nama file berdasarkan nama file asli dataset yang diunggah
    base_filename = os.path.splitext(original_filename)[0]
    filename = f"hasil_analisis_{base_filename}.csv"
    path = os.path.join(directory, filename)
    dataframe.to_csv(path, index=False)  # Simpan dataframe ke CSV
    return path

# Streamlit App
st.title("Analisis Sentimen dengan ANN, SMOTE, dan Visualisasi Sentimen")

# Upload dataset
uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
if uploaded_file is not None:
    # Membaca dataset
    df = pd.read_csv(uploaded_file)
    original_filename = uploaded_file.name  # Ambil nama file asli
    st.subheader("Dataset Lengkap")
    st.write("Berikut adalah semua data dari dataset yang diunggah:")
    st.dataframe(df)  # Menampilkan semua data

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
                df['tokens_cleaned'] = df['tokens'].apply(stopwords_text)
                df['tokens_stemmed'] = df['tokens_cleaned'].apply(stemming_text)
                df['processed_text'] = df['tokens_stemmed'].apply(lambda x: ' '.join(x))
                df['label'] = df[selected_column].apply(auto_label_sentiment)

                st.subheader("Dataset Setelah Analisis")
                st.dataframe(df)  # Menampilkan dataset lengkap dengan kolom tambahan

                # Pembobotan TF-IDF
                vectorizer = TfidfVectorizer(max_features=5000)
                X = vectorizer.fit_transform(df['processed_text']).toarray()
                le = LabelEncoder()
                y = le.fit_transform(df['label'])

                # Simpan LabelEncoder dalam session_state untuk digunakan nanti
                st.session_state['le'] = le

                # SMOTE untuk data tidak seimbang
                smote = SMOTE(random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X, y)
                st.write(f"Data setelah SMOTE: {X_resampled.shape}")

                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

                # Konversi label ke one-hot encoding
                y_train_one_hot = pd.get_dummies(y_train).values
                y_test_one_hot = pd.get_dummies(y_test).values

                # Bangun dan latih model
                model = Sequential()
                model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
                model.add(Dropout(0.5))
                model.add(Dense(64, activation='relu'))
                model.add(Dropout(0.5))
                model.add(Dense(len(le.classes_), activation='softmax'))  # Softmax untuk multi-class
                model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                st.write("Melatih model ANN...")
                model.fit(X_train, y_train_one_hot, epochs=10, batch_size=32, validation_split=0.2, verbose=0)

                # Evaluasi model
                y_pred = model.predict(X_test).argmax(axis=1)
                acc = accuracy_score(y_test, y_pred)
                st.write(f"Akurasi Model: {acc:.2f}")
                st.text("Classification Report:")
                st.text(classification_report(y_test, y_pred, target_names=le.classes_))

                # Visualisasi Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                st.subheader("Confusion Matrix:")
                fig, ax = plt.subplots(figsize=(5, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

                # Visualisasi Pie Chart Sentimen
                sentiment_counts = pd.Series(y_pred).value_counts(normalize=True)
                sentiment_counts.index = le.inverse_transform(sentiment_counts.index)
                fig, ax = plt.subplots()
                sentiment_counts.plot.pie(autopct='%1.1f%%', labels=sentiment_counts.index, ax=ax)
                ax.set_ylabel("")
                st.pyplot(fig)

                # Simpan hasil analisis di penyimpanan lokal dengan nama file yang bervariasi
                path = save_to_local(df, original_filename)
                st.success(f"Data berhasil disimpan di: {path}")

# Tambahkan tombol untuk mengarahkan ke link Power BI
if st.button("Lihat Laporan di Power BI"):
    st.markdown("[Klik di sini untuk membuka laporan Power BI](https://app.powerbi.com/groups/me/reports/a7f6761d-41fb-41cc-96cf-3ca2c89db735/c60fca462db0cce0a67b?experience=power-bi)")

else:
    st.write("Harap unggah file CSV terlebih dahulu.")
