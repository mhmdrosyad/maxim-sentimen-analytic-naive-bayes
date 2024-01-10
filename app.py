from flask import Flask, render_template, \
    request, redirect, url_for, session, flash, get_flashed_messages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import base64

from sqlalchemy import create_engine, text

engine = create_engine('mysql://root:@localhost/db_naive')


application = Flask(__name__)

@application.route('/')
def index():
    return render_template('index.html')

@application.route('/analisis')
def analisis():
    return render_template('analisis.html')


@application.route('/data-set')
def data_set():
    with engine.connect() as conn:
        # Menjalankan query SQL untuk mengambil data dari tabel
        query = text("SELECT * FROM dataset")
        result = conn.execute(query)

        # Mendapatkan semua baris hasil query
        rows = result.fetchall()
    # Mengirim data DataFrame ke template HTML
    return render_template('upload-data.html', data=rows)

@application.route('/delete-all')
def delete_all():
    with engine.connect() as conn:
        # Menjalankan query SQL untuk mengambil data dari tabel
        query = text("DELETE FROM dataset")
        conn.execute(query)
        conn.commit()
    return redirect('/data-set')



@application.route('/upload-data', methods=['POST'])
def upload_data():
    try:
        # Mendapatkan file dari formulir
        file = request.files['file']
        df = pd.read_csv(file)
        df = df.dropna()
        df.to_sql(name='dataset', con=engine, if_exists='append', index=False)
        return redirect('/data-set')
    except Exception as e:
        return f"Error: {str(e)}", 500

@application.route('/proses-naive', methods=['POST'])
def proses_naive():
    try:
        # Mendapatkan file dari formulir
        file = request.files['file']

        # Membaca file CSV
        df_upload = pd.read_csv(file)
        
        df_upload = df_upload.dropna()

         # Memeriksa kolom yang diperlukan pada file upload
        if 'content' not in df_upload.columns or 'Label' not in df_upload.columns:
            return "File CSV harus memiliki kolom 'content' dan 'Label'", 400
        
                # Menghitung frekuensi kata di kategori positif dan negatif
        df_upload['Label'] = df_upload['Label'].str.lower()

        pos_words = ' '.join(df_upload[df_upload['Label'] == 'positif']['content'].str.lower())
        neg_words = ' '.join(df_upload[df_upload['Label'] == 'negatif']['content'].str.lower())

        word_freq_pos = df_upload[df_upload['Label'] == 'positif']['content'].str.lower().str.split().explode().value_counts()
        word_freq_neg = df_upload[df_upload['Label'] == 'negatif']['content'].str.lower().str.split().explode().value_counts()

        # Mengonversi ke format JSON untuk dikirim ke template
        
        word_freq_pos_dict = word_freq_pos.head(10).reset_index().to_dict(orient='records')
        word_freq_neg_dict = word_freq_neg.head(10).reset_index().to_dict(orient='records')



        if not pos_words or not neg_words:
            return "Tidak ada kata-kata untuk membuat WordCloud", 400

        # Membuat WordCloud untuk kategori positif
        wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(pos_words)

        # Membuat WordCloud untuk kategori negatif
        wordcloud_neg = WordCloud(width=800, height=400, background_color='white').generate(neg_words)

        # Simpan gambar WordCloud ke dalam BytesIO
        img_pos = BytesIO()
        wordcloud_pos.to_image().save(img_pos, format='PNG')
        img_pos_b64 = base64.b64encode(img_pos.getvalue()).decode('utf-8')

        img_neg = BytesIO()
        wordcloud_neg.to_image().save(img_neg, format='PNG')
        img_neg_b64 = base64.b64encode(img_neg.getvalue()).decode('utf-8')


        # Memisahkan data uji dari file upload menjadi fitur dan label
        X_test = df_upload['content']
        y_test = df_upload['Label']

        # Melakukan tokenisasi dan pembobotan TF-IDF pada data uji
        vectorizer = TfidfVectorizer()
        X_test_vectorized = vectorizer.fit_transform(X_test)

        # Membuat dan melatih model Naive Bayes hanya dengan data uji
        clf = MultinomialNB()
        clf.fit(X_test_vectorized, y_test)

        # Melakukan prediksi pada data uji
        y_pred = clf.predict(X_test_vectorized)

        # Evaluasi model
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)


        test_results = list(zip(X_test, y_pred, y_test))

        count_pos = df_upload[df_upload['Label'] == 'positif'].shape[0]
        count_neg = df_upload[df_upload['Label'] == 'negatif'].shape[0]

        # Simpan jumlah sentimen ke dalam format JSON
        sentiment_count_json = {'positif': count_pos, 'negatif': count_neg}
        accuracy_percentage = round(accuracy * 100, 1)

        return render_template('result.html', accuracy=accuracy_percentage, classification_report=report, test_results=test_results,
                           img_pos=img_pos_b64, img_neg=img_neg_b64, word_freq_pos=word_freq_pos_dict, word_freq_neg=word_freq_neg_dict, sentiment_count=sentiment_count_json, count_pos=count_pos,count_neg=count_neg)

    except Exception as e:
        return f"Error: {str(e)}", 500
    
def get_top_words_per_label(model, feature_names, label, top_n):
    # Mendapatkan probabilitas log dari model untuk setiap fitur
    log_prob = model.feature_log_prob_[label]

    # Membuat kamus dengan fitur dan probabilitasnya
    word_prob_dict = dict(zip(feature_names, log_prob))

    # Mengurutkan kamus berdasarkan nilai probabilitas dan mengambil kata-kata teratas
    top_words = sorted(word_prob_dict, key=word_prob_dict.get, reverse=True)[:top_n]

    return top_words

def create_bar_chart_url(labels, values, title):
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, color='skyblue')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    
    # Simpan gambar ke BytesIO
    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    plt.close()

    # Konversi gambar ke base64
    image_base64 = base64.b64encode(image_stream.getvalue()).decode('utf-8')

    # Buat URL gambar
    image_url = f'data:image/png;base64,{image_base64}'

    return image_url

if __name__ == '__main__':
    application.run(debug=True)