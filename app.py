from flask import Flask, render_template, \
    request, redirect, url_for, session, flash, get_flashed_messages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from io import StringIO

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

        with engine.connect() as conn:
            # Menjalankan query SQL untuk mengambil data dari tabel
            query = text("SELECT * FROM dataset")
            result = conn.execute(query)
            # Mendapatkan semua baris hasil query
            rows = result.fetchall()
        
        if rows:
            df_query = pd.DataFrame(rows, columns=result.keys())
            # Memeriksa kolom yang diperlukan pada file upload
            if 'content' not in df_upload.columns or 'Label' not in df_upload.columns:
                return "File CSV harus memiliki kolom 'content' dan 'Label'", 400

            # Memeriksa kolom yang diperlukan pada hasil query
            if 'content' not in df_query.columns or 'label' not in df_query.columns:
                return "Query SQL harus mengembalikan kolom 'content' dan 'Label'", 500

            # Memisahkan data latihan dari hasil query menjadi fitur dan label
            X_train = df_query['content']
            y_train = df_query['label']

            # Memisahkan data uji dari file upload menjadi fitur dan label
            X_test = df_upload['content']
            y_test = df_upload['Label']

            # Melakukan tokenisasi dan pembobotan TF-IDF pada data latihan
            vectorizer = TfidfVectorizer()
            X_train_vectorized = vectorizer.fit_transform(X_train)

            # Membuat dan melatih model Naive Bayes
            clf = MultinomialNB()
            clf.fit(X_train_vectorized, y_train)

            # Melakukan tokenisasi dan pembobotan TF-IDF pada data uji
            X_test_vectorized = vectorizer.transform(X_test)

            # Melakukan prediksi pada data uji
            y_pred = clf.predict(X_test_vectorized)

            # Evaluasi model
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)

            test_results = list(zip(X_test, y_pred, y_test))

            return render_template('result.html', accuracy=accuracy, classification_report=report, test_results=test_results)
        else:
            return redirect('/analisis')

    except Exception as e:
        return f"Error: {str(e)}", 500


if __name__ == '__main__':
    application.run(debug=True)