from flask import Flask, render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import pickle
from flask import Flask
from flask import Flask, jsonify, request, render_template
import base64
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)

@app.route('/')
def rumah():
    return render_template('home.html')

@app.route('/heart')
def iris():
    return render_template('heart.html')

@app.route('/klasifikasi', methods=['POST', 'GET'])
def hasil():
    if request.method == 'POST':
        input_form = request.form
        age = float(input_form['age'])
        sex = float(input_form['sex'])
        cp=float(input_form['cp'])
        trestbps=float(input_form['trestbos'])
        chol=float(input_form['chol'])
        fbs=float(input_form['fbs'])
        restecg=float(input_form['restecg'])
        thalach=float(input_form['thalach'])
        exang=float(input_form['exang'])
        oldpeak= float(input_form['oldpeak'])
        slope=float(input_form['slope'])
        ca=float(input_form['ca'])
        thal=float(input_form['thal'])
        # target=float(input_form['target'])

        pred = Model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
        exang, oldpeak, slope, ca, thal]])
        prediksi = ""
        if pred[0] == 0:
            prediksi = 'Tidak Heart Disease'
        else:
            prediksi = 'Heart Disease'
        return render_template('hasil.html', data=input_form, prediksi=prediksi)


@app.route("/datviz")
def home():
    df_import = pd.read_csv('heart.csv')
    df1 = df_import.copy()
    def sex_cat(x):
        if x == 0:
            return 'female'
        else:
            return 'male'
        
    def target_cat(x):
        if x == 0:
            return 'no heart disease' # 0: no
        else:
            return 'heart disease' # 1: yes
        
    def cp_cat(x):
        if x == 0:
            return 'typical '
        elif x == 1:
            return 'asymptotic'
        elif x == 2:
            return 'nonanginal'
        else:
            return 'nontypical'
    
    def chol_cat(x):
        if x <= 200:
            return '1_normal'
        elif 201 <= x <= 239:
            return '2_high'
        else:
            return '3_very high'
        
        
    df1['sex_cat'] = df1['sex'].apply(sex_cat)
    df1['target_cat'] = df1['target'].apply(target_cat)
    df1['cp_cat'] = df1['cp'].apply(cp_cat)
    df1['chol_cat'] = df1['chol'].apply(chol_cat)

    # Age
    fig = plt.figure(figsize=(8,3),dpi=300)
    fig.add_subplot()
    sns.countplot(data= df1, x='sex_cat',hue='target_cat')
    plt.title('Grafik Plot Jumlah Jenis Kelamin VS Target')
    plt.savefig('sex_target.png',bbox_inches="tight") 
    # plt.show()

    # Menentukan Size
    # fig = plt.figure(figsize=(8,3),dpi=300)
    # fig.add_subplot()
    # sns.catplot("mfr", data = df, kind = "count")

    # plt.savefig('mfr.png',bbox_inches="tight") 


    # Mengubah Plot ke dalam base64 agar dapat ditampilkan di HTML
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue())

    # memasukkan kedalam variabel
    result = str(figdata_png)[2:-1]


    ##################################### Beda Plot
    fig.add_subplot()

    target_0 = df1.loc[df1['target'] == 0]
    target_1 = df1.loc[df1['target'] == 1]

    plt.title('Grafik Persebaran Kadar Kolesterol dengan Hue="target"')
    sns.distplot(target_0[['chol']], hist=False, rug=True, label='No Heart Disease')
    sns.distplot(target_1[['chol']], hist=False, rug=True, label='Heart Disease')
    
    # plt.show()
    # sns.catplot("mfr", "rating", data = df)


    plt.savefig('chol_target.png',bbox_inches="tight") 


    # Mengubah Plot ke dalam base64 agar dapat ditampilkan di HTML
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue())

    # memasukkan kedalam variabel
    result2 = str(figdata_png)[2:-1]

    ##################################### Beda Plot
    fig.add_subplot()

    target_0 = df_import.loc[df_import['target'] == 0]
    target_1 = df_import.loc[df_import['target'] == 1]

    plt.title('Grafik Persebaran Usia dengan Hue="target"')
    sns.distplot(target_0[['age']], hist=False, rug=True, label='No Heart Disease')
    sns.distplot(target_1[['age']], hist=False, rug=True, label='Heart Disease')
    # plt.show()


    plt.savefig('age_target.png',bbox_inches="tight") 


    # Mengubah Plot ke dalam base64 agar dapat ditampilkan di HTML
    figfile = BytesIO()
    plt.savefig(figfile, format='png')
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue())

    # memasukkan kedalam variabel
    result3 = str(figdata_png)[2:-1]

    return render_template('plot.html', plot=result, plot2= result2, plot3=result3 )




if __name__ == "__main__":
    with open('heartModel', 'rb') as model:
        Model = pickle.load(model)
    app.run(debug=True)

