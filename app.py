import numpy as np
import keras
# from keras.models import Sequential
# from keras.layers import Dense,Conv2D,MaxPool2D,Dropout,BatchNormalization,Flatten,Activation
# from keras.preprocessing import image 
# from keras.preprocessing.image import ImageDataGenerator
import datetime 
from datetime import date
import pickle
from flask_mysqldb import MySQL, MySQLdb
from flask import Flask, jsonify, make_response,request,flash,redirect,render_template, session,url_for
from itsdangerous import json
from werkzeug.utils import secure_filename
import os
#from flask_cors import CORS
#from flask_restful import Resource, Api
#import pymongo
import re
from flask_ngrok import run_with_ngrok
#import pyngrok
from PIL import Image
#import requests
import datetime
import random
import string
import librosa
from sklearn.preprocessing import LabelEncoder

#-----------Konfigurasi------------

app = Flask(__name__)
app.secret_key = "xxxxx"

#run_with_ngrok(app)
#Konfigurasi folder menyiman upload
UPLOAD_FOLDER_IMG = 'foto_burung'
UPLOAD_FOLDER_AU = 'audio_burung'
ALLOWED_EXTENSIONS_IMG = set(['png', 'jpg', 'jpeg'])
ALLOWED_EXTENSIONS_AU = set(['mp3','wav','ogg','m4a'])
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER_IMG'] = UPLOAD_FOLDER_IMG
app.config['UPLOAD_FOLDER_AU'] = UPLOAD_FOLDER_AU

#konfigurasi database
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'mibird'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
mysql = MySQL(app)

# conn = pymongo.MongoClient(MONGO_ADDR)
# db = conn[MONGO_DB]

#api = Api(app)


from tensorflow.keras.models import load_model
MODEL_IMG_PATH = 'model_img.h5'
MODEL_AU_PATH = 'model_au.h5'
model_img = load_model(MODEL_IMG_PATH,compile=False)
model_au = load_model(MODEL_AU_PATH,compile=False)

pickle_img = open('class_img.pkl','rb')
pickle_au = open('extracted_au.pkl','rb')
num_classes_img = pickle.load(pickle_img)
extr_au = pickle.load(pickle_au)


def allowed_file_img(filename):     
  return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_IMG
def allowed_file_au(filename):     
  return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS_AU
  
@app.route('/api/predict', methods=['POST'])
def predict():
    
    # if 'image' not in request.files:
    #   flash('No file part')
    #   return jsonify({
    #         "pesan":"tidak ada form image"
    #       })
    file = request.files['file']
    if file.filename == '':
      return jsonify({
            "pesan":"tidak ada file image yang dipilih"
          })
    if file and allowed_file_img(file.filename):
      
        letters = string.ascii_lowercase
        result_str = ''.join(random.choice(letters) for i in range(5))
        
        filename = secure_filename(file.filename+result_str+".jpg")
        print(filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER_IMG'], filename))
        path=("foto_burung/"+filename)
    
        
        today = date.today()
        cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cur_event = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cur.execute("INSERT INTO riwayat (nama_file,path, prediksi,akurasi,tanggal) VALUES (%s,%s,%s,%s,%s)", (filename,path,'No predict',int(0), today.strftime("%d/%m/%Y")))
        mysql.connection.commit()

        img=keras.utils.load_img(path,target_size=(224,224))
        img1=keras.utils.img_to_array(img)
        img1=img1/255
        img1=np.expand_dims(img1,[0])
        predict=model_img.predict(img1)
        classes=np.argmax(predict,axis=1)
        
        for key,values in num_classes_img.items():
            if classes==values:
                accuracy = float(round(np.max(model_img.predict(img1))*100,2))
                cur.execute("SELECT * FROM data_burung WHERE nama_burung=%s",(str(key),))
                cur_event.execute("SELECT * FROM event_burung WHERE burung=%s OR burung1=%s OR burung2=%s",(str(key),str(key),str(key),))
                result = cur.fetchone()
                result_event=cur_event.fetchone()
                print(result_event)
                cur.execute("""
                    UPDATE riwayat
                    SET prediksi = %s,
                        akurasi = %s
                    WHERE nama_file = %s
                """, (str(key),accuracy,filename))
                mysql.connection.commit()
            
                print("The predicted image of the bird is: "+str(key)+" with a probability of "+str(accuracy)+"%")            
                print(result)
                if result_event==None:

                    return jsonify({
                    "Nama_Burung":str(key),
                    "Accuracy":str(accuracy)+"%",
                    "Spesies" : result['spesies'],
                    "Makanan" : result['makanan'],
                    "Status" :  result['status'],
                    "Event" : "Belum ada Event",
                    "Alamat" : "",
                    "Tempat" : "",
                    "Tanggal" : "",
                    "Poster" : "",
                    "Url Event" : "" 
                    }) 
                else :
                    return jsonify({
                    "Nama_Burung":str(key),
                    "Accuracy":str(accuracy)+"%",
                    "Spesies" : result['spesies'],
                    "Makanan" : result['makanan'],
                    "Status" :  result['status'],
                    "Event" : result_event['nama_event'],
                    "Alamat" : result_event['alamat'],
                    "Tempat" : result_event['tempat'],
                    "Tanggal" : result_event['tanggal'],
                    "Poster" : result_event['url_img'],
                    "Url Event" : result_event['url_event']
                })       
    elif file and allowed_file_au(file.filename):
      
        letters = string.ascii_lowercase
        result_str = ''.join(random.choice(letters) for i in range(5))
        
        filename = secure_filename(file.filename+result_str+".mp3")
        print(filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER_AU'], filename))
        path=("audio_burung/"+filename)
    
        
        today = date.today()
        cur_event = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cur.execute("INSERT INTO riwayat (nama_file,path, prediksi,akurasi,tanggal) VALUES (%s,%s,%s,%s,%s)", (filename,path,'No predict',int(0), today.strftime("%d/%m/%Y")))
        mysql.connection.commit()

        audio_data, sample_rate = librosa.load(path, res_type="kaiser_fast")
        # get the feature
        feature = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=128)
        # scale the features
        feature_scaled = np.mean(feature.T, axis=0)
        # array of features
        prediction_feature = np.array([feature_scaled])
        # expand dims
        final_prediction_feature = np.expand_dims(prediction_feature, axis=2)
        # get the id of label using argmax
        predicted_vector = np.argmax(model_au.predict(final_prediction_feature), axis=-1)
        pred_acc = model_au.predict(final_prediction_feature)
        # get the class label from class id
        le=LabelEncoder()
        # name_class'])
        from tensorflow.keras.utils import to_categorical
        to_categorical(le.fit_transform(extr_au['Name_class']))
        predicted_class = le.inverse_transform(predicted_vector)
        # display the result
        print("CNN1D has predicted the class as  --> ", predicted_class[0],str(np.max(pred_acc)*100))            

        cur.execute("SELECT * FROM data_burung WHERE nama_burung=%s",(str(predicted_class[0]),))
        cur_event.execute("SELECT * FROM event_burung WHERE burung=%s OR burung1=%s OR burung2=%s",(str(predicted_class[0]),str(predicted_class[0]),str(predicted_class[0]),))
        result = cur.fetchone()
        result_event=cur_event.fetchone()
        print(predicted_class[0])
        
        cur.execute("""
            UPDATE riwayat
            SET prediksi = %s,
                akurasi = %s
                WHERE nama_file = %s
            """, (str(predicted_class[0]),str(round(np.max(pred_acc)*100,2)),filename))
        mysql.connection.commit()
                    
        return jsonify({
            "Nama_Burung":str(predicted_class[0]),
            "Accuracy":str(round(np.max(pred_acc)*100,2))+"%",
            #"Nama_Ilmiah": info['nama_ilmiah'],
            "Spesies" : result['spesies'],
            "Makanan" : result['makanan'],
            "Status" :  result['status'],
            "Event" : result_event['nama_event'],
            "Alamat" : result_event['alamat'],
            "Tempat" : result_event['tempat'],
            "Poster" : result_event['url_img'],
            "Url Event" : result_event['url_event'] 
        })
    else:
      return jsonify({
        "Message":"bukan file yang didukung"
      })

@app.route('/api/event', methods=['GET'])
def api_event():
    
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)    
    cur.execute("SELECT * FROM event_burung")
    result = cur.fetchall()
    return make_response(jsonify({"event":[dict(row) for row in result]}), 200)

@app.route('/admin')
def admin():
    return render_template("login.html")
@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        curl = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        curl.execute("SELECT * FROM admin WHERE username=%s", (username,) )
        user = curl.fetchone()
        curl.close()
        print(user)

        if user is not None and len(user) > 0:
            if password == user['password']:
                
                session['username'] = user['username']
                
                return redirect(url_for('dataBurung'))
            else:
                flash("Gagal, username dan password tidak cocok")
                return redirect(url_for('login'))
        else:
            #flash("Gagal, user tidak ditemukan")
            return redirect(url_for('login'))
    else:
        return render_template('login.html')
    
    return render_template('dashboard.html')

@app.route('/dataBurung')
def dataBurung():
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute('SELECT * FROM data_burung')
    data = cur.fetchall()
    cur.close()
    print(data)
    return render_template('dataBurung.html',dataBurung  = data)


@app.route('/tambahData')
def tambahData():

    return render_template('tambahData.html')

@app.route('/daftarBurung', methods=["POST"])
def daftarBurung():
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    if request.method == "POST":
        nm_burung = request.form['nm_burung']
        #nm_ilm = request.form['nm_ilmiah']
        spesies = request.form['spesies']
        makanan = request.form['makanan']
        status = request.form['status']
        if not re.match(r'[A-Za-z]+', nm_burung):
            flash("Nama harus pakai huruf Dong!")
        
        else:
            cur.execute("INSERT INTO data_burung (nama_burung,spesies,makanan,status) VALUES (%s,%s,%s,%s)", (nm_burung,spesies, makanan, status))
            mysql.connection.commit()
            flash('Data Burung berhasil ditambah')
            return redirect(url_for('dataBurung'))

    return render_template("tambahData.html")

@app.route('/editBurung/<nama_burung>', methods = ['POST', 'GET'])
def editBurung(nama_burung):
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    
    cur.execute('SELECT * FROM data_burung WHERE nama_burung = %s', [nama_burung])
    data = cur.fetchone()
    cur.close()
    print(data)
    return render_template('editBurung.html', editBurung = data)

@app.route('/updateBurung/<nama_burung>', methods=['POST'])
def updatBurung(nama_burung):
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    if request.method == 'POST':
        print(nama_burung)
        #nm_ilm = request.form['nm_ilmiah']
        spesies = request.form['spesies']
        makanan = request.form['makanan']
        status = request.form['status']
        if not re.match(r'[A-Za-z]+', nama_burung):
            flash("Nama harus pakai huruf Dong!")
        else:
          cur.execute("""
              UPDATE data_burung
              SET spesies = %s,
                  makanan = %s,
                  status = %s
              WHERE nama_burung = %s
            """, (spesies, makanan, status, nama_burung))
          flash('Data Burung berhasil diupdate')
          cur.close()
          return render_template("popUpEdit.html")

    return render_template("dataBurung.html")

@app.route('/riwayat')
def riwayat():
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute("SELECT * FROM riwayat")
    dataRiwayat = cur.fetchall()
    cur.close()
    print(dataRiwayat)
    return render_template('riwayat.html',riwayat  = dataRiwayat)
    
@app.route('/hapusRiwayat/<nama_file>', methods = ['POST','GET'])
def hapusRiwayat(nama_file):
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
  
    cur.execute('DELETE FROM riwayat WHERE nama_file =%s',(nama_file))
    mysql.connection.commit() 
    flash(' Berhasil Dihapus!')
    return redirect(url_for('riwayat'))

@app.route('/event')
def event():
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cur.execute("SELECT * FROM event_burung")
    dataEvent = cur.fetchall()
    cur.close()
    print(dataEvent)
    return render_template('event.html',event  = dataEvent)

@app.route('/tambahEvent')
def tambahEvent():

    return render_template('tambahEvent.html')

@app.route('/daftarEvent', methods=["POST"])
def daftarEvent():
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    if request.method == "POST":
        nm_event = request.form['nama_event']
        alamat = request.form['alamat']
        tempat = request.form['tempat']
        tanggal = request.form['tanggal']
        poster = request.form['url_img']
        url_event = request.form['url_event']
        if not re.match(r'[A-Za-z]+', nm_event):
            flash("Nama harus pakai huruf Dong!")
        
        else:
            cur.execute("INSERT INTO event_burung (nama_event,alamat,tempat,tanggal,url_img,url_event) VALUES (%s,%s,%s,%s,%s,%s)", (nm_event,alamat, tempat, tanggal, poster, url_event))
            mysql.connection.commit()
            flash('Data Event berhasil ditambah')
            cur.close()
            return redirect(url_for('event'))

    return render_template("tambahEvent.html")

@app.route('/editEvent/<nama_event>', methods = ['POST', 'GET'])
def editEvent(nama_event):
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    
    cur.execute('SELECT * FROM event_burung WHERE nama_event = %s', [nama_event])
    data = cur.fetchone()
    cur.close()
    print(data)
    return render_template('editEvent.html', editEvent = data)

@app.route('/updateEvent/<nama_event>', methods=['POST'])
def updateEvent(nama_event):
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    if request.method == 'POST':
        print(nama_event)
        nama_event = request.form['nama_event']
        alamat = request.form['alamat']
        tempat = request.form['tempat']
        tanggal = request.form['tanggal']
        url_img = request.form['url_img']
        url_event = request.form['url_event']
        if not re.match(r'[A-Za-z]+', nama_event):
            flash("Nama harus pakai huruf Dong!")
        else:
          cur.execute("""
              UPDATE event_burung
              SET nama_event = %s,
              alamat= %s,
              tempat= %s,
              tanggal= %s,
              url_img= %s,
              url_event= %s
              WHERE nama_event = %s
            """, [nama_event,alamat, tempat, tanggal, url_img, url_event,nama_event])
          flash('Data Event berhasil diupdate')
          return render_template("popUpEditEvent.html")

    return render_template("event.html")

@app.route('/hapusEvent/<nama_event>', methods = ['POST','GET'])
def hapusEvent(nama_event):
    print(nama_event)
    cur = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
  
    cur.execute('DELETE FROM event_burung WHERE nama_event =%s',[nama_event])
    mysql.connection.commit() 
    flash(' Berhasil Dihapus!')
    return redirect(url_for('event'))


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':

  app.run(debug=True, host="0.0.0.0",)
