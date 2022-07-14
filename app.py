from crypt import methods
from fileinput import filename
from flask import Flask, redirect,render_template,request,flash
!pip install opencv-python
!apt update && apt install -y libsm6 libxext6
!apt-get install -y libxrender-dev
import cv2
import face_recognition
import time
import numpy as np
import keras
import matplotlib.pyplot as plt

upload_folder='static/uploads/image.png'
model=keras.models.load_model("model1.h5")
app=Flask(__name__)
app.secret_key="asdfghkl;"
# app.config['UPLOAD_FOLDER']=upload_folder
@app.route("/home")
def index():
    return render_template("index.html",song_link="",show=False)

@app.route("/predict",methods=["POST","GET"])
def recommend():
    cam=cv2.VideoCapture(0)
    cam.set(3,500)
    cam.set(4,500)
    
    time.sleep(1)
    result,main_img=cam.read()
    if result:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    cam.release()
    cv2.imwrite(upload_folder,main_img)
    img=cv2.cvtColor(main_img,cv2.COLOR_BGR2GRAY)
    face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_alt2.xml')
    faces = face_cascade.detectMultiScale(img, 1.1, 4)
    face_locations=face_recognition.face_locations(img)
    face_imgs=[]
    for (x,y,w,h) in faces:
        img=img[y:y+w,x:x+h]
        img=cv2.cvtColor(img,cv2.COLOR_BayerRG2GRAY)
        img=cv2.resize(img,(48,48),interpolation=cv2.INTER_AREA)
        img=np.array(img)
        img=np.expand_dims(img,axis=0)
        img=np.expand_dims(img,axis=-1)
        img=img.repeat(3,axis=-1)
        img=img/255
        face_imgs.append(img)
        face_imgs=np.array(face_imgs)
        break

    emotion_label={0:"angry",1:"disgusted",2:"fearful",3:"happy",4:"neutral",5:"sad",6:"surprised"}
    link="yt"
    pred=model.predict(face_imgs[0])
    em=emotion_label[pred[0].argmax(axis=0)]
    msg="You seem to be "+em 

    url='https://www.youtube.com/results?search_query='
    lang=str(request.form['lang'])
    singer=str(request.form['singer'])
    age=str(request.form['age'])

    if singer!='':
        singer=singer.replace(' ','+')
        url="".join([url,singer,'+'])
    if lang!='':
        url="".join([url,lang,'+'])
    url="".join([url,em,'+song+'])
    if age!='':
        if age=="0-12":
            url+='for+kids'
        elif age=="40-above":
            url+="of+90's"

    return render_template("index.html",show=True,message=msg,song_link=url)

if __name__=="__main__":
    app.run(debug=True)
