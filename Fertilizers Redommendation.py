import requests
from keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for
from flask import flash
import os
from werkzeug.utils import secure_filename
from tensorflow.python.keras.backend import set_session

app = Flask(__name__)
app.secret_key = "abc"

model = load_model("vegetable.h5")
model1 = load_model("fruit.h5")
@app.route("/hello")
def hello_world():
   return "hello world"

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/prediction', methods = ["GET", "POST"])
def prediction():
    #flash("Showing Prediction Results " + request.method);
    #return render_template('predict.html')
    dat =  str(request.get_data())
    dat1 = dat.split("&")[0]
    pos = dat1.index("=")
    filename=dat1[pos+1:]
    #return filename
    #return request.method + " ----TEST----"
    #if request.method == "POST" :
    #f = request.files['image']
    #basepath = os.path.dirname(__file__)
    #file_path = os.path.join(
    #    basepath, 'uploads', secure_filename(f.filename))
    #f.save(file_path)
    print(filename)
    img = image.image_utils.load_img(filename, target_size=(64,64))
    print(img)
    
    x = image.image_utils.img_to_array(img)
    print(x)
    x = np.expand_dims(x, axis=0)
    print(x)
    pred = model.predict(x)
    print(np.argmax(pred, axis=1))
    class_names = ["Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight", "Potato___healthy", "Potato___Late_blight", "Tomato___Bacterial_spot", "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot"]
    print(class_names[int(np.argmax(pred, axis=1))])
    flash(class_names[int(np.argmax(pred, axis=1))]);
    return render_template('predict.html', prediction=class_names[int(np.argmax(pred, axis=1))]);
    #return class_names[int(np.argmax(pred, axis=1))]
    # plant = request.form['plant']
    # print(plant)
    # if (plant == "vegetable"):
    #     preds = model.predict(x)
    #     print(preds)
    #     df = pd.read_excel('precautions - veg.xlsx')
    #     print(df.iloc[preds[0]]['caution'])
    # else:
    #     preds = model.predict(x)
    #     print(preds)
    #     df = pd.read_excel('precautions - fruits.xlsx')
    #     print(df.iloc[preds[0]]['caution'])
    # return df.iloc[preds[0]]['caution']
        
if __name__ == "__main__" :
    app.run(debug=False)
