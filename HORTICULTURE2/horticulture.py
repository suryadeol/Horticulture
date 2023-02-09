import sys
import os
import glob
import re
from numpy import *

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer
from flask import Flask , render_template , request , url_for
import pickle
        
import tensorflow as tf
from tensorflow.keras import models, layers
import math
import matplotlib.pyplot as plt

from matplotlib.image import imread
import cv2
from PIL import Image as im

#chatter bot
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer

#english to telugu
from googletrans import Translator
trans = Translator()

app = Flask('__name__')

crop_yeild = pickle.load(open('crop_yeild.pkl', 'rb'))
fertilizers = pickle.load(open('fertilizers.pkl', 'rb'))
crop_prediction = pickle.load(open('crop_prediction.pkl', 'rb'))
c_w_r = pickle.load(open('c_w_r.pkl', 'rb'))
crop_mixed = pickle.load(open('crop_mixed.pkl', 'rb'))

###############################--- home page --#################################
@app.route('/')
def index():
    return render_template('home.html')
    
@app.route('/server_home')
def server_home():
    return render_template('home.html')

###############################--- About Us --#################################
@app.route('/about_us')
def about_us():
    return render_template('about_us.html')

###############################--- Contact-us --#################################
@app.route('/contact_us')
def contact_us():
    return render_template('contact_us.html')

##################--- crop yeild prediction ---##########################

@app.route('/crop_demand', methods = ['GET','POST'])
def crop_demand():
    return render_template('yeild_prediction.html')

@app.route('/yeild', methods = ['GET', 'POST'])
def yeild():
    if(request.method == "POST"):
        s = request.form['state']
        d = request.form['dis']
        ses =request.form['season']
        c = request.form['crop']
        a = float(request.form['area'])
        s = 0

        district = ['ANANTAPUR', 'CHITTOOR', 'EAST GODAVARI', 'GUNTUR', 'KADAPA', 'KRISHNA', 'KURNOOL', 'PRAKASAM', 'SPSR NELLORE', 'SRIKAKULAM', 'VISAKHAPATANAM', 'VIZIANAGARAM', 'WEST GODAVARI']
        d1=district.index(d.upper())

        season=['Kharif', 'Rabi', 'Whole Year']
        ses1=season.index(ses.title())
        
        crop=['Arecanut', 'Arhar/Tur', 'Bajra', 'Banana', 'Beans & Mutter(Vegetable)', 'Bhindi', 'Bottle Gourd', 'Brinjal', 'Cabbage', 'Cashewnut', 'Castor seed', 'Citrus Fruit', 'Coconut ', 'Coriander', 'Cotton(lint)', 'Cowpea(Lobia)', 'Cucumber', 'Dry chillies', 'Dry ginger', 'Garlic', 'Ginger', 'Gram', 'Grapes', 'Groundnut', 'Horse-gram', 'Jowar', 'Korra', 'Lemon', 'Linseed', 'Maize', 'Mango', 'Masoor', 'Mesta', 'Moong(Green Gram)', 'Niger seed', 'Onion', 'Orange', 'Other  Rabi pulses', 'Other Fresh Fruits', 'Other Kharif pulses', 'Other Vegetables', 'Papaya', 'Peas  (vegetable)', 'Pome Fruit', 'Pome Granet', 'Potato', 'Ragi', 'Rapeseed &Mustard', 'Rice', 'Safflower', 'Samai', 'Sapota', 'Sesamum', 'Small millets', 'Soyabean', 'Sugarcane', 'Sunflower', 'Sweet potato', 'Tapioca', 'Tobacco', 'Tomato', 'Turmeric', 'Urad', 'Varagu', 'Wheat', 'other fibres', 'other misc. pulses', 'other oilseeds']
        c1=crop.index(c.title())

        arr=array([s,d1,ses1,c1,a])
        arr=arr.reshape(1,-1)

        res = crop_yeild.predict(arr)

        name = "The yeild produced by the crop \'"+ str(c)+"\' in tons per hectare \'"+str(res[0]) + "\'"

        return render_template("open.html", n = name)

    else:
        return "Sorry!!"

###########################---- fertilizer type prediction ---#########################

@app.route('/fertilizer_type', methods = ['GET', 'POST'])
def fertilizer_type():
    return render_template('fertilizer.html')

@app.route('/fertilizer', methods=['GET', 'POST'])
def fertilizer():
    if(request.method == 'POST'):
        t = int(request.form['temp'])
        h = int(request.form['humi'])
        m = int(request.form['moist'])
        st = request.form['st']
        ct = request.form['ct']
        n = int(request.form['nitro'])
        k = int(request.form['potash'])
        p = int(request.form['phosp'])

        soil_type = ['Black', 'Clayey', 'Loamy', 'Red', 'Sandy']
        st1 = soil_type.index(st.title())

        crop_type = ['Barley', 'Cotton', 'Ground Nuts', 'Maize', 'Millets', 'Oil seeds', 'Paddy', 'Pulses', 'Sugarcane', 'Tobacco', 'Wheat']
        ct1 = crop_type.index(ct.title())

        arr=array([t,h,m,st1,ct1,n,k,p])
        arr=arr.reshape(1,-1)
        res = fertilizers.predict(arr)

        f_names = ['10-26-26', '14-35-14', '17-17-17', '20-20', '28-28', 'DAP', 'Urea']
        c1 = f_names[math.ceil(res[0])]
        name = "The fertilizer recommended for your feild is \'" +str(c1)+ "\'"

        return render_template("open.html", n = name)

##########################--- crop type prediction --#############################

@app.route('/crop_recommend', methods = ['GET', 'POST'])
def crop_recommend():
    return render_template('crp_type.html')

@app.route('/predict', methods = ['GET', 'POST'])
def predict():
    if(request.method == "POST"):
        n = int(request.form['nit'])
        k = int(request.form['pot'])
        p = int(request.form['phos'])
        t = float(request.form['tem'])
        ph = float(request.form['ph'])
        r = float(request.form['rf'])
        h = float(request.form['hum'])

        arr=array([n,k,p,t,h,ph,r])
        arr=arr.reshape(1,-1)
        res = crop_prediction.predict(arr)

        crop_names = ['banana','bananan','cashewnut','coconut' ,'cotton' ,'drychillies', 'grapes',
 'groundnut' ,'maize' ,'mango', 'onion' ,'orange' ,'papaya', 'pomegranate',
 'ragi' ,'rice', 'sugarcane' ,'sunflower', 'sweetpoatato' ,'tobacco' ,'tomato',
 'tumeric']
        c1 = crop_names[res[0]]

        name = "The best crop for your feild is \'"+str(c1)+"\'"

        return render_template("open.html", n = name)

###############################--- mixed crop prediction ---################################

@app.route('/mixed_crop', methods = ['GET','POST'])
def mixed_crop():
    return render_template('mixed_crop.html')

@app.route('/crop_mixed', methods = ['GET', 'POST'])
def crop_mixed():
    if(request.method == "POST"):
        n = int(request.form['nit'])
        k = int(request.form['pot'])
        p = int(request.form['phos'])
        t = float(request.form['tem'])
        ph = float(request.form['ph'])
        r = float(request.form['rf'])
        h = float(request.form['hum'])

        arr=array([n,k,p,t,h,ph,r])
        arr=arr.reshape(1,-1)
        res = crop_prediction.predict(arr)

        crop_names = [['coffee','roses'],['coffee','cabbage'],['coffee','carrot'],['jute','rice'],['jute','green gram'],['jute','paddy'],['cotton','pulses'],['cotton','oilseeds'],['cotton','maize'],['coconut','sapota'],['coconut','banana'],['coconut','pepper'],['coconut','cocoa'],['papaya','mango'],['papaya','sapota'],['papaya','litche'],['orange','ground nut'],['orange','beans'],['orange','peas'],['apple','soya beans'],['apple','peanuts'],['apple','millets'],['muskmelon','suger cane'],['muskmelon','cotton'],['muskmelon','rice'],['watermelon','maize'],['watermelon','sorghum'],['grapes','pomogranate'],['grapes','brinjal'],['grapes','raddish'],['mango','lemon'],['mango','guava'],['mango','coconut'],['banana','spanish'],['banana','cauliflower'],['pomegranate','cauliflower'],['pomegranate','cabbage'],['lentil','musturd'],['lentil','barley'],['lentil','wheat'],['blackgram','sesame'],['blackgram','banana'],['blackgram','rice'],['mungbean','maize'],['mungbean','piegonpea'],['mungbean','sorghum'],['mothbeans','maize'],['mothbeans','sorghum'],['pigeonpeas','ground nuts'],['pigeonpeas','black gram'],['pigeonpeas','millets'],['kidneybeans','maize'],['kidneybeans','brinjal'],['chickpea','wheat'],['chickpea','musturd'],['maize','cotton'],['maize','piegonpea'],['rice','musturd'],['rice','black gram']]
        c1 = crop_names[res[0]]

        name = "The combination of crops to produce the high yeild is \'" + str(c1[0])+ "," + str(c1[1]) + "\'"

        return render_template("open.html", n = name)

#########################--- disease prediction ---##########################

@app.route('/image_disease', methods = ['GET', 'POST'])
def image_disease():
    return render_template('disease_prediction.html')

model_path = 'disease_prediction.h5'

model = load_model(model_path)
#model._make_predict_function()

cls1 = ['Black rot', 'healthy']

def predict1(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array,0)
    predictions = model.predict(img_array)
    val=argmax(predictions[0])
    predicted_class = cls1[val]
    confidence = round(100 * (max(predictions[0])), 2)
    return predicted_class, confidence, val

@app.route('/image_predict', methods = ['GET', 'POST'])
def image2():
    if(request.method == 'POST'):
        f=request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        image=imread(file_path)
        predicted_class, confidence,val  = predict1(model,image)

        img = im.fromarray(image)

        res = 'The predicted disease is \''+ predicted_class +"\'"
    return render_template("open.html", n = res)

######################################-- crop water requirement --#############################

@app.route('/crop_water', methods = ['GET', 'POST'])
def crop_water():
    return render_template('crop_water.html')

@app.route('/irrigation', methods = ['GET', 'POST'])
def irrigation():
    if(request.method == "POST"):
        ct = request.form['c_type']
        st = request.form['s_type']
        re = request.form['r']
        tp = request.form['t']
        wh = request.form['w']

        c_t = ['BANANA', 'BEAN', 'CABBAGE', 'CITRUS', 'COTTON', 'MAIZE', 'MELON', 'MUSTARD', 'ONION', 'POTATO', 'RICE', 'SOYABEAN', 'SUGARCANE', 'TOMATO', 'WHEAT']
        ct1 = c_t.index(ct.upper())

        s_t = ['DRY', 'HUMID', 'WET']
        st1 = s_t.index(st.upper())

        r_e = ['DESERT', 'HUMID', 'SEMI ARID', 'SEMI HUMID'] 
        re1 = r_e.index(re.upper())

        t_p = ['20-30', '30-40', '40-50', '10-20']
        tp1 = t_p.index(tp.upper())

        w_h = ['NORMAL', 'RAINY', 'SUNNY', 'WINDY']
        wh1 = w_h.index(wh.upper())

        arr=array([ct1,st1,re1,tp1,wh1])
        arr=arr.reshape(1,-1)
        res = c_w_r.predict(arr)

        name  = "The irrigation level requried for the crop is \'" + str(res[0]) + "\' centimeters"
        return render_template("open.html", n = name)

    else:
        return "Sorry!!"

###########################------ seed quality -----###########################

@app.route('/seed_quality', methods = ['GET', 'POST'])
def seed_quality():
    return render_template('seed_quality.html')

model_path2 = 'seed_quality.h5'

model2 = load_model(model_path2)
#model._make_predict_function()

cls2 =  ['good quality','bad quality']


def predict2(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    #print(img_array)
    img_array = tf.expand_dims(img_array,0)
    predictions = model.predict(img_array)
    print(predictions[0])
    val=argmax(predictions[0])
    print(val)
    print(cls2)
    predicted_class = cls2[val]
    confidence = round(100 * (max(predictions[0])), 2)
    return predicted_class, confidence,val


@app.route('/test_seed', methods = ['GET', 'POST'])
def test_seed():
    if(request.method == 'POST'):
        f=request.files['file']

        # Save the file to ./uploads2
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads2', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        image=imread(file_path)
        predicted_class, confidence,val = predict2(model2,image)

        img = im.fromarray(image)

        res = 'The seed quality is '+predicted_class + 'the cofidence value is '+str(confidence)+' '+str(val)
    return render_template("open.html", n = res)


############################-- chatter bot  --##############################

@app.route("/chatter_bot")
def chatter_bot():
    return render_template("chatter_bot.html")    

english_bot = ChatBot("Chatterbot")
trainer = ChatterBotCorpusTrainer(english_bot)
trainer.train("chatterbot.corpus.english")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return str(english_bot.get_response(userText))

###############################--- english to telugu --#################################
@app.route('/tel_eng')
def home_tel():
    return render_template('home_tel.html')


###############################--- About Us telugu--#################################
@app.route('/about_us_telugu')
def about_us_telugu():
    return render_template('about_us.html')

###############################--- Contact-us telugu--#################################
@app.route('/contact_us_telugu')
def contact_us_telugu():
    return render_template('contact_us.html')

##################--- crop yeild prediction telugu---##########################

@app.route('/yeild_tel', methods = ['GET', 'POST'])
def crop_yeild_telugu():
    return render_template('yeild_tel.html')
    
@app.route('//yeild_pred_tel', methods = ['GET', 'POST'])
def yeild_pred_tel():
    if(request.method == "POST"):

        s = request.form['s']
        d = request.form['d']
        ses =request.form['se']
        c = request.form['c']
        a = float(request.form['a'])
        s = 0

        dist=['అనంతపురం','చిత్తూరు', 'తూర్పు గోదావరి', 'గుంటూరు' ,'కడప', 'కృష్ణ','కర్నూలు' ,'ప్రకాశం', 'నెల్లూరు', 'శ్రీకాకుళం', 'ిఖపట్నం','విజయనగరం', 'పశ్చిమ గోదావరి']
        d1=dist.index(d)

        seso=['ఖరీఫ్', 'రబీ', 'మొత్తం సంవత్సరం']
        ses1=seso.index(ses)
 
        crp=['అరెకనట్', 'అర్హర్/తుర్', 'బజ్రా', 'అరటి', 'బీన్స్ & మట్టర్ (కూరగాయలు)', 'భిండి', 'బాటిల్ గోర్డ్','బ్రింజాల్', 'క్యాబేజీ', 'జీడిపప్పు', 'ఆముదం', 'సిట్రస్ ఫ్రూట్', 'కొబ్బరి', 'కొత్తిమీర', 'పత్తి(లింట్)','ఆవుపేడ(లోబియా)', 'దోసకాయ', 'ఎండు మిరపకాయలు', 'ఎండు అల్లం', 'వెల్లుల్లి', 'అల్లం', 'పప్పు', 'ద్రాక్ష', 'వేరుశనగ','హార్స్-గ్రామ్', 'జోవర్', 'కొర్ర', 'నిమ్మకాయ', 'లిన్సీడ్', 'మొక్కజొన్న', 'మామిడి', 'మసూర్', 'మేస్తా', 'మూంగ్(గ్రీన్ గ్రామ్)','నైజర్ సీడ్', 'ఉల్లి', 'నారింజ', 'ఇతర రబీ పప్పులు', 'ఇతర తాజా పండ్లు', 'ఇతర ఖరీఫ్ పప్పులు', 'ఇతర కూరగాయలు','బొప్పాయి', 'బఠానీలు (కూరగాయలు)', 'పోమ్ ఫ్రూట్', 'పోమ్ గ్రానెట్', 'పొటాటో', 'రాగి', 'రాప్‌సీడ్ & ఆవాలు', 'బియ్యం','కుసుమ', 'సమై', 'సపోటా', 'నువ్వు', 'చిన్న మినుములు', 'సోయాబీన్', 'చెరకు', 'పొద్దుతిరుగుడు', 'చిలగడదుంప','టపియోకా', 'పొగాకు', 'టమోటా', 'పసుపు', 'ఉరాడ్', 'వరగు', 'గోధుమలు', 'ఇతర ఫైబర్లు', 'ఇతర ఇతరాలు. పప్పులు', 'ఇతర నూనె గింజలు']
        c1=crp.index(c)

        arr=array([s,d1,ses1,c1,a])
        arr=arr.reshape(1,-1)
        res = crop_yeild.predict(arr)

        name = "ఉత్పత్తి చేయబడిన దిగుబడి \'"+str(res[0])+"\' హెక్టారుకు టన్నులు"
        return render_template("open.html", n = name)

###########################---- fertilizer type prediction telugu---#########################

@app.route('/fertilizer_tel', methods = ['GET', 'POST'])
def fertilizer_telugu():
    return render_template('fertilizer_tel.html')

@app.route('/tel_fertilizer', methods=['GET', 'POST'])
def tel_fertilizer():
    if(request.method == "POST"):

        nit = int(request.form['nitro'])
        potas = int(request.form['potash'])
        phos = int(request.form['phosp'])
        tem = int(request.form['temp'])
        soil = request.form['st']
        mo = int(request.form['moist'])
        humidity = int(request.form['humi'])
        crptyp = request.form['ct']

        soil1 = ['బ్లాక్', 'క్లేయ్', 'లోమీ', 'రెడ్', 'శాండీ']
        s1=soil1.index(soil.title())

        plants = ['బార్లీ', 'పత్తి', 'నేల గింజలు', 'మొక్కజొన్న', 'మిల్లెట్స్', 'నూనె గింజలు', 'వరి','పప్పులు', 'చెరకు', 'పొగాకు', 'గోధుమలు']
        c1 = plants.index(crptyp.title())

        arr=array([tem,humidity,mo,s1,c1,nit,potas,phos])
        arr=arr.reshape(1,-1)

        res = fertilizers.predict(arr)
        fer = ['10-26-26','14-35-14','17-17-17','20-20','28-28','DAP','Urea']
        res1 = fer[math.ceil(res[0])]

        print(res1)

        name = "మీ \'"+str(crptyp)+"\' పంటకు ఉత్తమ ఎరువులు \'"+str(res1) + "\'"
        print(name)

        return render_template("open.html", n = name)


##########################--- crop type prediction telugu--#############################
@app.route('/type_tel', methods = ['GET', 'POST'])
def crop_type_telugu():
    return render_template('type_tel.html')

@app.route('/predict_tel', methods=['GET', 'POST'])
def predict_tel():
    if(request.method == "POST"):
        n = int(request.form['nit'])
        k = int(request.form['pot'])
        p = int(request.form['phos'])
        t = float(request.form['tem'])
        ph = float(request.form['ph'])
        r = float(request.form['rf'])
        h = float(request.form['hum'])

        arr=array([n,k,p, t,h,ph,r])
        arr=arr.reshape(1,-1)

        res = crop_prediction.predict(arr)

        crop_names = ['అరటి', 'అరటి', 'జీడిపప్పు', 'కొబ్బరి', 'పత్తి', 'ఎండుమిర్చి', 'ద్రాక్ష','వేరుశెనగ', 'మొక్కజొన్న', 'మామిడి', 'ఉల్లి', 'నారింజ', 'బొప్పాయి', 'దానిమ్మ','రాగి' , 'బియ్యం', 'చెరకు' , 'పొద్దుతిరుగుడు', 'తీపి పొటాటో', 'పొగాకు', 'టమోటా','ట్యూమరిక్']
        c1 = crop_names[res[0]]

        name = "మీ పొలానికి ఉత్తమమైన పంట \'"+str(c1) +"\'"
        return render_template("open.html", n = name)

###############################--- mixed crop prediction telugu---################################

@app.route('/mixed_crop_telugu', methods = ['GET','POST'])
def mixed_crop_telugu():
    return render_template('mixed_tel.html')

@app.route('/crop_mixed_telugu', methods = ['GET', 'POST'])
def crop_mixed_telugu():
    if(request.method == "POST"):
        n = int(request.form['nit'])
        k = int(request.form['pot'])
        p = int(request.form['phos'])
        t = float(request.form['tem'])
        ph = float(request.form['ph'])
        r = float(request.form['rf'])
        h = float(request.form['hum'])

        arr=array([n,k,p,t,h,ph,r])
        arr=arr.reshape(1,-1)
        res = crop_prediction.predict(arr)

        crop_names = [['కాఫీ', 'గులాబీలు'],['కాఫీ', 'క్యాబేజీ'],['కాఫీ', 'క్యారెట్'],['జనపనార', 'బియ్యం'],['జనపనార', 'గ్రీన్ గ్రాము' ],['జనపనార', 'వరి'],['పత్తి', 'పప్పులు'],['పత్తి', 'నూనె గింజలు'],['పత్తి', 'మొక్కజొన్న'],['కొబ్బరి', 'సపోటా' ],['కొబ్బరి','అరటి'],['కొబ్బరి','మిరియాలు'],['కొబ్బరి','కోకో'],['బొప్పాయి','మామిడి'],['బొప్పాయి','సపోటా' ],['బొప్పాయి', 'లిచే'],['నారింజ', 'నేల గింజ'],['నారింజ', 'బీన్స్'],['నారింజ', 'బఠానీలు'],['యాపిల్', 'సోయా బీన్స్'],['యాపిల్', 'వేరుశెనగలు'],['యాపిల్', 'మిల్లెట్స్'],['మస్క్మెలోన్', 'పంచదార చెరకు'],['మస్కమెలన్', 'పత్తి'],['మస్కమెలన్', 'బియ్యం'],['పుచ్చకాయ', 'మొక్కజొన్న'],['పుచ్చకాయ', 'జొన్న'],['ద్రాక్ష', 'దానిమ్మ'],['ద్రాక్ష', 'వంకాయ'],['ద్రాక్ష', 'ముల్లంగి'],['మామిడి', 'నిమ్మకాయ'],['మామిడి', 'జామ'],['మామిడి', 'కొబ్బరి'],['అరటి', 'స్పానిష్'],['అరటి', 'కాలీఫ్లవర్'],['దానిమ్మ', 'కాలీఫ్లవర్'],['దానిమ్మ', 'క్యాబేజీ'],['పప్పు', 'ఆవాలు'],['పప్పు', 'బార్లీ'],['పప్పు', 'గోధుమ'],['బ్లాక్గ్రామ్', 'నువ్వులు'],['బ్లాక్గ్రామ్', 'అరటి'],['బ్లాక్గ్రామ్', 'బియ్యం'],['ముంగ్బీన్', 'మొక్కజొన్న'],['ముంగ్బీన్', 'పైగాన్పీ'],['ముంగ్బీన్', 'జొన్న'],['మోత్బీన్స్','మొక్కజొన్న'],['మోత్బీన్స్', 'జొన్న'],['పావురం', 'నేల గింజలు'],['పావురం', 'నల్లపప్పు'],['పావురం', 'మిల్లెట్లు'],['కిడ్నీబీన్స్' ,'మొక్కజొన్న'],['కిడ్నీబీన్స్','వంకాయ'],['చికపా', 'గోధుమ'],['చిక్పా', 'ఆవాలు'],['మొక్కజొన్న', 'పత్తి'],['మొక్కజొన్న' ,'పైగాన్‌పా'],['బియ్యం', 'ఆవాలు'],['బియ్యం', 'నల్లపప్పు']]

        #crop_names = [['coffee','roses'],['coffee','cabbage'],['coffee','carrot'],['jute','rice'],['jute','green gram'],['jute','paddy'],['cotton','pulses'],['cotton','oilseeds'],['cotton','maize'],['coconut','sapota'],['coconut','banana'],['coconut','pepper'],['coconut','cocoa'],['papaya','mango'],['papaya','sapota'],['papaya','litche'],['orange','ground nut'],['orange','beans'],['orange','peas'],['apple','soya beans'],['apple','peanuts'],['apple','millets'],['muskmelon','suger cane'],['muskmelon','cotton'],['muskmelon','rice'],['watermelon','maize'],['watermelon','sorghum'],['grapes','pomogranate'],['grapes','brinjal'],['grapes','raddish'],['mango','lemon'],['mango','guava'],['mango','coconut'],['banana','spanish'],['banana','cauliflower'],['pomegranate','cauliflower'],['pomegranate','cabbage'],['lentil','musturd'],['lentil','barley'],['lentil','wheat'],['blackgram','sesame'],['blackgram','banana'],['blackgram','rice'],['mungbean','maize'],['mungbean','piegonpea'],['mungbean','sorghum'],['mothbeans','maize'],['mothbeans','sorghum'],['pigeonpeas','ground nuts'],['pigeonpeas','black gram'],['pigeonpeas','millets'],['kidneybeans','maize'],['kidneybeans','brinjal'],['chickpea','wheat'],['chickpea','musturd'],['maize','cotton'],['maize','piegonpea'],['rice','musturd'],['rice','black gram']]
        c1 = crop_names[res[0]]
        name = "పంట \'"+ str(c1[0]) + "," +str(c1[1]) +"\' కలయిక మీ ఫీల్డ్కు ఉత్తమంగా సరిపోతుంది."
        return render_template("open.html", n = name)

#########################--- disease prediction telugu---##########################

@app.route('/image_disease_telugu', methods = ['GET', 'POST'])
def image_disease_telugu():
    return render_template('disease_tel.html')

model_path = 'disease_prediction.h5'

model = load_model(model_path)
#model._make_predict_function()

cls3 = ['నల్ల తెగులు', 'ఆరోగ్యకరమైన పంట']

def predict3(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array,0)
    predictions = model.predict(img_array)
    val=argmax(predictions[0])
    predicted_class = cls3[val]
    confidence = round(100 * (max(predictions[0])), 2)
    return predicted_class, confidence, val

@app.route('/image_predict_telugu', methods = ['GET', 'POST'])
def image_predict_telugu():
    if(request.method == 'POST'):
        f=request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        image=imread(file_path)
        predicted_class, confidence,val  = predict3(model,image)

        img = im.fromarray(image)

        res = 'ఊహించిన వ్యాధి '+ predicted_class
    return render_template("open.html", n = res)

######################################-- crop water requirement telugu --#############################

@app.route('/crop_water_telugu', methods = ['GET', 'POST'])
def crop_water_telugu():
    return render_template('crop_water_tel.html')

@app.route('/irrigation_telugu', methods = ['GET', 'POST'])
def irrigation_telugu():
    if(request.method == "POST"):
        ct = request.form['c_type']
        st = request.form['s_type']
        re = request.form['r']
        tp = request.form['t']
        wh = request.form['w']

        c_t = ['అరటి', 'బీన్', 'క్యాబేజీ', 'సిట్రస్', 'పత్తి', 'మొక్కజొన్న', 'పుచ్చకాయ', 'ఆవాలు', 'ఉల్లిపాయ', 'బంగాళదుంప', 'బియ్యం', 'సోయాబీన్', 'చెరకు ', 'టమోటో', 'గోధుమ']
        ct1 = c_t.index(ct)

        s_t = ['పొడి', 'తేమ', 'తడి']
        st1 = s_t.index(st)

        r_e = ['ఎడారి', 'తేమ', 'సెమీ ఎరిడ్', 'సెమీ హ్యూమిడ్'] 
        re1 = r_e.index(re)

        t_p = ['20-30', '30-40', '40-50', '10-20']
        tp1 = t_p.index(tp)

        w_h = ['నార్మల్', 'వర్షం', 'సన్నీ', 'గాలులు']
        wh1 = w_h.index(wh)

        arr=array([ct1,st1,re1,tp1,wh1])
        arr=arr.reshape(1,-1)
        res = c_w_r.predict(arr)

        name = "పొలం కోసం నీటిపారుదల స్థాయి \'" + str(res[0]) + "\' సెంటీమీటర్లు"
        return render_template("open.html", n = name)

    else:
        return "Sorry!!"

##################--- seed quality telugu ---##############

@app.route('/seed_quality_telugu', methods = ['GET', 'POST'])
def seed_quality_telugu():
    return render_template('seed_tel.html')

model_path2 = 'seed_quality.h5'

model2 = load_model(model_path2)
#model._make_predict_function()

cls4 = ['మంచి నాణ్యత', 'చెడు నాణ్యత']

def predict4(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    #print(img_array)
    img_array = tf.expand_dims(img_array,0)
    predictions = model.predict(img_array)
    print(predictions[0])
    val=argmax(predictions[0])
    predicted_class = cls4[val]
    confidence = round(100 * (max(predictions[0])), 2)
    return predicted_class, confidence,val


@app.route('/test_seed_telugu', methods = ['GET', 'POST'])
def test_seed_telugu():
    if(request.method == 'POST'):
        f=request.files['file']

        # Save the file to ./uploads2
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads2', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        image=imread(file_path)
        predicted_class, confidence,val = predict4(model2,image)

        img = im.fromarray(image)

        res = "విత్తన నాణ్యత \'"+predicted_class + "\'"

    return render_template("open.html", n = res)

###################################################################################

if __name__ == "__main__":
    app.run(debug = True)

    
