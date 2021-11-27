
from flask import Flask, render_template, Response, redirect,  url_for, jsonify, request
from dotenv import load_dotenv
from faker import Faker
from twilio.jwt.access_token import AccessToken
from twilio.jwt.access_token.grants import VoiceGrant
from twilio.twiml.voice_response import Dial, VoiceResponse
import cv2
import face_recognition
import numpy as np
import time
from pygame import mixer
import logging
import os
import re
import webbrowser
from PIL import Image
from os import listdir
from os.path import isfile, join
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import argparse


app = Flask(__name__, static_folder='static')
load_dotenv()
fake = Faker()
alphanumeric_only = re.compile("[\W_]+")
phone_pattern = re.compile(r"^[\d\+\-\(\) ]+$")
twilio_number = os.environ.get("TWILIO_CALLER_ID")
IDENTITY = {"identity": ""}
a = cv2.face.LBPHFaceRecognizer_create()
path = os.path.abspath('dataset')
print(os.path.dirname(os.path.abspath(__file__)))

def detect_and_predict_mask(frame, faceNet, maskNet):
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))
	faceNet.setInput(blob)
	detections = faceNet.forward()
	faces = []
	locs = []
	preds = []

	for i in range(0, detections.shape[2]):

		confidence = detections[0, 0, i, 2]

		if confidence > args["confidence"]:

			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			face = frame[startY:endY, startX:endX]
			if face.any():
				face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
				face = cv2.resize(face, (224, 224))
				face = img_to_array(face)
				face = preprocess_input(face)
				faces.append(face)
				locs.append((startX, startY, endX, endY))

	if len(faces) > 0:
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	return (locs, preds)


def getImageWithID(path):
    c = [os.path.join(path, f) for f in os.listdir(path)]
    d = []
    e = []  
    f = []  
    g = []
    for ca in c:
        h = Image.open(ca).convert('L')
        i = np.array(h, 'uint8')
        j = os.path.split(ca)[-1].split('.')[1]
        k = int(os.path.split(ca)[-1].split('.')[0])
        e.append(i)
        g.append(k)
        if j not in d:
            f.append(j)
        d.append(j)
        # cv2.imshow("training", i)
        # cv2.waitKey(10)
    return g, d, e, f
ga, d, e, f = getImageWithID(path)
a.train(e, np.array(ga))
print(a.write('trainingData.yml'))

# construct the argument parser and parse the arguments between 2 models
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskNet = load_model(args["model"])

print("[INFO] starting video stream...")
# cv2.destroyAllWindows()
l = cv2.VideoCapture(0)
m = [] 
n = [] 
o = 0
p = 1
q = 0
r = False
s = ""
t = True
# webbrowser.open(
#     'http://localhost:5000',  new=0, autoraise=True)
def classify_face():
    
    u = "assets/voice"
    v = [f for f in listdir(u) if isfile(join(u, f))]
    w = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    x = cv2.face.LBPHFaceRecognizer_create()
    print(x.read('trainingData.yml'))
    x.read('trainingData.yml')
    y= 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    m = [] 
    n = [] 
    o = 0
    p = 1
    q = 0
    r = False
    s = ""
    t = True
    z = []
    while True:
        ret, aa = l.read()
        ab = cv2.cvtColor(aa, cv2.COLOR_BGR2GRAY)
        e = w.detectMultiScale(ab, 1.3, 5)
        m = []
        ## co with Xmask here !
        (locs, preds) = detect_and_predict_mask(aa, faceNet, maskNet)
        
        if t:
            for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred
                
                if mask > withoutMask:
                    m.append("unknown")
                
            for (ac, ad, ae, af) in e:
                if(ae > 100 and af > 100):
                    cv2.rectangle(aa, (ac, ad),
                                  (ac+ae, ad+af), (0, 0, 255), 2)
                    y, ag = x.predict(ab[ad:ad+af, ac:ac+ae])
                    ah = ga.index(y) 
                    if y in ga and ag < 60:
                        m.append(d[ah])
                    else:
                        m.append("unknown") 
                    pass
            if set(m).issubset(set(n)) and (n and m) and r == False:
                r = True
                o = time.time()
                s = m[0]
                print("True")
            elif not set(m).issubset(set(n)) and r == True:
                o = time.time()
                r = False
                s = ""
            elif not m and not n:
                o = time.time()
                r = False
                s = ""
            print(time.time() - o)
            if (time.time() - o > p) and r and s != "unknown" and s and s not in z:
                ah = f.index(s)
                z.append(s)
                print("say hi :", s)
                r = False
                s = ""
                webbrowser.open(
                    'http://localhost:5000/', new=0, autoraise=True)
                mixer.init()
                mixer.music.load(
                    "assets/voice/"+v[ah])
                mixer.music.play()
                while mixer.music.get_busy(): 
                    time.sleep(4)
                return
            if (time.time() - o > p) and r and s == "unknown" and s:
                r = False
                s = ""
                mixer.init()
                mixer.music.load("voice/Hello.mp3")
                mixer.music.play()
                while mixer.music.get_busy(): 
                    time.sleep(4)
                webbrowser.open(
                    'http://localhost:5000/department', new=0, autoraise=True)
                return
            n = m
            print("-----")
        t = not t
@app.route('/')
def index():
    return render_template('index.html',)
@app.route('/department')
def department():
    return render_template('department.html',)
@app.route('/recording')
def rec():
    return Response(classify_face(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route("/token", methods=["GET"])
def token():
    account_sid = os.environ["TWILIO_ACCOUNT_SID"]
    application_sid = os.environ["TWILIO_TWIML_APP_SID"]
    api_key = os.environ["API_KEY"]
    api_secret = os.environ["API_SECRET"]
    identity = alphanumeric_only.sub("", fake.user_name())
    IDENTITY["identity"] = identity
    token = AccessToken(account_sid, api_key, api_secret, identity=identity)
    voice_grant = VoiceGrant(
        outgoing_application_sid=application_sid,
        incoming_allow=True,
    )
    token.add_grant(voice_grant)
    token = token.to_jwt()
    return jsonify(identity=identity, token=token)
@app.route("/voice", methods=["POST"])
def voice():
    resp = VoiceResponse()
    if request.form.get("To") == twilio_number:
        dial = Dial()
        dial.client(IDENTITY["identity"])
        resp.append(dial)
    elif request.form.get("To"):
        dial = Dial(caller_id=twilio_number)
        if phone_pattern.match(request.form["To"]):
            dial.number(request.form["To"])
        else:
            dial.client(request.form["To"])
        resp.append(dial)
    else:
        resp.say("Thanks for calling!")
    return Response(str(resp), mimetype="text/xml")
if __name__ == '__main__':
    app.run(debug=True)
     
