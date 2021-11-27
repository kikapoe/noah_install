import face_recognition
import argparse
import cv2
import numpy as np
import glob
import time
from pygame import Color, mixer
import os
import cv2
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

recognizer = cv2.face.LBPHFaceRecognizer_create()
path = 'dataset'

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			if face.any():
				face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
				face = cv2.resize(face, (224, 224))
				face = img_to_array(face)
				face = preprocess_input(face)

				# add the face and bounding boxes to their respective
				# lists
				faces.append(face)
				locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

def getImageWithID(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    names = []
    names_idx = []
    faces = []
    IDs = []
    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L')
        faceNp= np.array(faceImg, 'uint8')
        NAME = os.path.split(imagePath)[-1].split('.')[1]
        ID = int(os.path.split(imagePath)[-1].split('.')[0])
        faces.append(faceNp)
        IDs.append(ID)
        if NAME not in names:
            names_idx.append(NAME)
        names.append(NAME)
        cv2.imshow("training", faceNp)
        cv2.waitKey(10)
    return IDs, names, faces, names_idx

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

# load serialized face detector model from local
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from local
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")

Ids,names, faces, names_idx = getImageWithID(path)
recognizer.train(faces, np.array(Ids))
print(recognizer.write('trainingData.yml'))
cv2.destroyAllWindows()

print("face",len(Ids))
bgVDO = cv2.VideoCapture("video/test.mp4")
video_capture = cv2.VideoCapture(0)

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
rec = cv2.face.LBPHFaceRecognizer_create()
print(rec.read('trainingData.yml'))
id = 0
font = cv2.FONT_HERSHEY_SIMPLEX

# voice_list = [
#     'voice/Hello_Poom.mp3',
#     'voice/Hello_Panisara.mp3',
#     'voice/Hello_Pote.mp3',
# ]

mypath = "assets/voice/"
voice_list = [f for f in listdir(mypath) if isfile(join(mypath, f))]

# Count the number of loop that face not found
# if reach N then system can call the name again
# name_count = [0]*len(known_face_names)


same_face_count = 0
is_same_face = False
is_unknown = False
previous_face = [""]
current_face = [""]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
last_face = []
process_this_frame = True

curr = [] # current face detected
last = [] # previous face detected
start_time = 0
period_time = 1
current_time = 0
sameface = False
sameface_name = "" 
while True: 
    ret, frame = video_capture.read()
    ret2, frame2 = bgVDO.read() 
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    # curr is using to nagivate the current events
    curr = []
    
    # Mask detection works in other thread
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
    
  
    # Only process every other frame of video to save time
    if process_this_frame:
        
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            label = "Mask" if mask > withoutMask else "No Mask"
            if label == "Mask":
                color = (0, 255, 0)
                curr.append("unknown")
            else :
                color = (0,0, 255)
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        
        for (x,y,w,h) in faces:
            if(w > 100 and h > 100):
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
                id, conf = rec.predict(gray[y:y+h, x:x+w])
                index = Ids.index(id) # order of item in list 
                # index = Ids.index(id) # order of item in list 
                if id in Ids and conf < 55:
                    cv2.putText(frame,str(names[index])+","+str(round(conf,4)),(x,y+h), font, 1,(255,255,255),2,cv2.LINE_AA)
                    curr.append(names[index])
                else :
                    cv2.putText(frame,"Unknown",(x,y+h), font, 1,(255,255,255),2,cv2.LINE_AA)
                    curr.append("unknown")
                pass
        
        print(last)
        print(curr)
        if set(curr).issubset(set(last)) and ( last and curr ) and sameface == False :
            sameface = True
            start_time = time.time()
            sameface_name = curr[0]
            print("True")
        elif not set(curr).issubset(set(last)) and sameface == True:
            start_time = time.time()
            sameface = False
            sameface_name = ""
        elif not curr and not last :
            start_time = time.time()
            sameface = False
            sameface_name = ""
            
        print(time.time() - start_time)
        if (time.time() - start_time > period_time) and sameface and sameface_name != "unknown" and sameface_name:
            index = names_idx.index(sameface_name)
            # mixer.init()
            # mixer.music.load("assets/voice/"+voice_list[index])
            # mixer.music.play()
            # while mixer.music.get_busy():  # wait for music to finish playing
            #     time.sleep(1)
            print("say hi :" , sameface_name)
            sameface = False
            sameface_name = ""

        if (time.time() - start_time > period_time) and sameface and sameface_name == "unknown" and sameface_name:
            mixer.init()
            mixer.music.load("voice/Hello.mp3")
            mixer.music.play()
            while mixer.music.get_busy():  # wait for music to finish playing
                time.sleep(1)
            sameface = False
            sameface_name = ""
            
        cv2.imshow("Faces",frame)
        if(cv2.waitKey(1) == ord('q')):
            break
        pass

        last = curr
        print("-----")
    process_this_frame = not process_this_frame

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # time.sleep(0.2)
# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
