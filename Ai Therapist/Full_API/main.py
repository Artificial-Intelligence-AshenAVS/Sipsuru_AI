import os
import io
import cv2
import json
import nltk
import random
import pickle
import uvicorn 
import tempfile
import numpy as np 
import pandas as pd
import firebase_admin
from PIL import Image 
from io import BytesIO
import mediapipe as mp
import tensorflow as tf
from pydantic import BaseModel
from keras import backend as K
from gramformer import Gramformer
from firebase_admin import firestore
from firebase_admin import credentials
from nltk.stem import WordNetLemmatizer
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import load_model
from keras.preprocessing import image as keras_image
from fastapi import FastAPI, File, UploadFile, HTTPException




app = FastAPI()

model_f4 = tf.keras.models.load_model('D:/My projects/sipsuru/Ai Therapist/Full_API/image_CRNN.h5')
model_f4.compile(optimizer='adam',  # Use the optimizer you used during training
              loss='binary_crossentropy',  # Use the loss function you used during training
              metrics=['accuracy'])

model_f2 = tf.keras.models.load_model('D:/My projects/sipsuru/Ai Therapist/Full_API/best_model.h5')
model_f1 = tf.keras.models.load_model('D:/My projects/sipsuru/Ai Therapist/Full_API/action.h5') 

gf = Gramformer(models=1, use_gpu=False)

actions = np.array(['ADHD', 'OCD', 'Null'])
# 1. New detection variables
sequence = []
sentence = []
output=[]
threshold = 0.8

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

alphabets = u"ABCDEFGHIJKLMNOPQRSTUVWXYZ-' "
max_str_len = 24 # max length of input labels
num_of_characters = len(alphabets) + 1 # +1 for ctc pseudo blank
num_of_timestamps = 64 # max length of predicted labels


cred = credentials.Certificate(r"D:/My projects/sipsuru/Ai Therapist/Full_API/sipsuru-9a489-firebase-adminsdk-qbv63-fae0dc2608.json")
firebase_admin.initialize_app(cred)

db = firestore.client()
class PredictionOut(BaseModel):
    Sipsuru: str


def num_to_label(num):
    ret = ""
    for ch in num:
        if ch == -1:  # CTC Blank
            break
        else:
            ret+=alphabets[ch]
    return ret

def label_to_num(label):
    label_num = []
    for ch in label:
        label_num.append(alphabets.find(ch))
        
    return np.array(label_num)

def preprocess(img):
    (h, w) = img.shape
    
    final_img = np.ones([64, 256])*255 # blank white image
    
    # crop
    if w > 256:
        img = img[:, :256]
        
    if h > 64:
        img = img[:64, :]
    
    
    final_img[:h, :w] = img
    return cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE)

def read_file_as_image(data) : 
    img = Image.open(BytesIO(data)).convert('RGB') 
    img = img.resize((224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0
    return img_array

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )
    
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

gameTrig = ["YES","NO"]
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('D:\My projects\sipsuru\Ai Therapist\Full_API\content_2ndgen.json').read())
words = pickle.load(open('D:/My projects/sipsuru/Ai Therapist/Full_API/words.pkl','rb'))
classes = pickle.load(open('D:/My projects/sipsuru/Ai Therapist/Full_API/classes.pkl','rb'))
model = load_model('D:/My projects/sipsuru/Ai Therapist/Full_API/chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word ==w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse = True)
    return_list = []
    for r in results:
        return_list.append({'intent':classes[r[0]], 'probability':str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def game_trig(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse = True)
    tag_list = []
    for r in results:
        tag_list.append(classes[r[0]])
    if tag_list[0]=="join" or "wantplaygame":
        return gameTrig[0]
    else:
        return gameTrig[1]

class TextIn(BaseModel):
    usertxt: str


class PredictionOut(BaseModel):
    SipsuruBot: str
    GameTrig:str

@app.get("/")
def home():
    return {"API_health_check": "OK", "model_version": "0.1.0"}


@app.post("/predict_f1")
async def predict(file: UploadFile = File(...)):
    try:
        # Save the video bytes to a temporary file
        temp_file_path = os.path.join(tempfile.gettempdir(), "temp_video.mp4")
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(await file.read())

        # Set mediapipe model 
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            # Read the video file using OpenCV
            cap = cv2.VideoCapture(temp_file_path)

            sequence = []  # Initialize an empty sequence

            while cap.isOpened():
                # Read feed
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                
                # Draw landmarks
                draw_styled_landmarks(image, results)
                
                # Prediction logic
                keypoints = extract_keypoints(results)

                sequence.append(keypoints)
                sequence = sequence[-30:]

                if len(sequence) == 30:
                    res = model_f1.predict(np.expand_dims(sequence, axis=0))[0]
                    print(actions[np.argmax(res)])
                    output.append(actions[np.argmax(res)])

            cap.release()
            max_output = max(output, key=output.count)
            return {"message": str(max_output)}

    except Exception as e:
        # If an error occurs
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_f2") 
async def predict(file: UploadFile = File(...)): 
    try: 
        
        img = read_file_as_image(await file.read())
        predictions = model_f2.predict(img)
        if predictions[0][0] < 0.432:
            result = 'No Focused'
        else:
             result = 'Focused'
        return {   
            'confidence': result,
            'value':float(predictions[0][0]) 
        }
    except Exception as e: # If an error occurs
        raise HTTPException(status_code=400, detail=str(e))
    
@app.get("/predict_arima", response_model=PredictionOut)
def predict():
    current_date = datetime.now().date()
    future_date = current_date + timedelta(days=2)
    try:
        trimmed_timestamps = []
        progress_marks = []
        def get_all_docs(collectionName):
            docs = (db.collection(collectionName).stream())
            for doc in docs:
                collections = db.collection("students").document(f"{doc.id}").collection("game_marks")
                for doc1 in collections.stream():
                    doc_data = doc1.to_dict()
                    doc_data['docData'] = doc1._data
                    timestamp = doc_data['docData']['timestamp']
                    trimmed_timestamp = timestamp.split('T')[0]
                    progress_mark = doc_data['docData']['marks']
                    trimmed_timestamps.append(trimmed_timestamp)
                    progress_marks.append(progress_mark)
        get_all_docs("students")
        data = {'DATE': trimmed_timestamps, 'Prograss': progress_marks}
        df = pd.DataFrame(data)
        df['DATE'] = pd.to_datetime(df['DATE'])
        df.set_index('DATE', inplace=True)

        model = ARIMA(df['Prograss'], order=(1, 0, 5))
        model_fit = model.fit()
        index_future_dates=pd.date_range(start=current_date,end=future_date) #'2019-01-21' '2019-02-20'
        
        pred=model_fit.predict(start=len(df),end=len(df)+2,typ='levels')
       
        pred.index=index_future_dates
        return {"Sipsuru": float(pred.iloc[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
    try:
        ints = predict_class(payload.usertxt)
        res = get_response(ints,intents)
        gameOut = game_trig(payload.usertxt)
        return {"SipsuruBot": res,
                "GameTrig":gameOut}
    except Exception as e:
        return {"SipsuruBot": 'Please ask about your feelings', "GameTrig": "NO"}

@app.post("/predict_f4") 
async def predict(file: UploadFile = File(...)): 
    try: 
        
        file_content = await file.read()
        img = Image.open(io.BytesIO(file_content))
        # Convert the image to grayscale
        img = img.convert('L')
        
        # Convert the PIL Image to a NumPy array
        img_array = np.array(img)
        
        # Make predictions
        image = preprocess(img_array)
        image = image/255.
        
        # Reshape the image for prediction
        image = image.reshape(1, 64, 256, 1)
        pred = model_f4.predict(image.reshape(1, 256, 64, 1))
        decoded = K.get_value(K.ctc_decode(pred, input_length=np.ones(pred.shape[0])*pred.shape[1], 
                                            greedy=True)[0][0])
        correct_let = gf.correct(num_to_label(decoded[0]))
        marks = (np.max(pred)-0.9999)*10e5
        rounded_marks = round(marks, 2)
        return { 
            'confidence marks': rounded_marks 
        }
    except Exception as e: # If an error occurs
        raise HTTPException(status_code=400, detail=str(e))
    


if __name__ == "__main__": 
    uvicorn.run(app, host="0.0.0.0", port=8001)