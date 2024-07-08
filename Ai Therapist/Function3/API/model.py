import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model

gameTrig = ["YES","NO"]

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('C:/Users/ashen/OneDrive/Documents/GitHub/Sipsuru-ML/Function3/API/content_2ndgen.json').read())

words = pickle.load(open('C:/Users/ashen/OneDrive/Documents/GitHub/Sipsuru-ML/Function3/API/words.pkl','rb'))
classes = pickle.load(open('C:/Users/ashen/OneDrive/Documents/GitHub/Sipsuru-ML/Function3/API/classes.pkl','rb'))
model = load_model('C:/Users/ashen/OneDrive/Documents/GitHub/Sipsuru-ML/Function3/API/chatbot_model.h5')

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