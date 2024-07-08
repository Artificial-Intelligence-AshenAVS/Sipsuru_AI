import json
from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel
from model import predict_class,get_response,game_trig
intents = json.loads(open('C:/Users/ashen/OneDrive/Documents/GitHub/Sipsuru-ML/Function3/API/content_2ndgen.json').read())

app = FastAPI()


class TextIn(BaseModel):
    usertxt: str


class PredictionOut(BaseModel):
    SipsuruBot: str
    GameTrig:str


@app.get("/")
def home():
    return {"API_health_check": "OK", "model_version": "0.1.0"}


@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
    try:
        ints = predict_class(payload.usertxt)
        res = get_response(ints,intents)
        gameOut = game_trig(payload.usertxt)
        return {"SipsuruBot": res,
                "GameTrig":gameOut}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Please ask about your feelings")
    
import uvicorn
if __name__ == "__main__": 
    uvicorn.run(app, host="localhost", port=8002) 
