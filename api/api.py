# api/api.py

from fastapi import FastAPI
from pydantic import BaseModel
import pickle

app = FastAPI()

# Load model from current directory
with open('sentiment_model.pkl','rb') as f:
    data = pickle.load(f)

vect, model = data['vect'], data['model']

class Tweet(BaseModel):
    text: str

@app.post('/sentiment')
def sentiment(req: Tweet):
    vec = vect.transform([req.text])
    pred = model.predict(vec)[0]
    return {'sentiment': 'Positive' if pred==1 else 'Negative'}
