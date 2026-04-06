from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
model = joblib.load("model.pkl")

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
    probability = model.predict_proba([input.text])[0]
    confidence = round(float(max(probability)) * 100, 1)
    result = model.predict([input.text])[0]
    if int(result) == 1:
        return {"label": "positive", "confidence": confidence}
    else:
        return {"label": "negative", "confidence": confidence}
    

