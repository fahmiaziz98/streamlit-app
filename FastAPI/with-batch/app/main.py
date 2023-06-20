import pickle
import numpy as np
from typing import List, Union
from pydantic import BaseModel, conlist
from fastapi import FastAPI


app = FastAPI(title="Prediction House rental with-batch")

class HouseRent(BaseModel):
    batches: List[Union[str, int, float]] = conlist(item_type=Union[str, int, float], min_items=10, max_items=10)
        
@app.on_event("startup")
def load_model():
    with open("final_model_v2.pkl", "rb") as file:
        global model
        model = pickle.load(file)
        
@app.get("/")
def home():
    return "Congratulations! Your API is working as expected. Now head over to http://localhost:81/docs"

@app.post("/predict")
def predict(rent: HouseRent):
    batches = [rent.batches]  # Wrap the batches in a list for batch prediction
    np_batches = np.array(batches)
    pred = model.predict(np_batches).tolist()
    return {"Prediction": pred}