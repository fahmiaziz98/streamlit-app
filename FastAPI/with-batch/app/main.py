import pickle
import numpy as np
from typing import List, Union
from pydantic import BaseModel, conlist
from fastapi import FastAPI


app = FastAPI(title="Prediction House rental with-batch")

class HouseRent(BaseModel):
    batches: List[conlist(item_type=Union[int, str, float], min_items=10, max_items=10)]
        
@app.on_event("startup")
def load_model():
    with open("../app/lgbm_model.pkl", "rb") as file:
        global model
        model = pickle.load(file)
        
@app.get("/")
def home():
    return "Your API is working as expected. Now head over to http://localhost:81/docs"

@app.post("/predict")
def predict(rent: HouseRent):
    batches = rent.batches  
    np_batches = np.array(batches)
    pred = model.predict(np_batches).tolist()
    return {"Prediction": pred}