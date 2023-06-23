import pickle
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI


app = FastAPI(title="Prediction House rental no-batch")

class HouseRent(BaseModel):
    bhk: int
    city: str
    furnishing_status: str
    tenant_prefered: str
    bathroom: int
    point_contract: str
    rental_floor: int
    total_num_floor: int
    fixed_size: float
    square_feet_rent: float
    
@app.on_event("startup")
def load_model():
    with open("../app/lgbm_model.pkl", "rb") as file:
        global model
        model = pickle.load(file)
        
@app.post("/predict")
def predict(rent: HouseRent):
    data_point = np.array(
        [
            [
                rent.bhk,
                rent.city,
                rent.furnishing_status,
                rent.tenant_prefered,
                rent.bathroom,
                rent.point_contract,
                rent.rental_floor,
                rent.total_num_floor,
                rent.fixed_size,
                rent.square_feet_rent
            ]
        ]
    )
    
    pred = model.predict(data_point).tolist()
    pred = pred[0]
    print(pred)
    return {"Prediction": np.round(pred, 2)}