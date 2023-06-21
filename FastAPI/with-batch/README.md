## Update the Server

```python
import pickle
import numpy as np
from typing import List, Union
from fastapi import FastAPI
from pydantic import BaseModel, conlist

class HouseRent(BaseModel):
    batches: List[conlist(item_type=Union[int, str, float], min_items=10, max_items=10)]
     
@app.post("/predict")
def predict(rent: HouseRent):
    batches = rent.batches
    np_batches = np.array(batches)
    pred = model.predict(np_batches).tolist()
    return {"Prediction": pred}
```



### Putting it all together

The resulting `Dockerfile` will look like this:

```Dockerfile
FROM frolvlad/alpine-miniconda3:python3.7

COPY requirements.txt .

RUN pip install -r requirements.txt && rm requirements.txt

EXPOSE 80

COPY ./app /app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
```


## Build the new Image
the `no-batch` directory and use the `docker build` command.
```bash
docker build -t houserent:with-batch .
```


## Run the Container
You can do so by using the following command:

```bash
docker run --rm -p 80:80 houserent:with-batch
```


## Make request to the server

```bash
curl -X POST http://localhost:80/predict \
    -d @./examples.json \
    -H "Content-Type: application/json"
```
