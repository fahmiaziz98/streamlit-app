# Dockerizing the server

This should result in a directory structure that looks like this:
```
..
└── no-batch
    ├── app/
    │   ├── main.py (server code)
    │   └── final_model_v2.pkl
    ├── requirements.txt (Python dependencies)
    ├── examples.json (prediction to test the server)
    ├── README.md (this file)
    └── Dockerfile
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


## Build the Image
the `no-batch` directory and use the `docker build` command.
```bash
docker build -t houserent:no-batch .
```


## Run the Container
You can do so by using the following command:

```bash
docker run --rm -p 80:80 houserent:no-batch
```


## Make request to the server

```bash
curl -X POST http://localhost:80/predict \
    -d @./examples.json \
    -H "Content-Type: application/json"
```
