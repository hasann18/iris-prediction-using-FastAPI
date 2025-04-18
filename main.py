from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pickle
import numpy as np
import os

app = FastAPI()

# Load the trained model using a relative path
model = None
model_path = os.path.join(os.path.dirname(__file__), 'iris.pkl')

try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    print(f"Model loaded successfully from {model_path}")

except FileNotFoundError:
    print(f"Model file not found at: {model_path}")
    
except Exception as e:
    print(f"Error loading model: {e}")

# Serve static files (CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates directory
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request,
                  sepal_length: float = Form(...),
                  sepal_width: float = Form(...),
                  petal_length: float = Form(...),
                  petal_width: float = Form(...)):

    if model is None:
        return templates.TemplateResponse("index.html", {"request": request, "error": "Model not loaded"})

    data = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)

    try:
        prediction = model.predict(data)[0]
        print("📢 Prediction Output:", prediction)

        # Smart handling: if model returns string, use it directly
        if isinstance(prediction, str):
            species = prediction
        else:
            species_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
            species = species_mapping.get(prediction, str(prediction))

    except Exception as e:
        return templates.TemplateResponse("index.html", {"request": request, "error": f"Error making prediction: {e}"})

    return templates.TemplateResponse("index.html", {"request": request, "species": species})


# API Route
class IrisRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class IrisResponse(BaseModel):
    species: str

@app.post("/api/predict", response_model=IrisResponse)
async def api_predict(iris: IrisRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    data = np.array([iris.sepal_length, iris.sepal_width, iris.petal_length, iris.petal_width]).reshape(1, -1)

    try:
        prediction = model.predict(data)[0]
        print("📢 API Prediction Output:", prediction)

        if isinstance(prediction, str):
            species = prediction
        else:
            species_mapping = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
            species = species_mapping.get(prediction, str(prediction))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {e}")

    return IrisResponse(species=species)
