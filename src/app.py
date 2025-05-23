from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import app_config

model = None
FEATURE_NAMES = None

# Модель входных данных
class InputData(BaseModel):
    features: list[float]


# Модель выходных данных
class OutputData(BaseModel):
    prediction: list[int]


@asynccontextmanager
async def lifespan(_: FastAPI):
    global model, FEATURE_NAMES
    model = joblib.load(app_config.path_to_modelfile)
    FEATURE_NAMES = app_config.feature_names
    print("Модель загружена")
    yield
    model = None
    FEATURE_NAMES = None
    print("Модель очищена")


app = FastAPI(lifespan=lifespan)

@app.post("/predict", response_model=OutputData)
async def predict(data: InputData) -> OutputData:
    if model is None or FEATURE_NAMES is None:
        raise HTTPException(status_code=500, detail="Модель не загружена")

    try:
        # Преобразование списка в массив
        input_df = pd.DataFrame([data.features], columns=FEATURE_NAMES)
        prediction = model.predict(input_df)
        predictions_list = [int(p) for p in prediction.tolist()]
        return OutputData(prediction=predictions_list)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
