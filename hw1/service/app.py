from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
import pickle
import pandas as pd
import numpy as np
from fastapi.responses import StreamingResponse
import csv
import io
model = pickle.load("../models/best_ridge_model.pkl")

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    df = pd.DataFrame([item.dict()])
    prediction = model.predict(df)

    return prediction[0]


@app.post("/predict_items")
def predict_items(file: UploadFile) -> StreamingResponse:
    try:
        # Проверка формата файла
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Uploaded file is not a CSV.")

        # Чтение загруженного файла
        content = file.file.read().decode("utf-8")
        data = pd.read_csv(io.StringIO(content))


        # Предсказания
        predictions = model.predict(data)
        data["predicted_price"] = predictions

        # Создание CSV с предсказаниями
        output = io.StringIO()
        data.to_csv(output, index=False)
        output.seek(0)

        # Возврат CSV в виде файла
        response = StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv"
        )
        response.headers["Content-Disposition"] = "attachment; filename=predictions.csv"
        return response

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")