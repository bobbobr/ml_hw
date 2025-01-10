from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
import pickle
import pandas as pd
import numpy as np
from fastapi.responses import StreamingResponse
import csv
import io



with open("../models/best_ridge_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("../models/scaler.pkl", "rb") as file:
    scaler = pickle.load(file)
with open("../models/encoder.pkl", "rb") as file:
    encoder = pickle.load(file)


app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    seats: int


class Items(BaseModel):
    objects: List[Item]

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df['mileage'] = df['mileage'].str.replace(' kmpl', '').str.replace(' km/kg', '').astype(float)
    df['engine'] = df['engine'].str.replace(' CC', '').astype(float)
    df['max_power'] = df['max_power'].str.replace(' bhp', '').astype(float)
    if 'name' in df.columns:
        #df['brand'] = df['name'].str.split(' ', n=1).str[0]
        df = df.drop(columns=['name'], errors='ignore')

    
    categorical_columns = ['seats','fuel', 'seller_type', 'transmission', 'owner']
    #df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    categorical_data = df[categorical_columns]

    print(categorical_data)

    encoded_categorical_data = encoder.transform(categorical_data)
    encoded_columns = encoder.get_feature_names_out(categorical_columns)
    encoded_df = pd.DataFrame(encoded_categorical_data, columns=encoded_columns, index=df.index)
    
    df = pd.concat([df.drop(columns=categorical_columns), encoded_df], axis=1)
    
    df = df.fillna(0)

    print(df)

    return df
    

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    df = pd.DataFrame([item.model_dump()])
    df = preprocess_data(df)
    print(df)
    scaled_data = scaler.transform(df)

    prediction = model.predict(scaled_data)

    return prediction[0]


@app.post("/predict_items")
def predict_items(file: UploadFile) -> StreamingResponse:
    try:
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Uploaded file is not a CSV.")

        content = file.file.read().decode("utf-8")
        data = pd.read_csv(io.StringIO(content))
        print(data)
        processed_data = preprocess_data(data)
        print(processed_data)

        scaled_data = scaler.transform(processed_data)
        print(scaled_data)
        predictions = model.predict(scaled_data)
        data["predicted_price"] = predictions

        output = io.StringIO()
        data.to_csv(output, index=False)
        output.seek(0)

        response = StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv"
        )
        response.headers["Content-Disposition"] = "attachment; filename=predictions.csv"
        return response

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")