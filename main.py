# main.py
from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf 
from tensorflow import keras


app = FastAPI()

MODEL = keras.models.load_model("model//1.keras")
CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.get("/")
async def p():
    return 'Working'



@app.post("/ping")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())

    img_batch = np.expand_dims(image, 0)
    prediction = MODEL.predict(img_batch)

    return prediction

    



if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)