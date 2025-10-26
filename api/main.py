from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from io import BytesIO
from PIL import Image
import numpy as np, os, logging
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = FastAPI(title='Potato Disease Detector API')

MODEL_PATH = "api/models/best_model.h5"
CLASS_NAMES = ['healthy','early_blight','late_blight']

# simple structured logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

_model = None
_version = "0.1.0"

def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError('Model not found at {}'.format(MODEL_PATH))
        _model = load_model(MODEL_PATH)
        logger.info("Loaded model from %s", MODEL_PATH)
    return _model

def prepare_image(file_bytes, target_size=(224,224)):
    img = Image.open(BytesIO(file_bytes)).convert('RGB')
    img = img.resize(target_size)
    arr = img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)

class PredictResponse(BaseModel):
    class_name: str
    confidence: float

@app.get('/health')
def health():
    ok = os.path.exists(MODEL_PATH)
    return {'status':'ok' if ok else 'missing_model'}

@app.get('/version')
def version():
    return {'api_version': _version, 'model_path': MODEL_PATH}

@app.post('/predict', response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        model = get_model()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    try:
        x = prepare_image(contents)
        preds = model.predict(x)
        idx = int(np.argmax(preds, axis=1)[0])
        prob = float(np.max(preds))
        return PredictResponse(class_name=CLASS_NAMES[idx], confidence=prob)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail='Prediction failed: '+str(e))
