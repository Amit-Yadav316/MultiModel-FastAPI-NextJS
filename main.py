from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import os
from inference import run_inference

app = FastAPI()

TEMP_DIR = "/tmp"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.post("/predict")
async def predict(video: UploadFile = File(...)):
    if not video.filename.endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only .mp4 files are supported")

    temp_path = os.path.join(TEMP_DIR, video.filename)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        result = run_inference(temp_path)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)