from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
import os
from fastapi.responses import StreamingResponse
from inference import run_inference_stream

router = APIRouter()

TEMP_DIR = "/tmp"
os.makedirs(TEMP_DIR, exist_ok=True)

@router.post("/predict-stream")
async def predict_stream(video: UploadFile = File(...)):
    if not video.filename.endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only .mp4 files are supported")

    temp_path = os.path.join("/tmp", video.filename)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    async def stream_results():
        yield '{"utterances": [\n'
        first = True
        async for segment_json in run_inference_stream(temp_path):
            if not first:
                yield ',\n'
            yield segment_json
            first = False
        yield '\n]}'

        if os.path.exists(temp_path):
            os.remove(temp_path)

    return StreamingResponse(stream_results(), media_type="application/json")