from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import os
from inference import run_inference
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from route.predict import router as predict_router

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],              
    allow_credentials=True,
    allow_methods=["*"],               
    allow_headers=["*"],               
)

app.include_router(predict_router, prefix="/api", tags=["Predict"])


@app.get("/", tags=["Root"])
async def root():
    return {"message": "Welcome to Real-Time Audio Matcher"}
