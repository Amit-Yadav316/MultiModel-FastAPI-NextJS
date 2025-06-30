import torch
from models import MultimodalSentimentModel
from transformers import AutoTokenizer
import os
import json
import torchaudio
import cv2
import numpy as np
import subprocess
import whisper
import asyncio
from preprocessor import VideoUtteranceProcessor
from utils import EMOTION_MAP, SENTIMENT_MAP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_video_duration(video_path):
    import ffmpeg
    try:
        probe = ffmpeg.probe(video_path)
        duration = float(probe['format']['duration'])
        return duration
    except Exception as e:
        print(f"Error getting duration: {e}")
        return None


model = MultimodalSentimentModel().to(device)
model.load_state_dict(torch.load("model/best_model.pth", map_location=device))
model.eval()

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
transcriber = whisper.load_model("base", device=device)

async def run_inference_stream(video_path):
    result = await asyncio.to_thread(transcriber.transcribe, video_path, word_timestamps=True)
    utterance_processor = VideoUtteranceProcessor()
    video_duration = get_video_duration(video_path)
    for segment in result["segments"]:
        start_time = float(segment.get("start", 0))
        end_time = float(segment.get("end", 0))
        duration = end_time - start_time
        if end_time > video_duration or start_time < 0 or duration <= 0.3:
            print(f"Skipping segment ")
            continue
        segment_path = None
        try:
            segment_path = await asyncio.to_thread(
                utterance_processor.extract_segment, video_path, segment["start"], segment["end"]
            )
            if segment_path is None:
              print(f"Skipping processing for segment ")
              continue

            video_frames = await asyncio.to_thread(
                utterance_processor.video_processor.process_video, segment_path
            )
            audio_features = await asyncio.to_thread(
                utterance_processor.audio_processor.extract_features, segment_path
            )

            text_inputs = tokenizer(
                segment["text"], padding="max_length", truncation=True,
                max_length=128, return_tensors="pt"
            )
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            video_frames = video_frames.unsqueeze(0).to(device)
            audio_features = audio_features.unsqueeze(0).to(device)

            with torch.inference_mode():
                outputs = model(text_inputs, video_frames, audio_features)

            emotion_probs = torch.softmax(outputs["emotions"], dim=1)[0]
            sentiment_probs = torch.softmax(outputs["sentiments"], dim=1)[0]

            emotion_values, emotion_indices = torch.topk(emotion_probs, 3)
            sentiment_values, sentiment_indices = torch.topk(sentiment_probs, 3)

            prediction = {
                "start_time": segment["start"],
                "end_time": segment["end"],
                "text": segment["text"],
                "emotions": [
                    {"label": EMOTION_MAP[idx.item()], "confidence": round(conf.item(), 3)}
                    for idx, conf in zip(emotion_indices, emotion_values)
                ],
                "sentiments": [
                    {"label": SENTIMENT_MAP[idx.item()], "confidence": round(conf.item(), 3)}
                    for idx, conf in zip(sentiment_indices, sentiment_values)
                ]
            }
            yield json.dumps(prediction) + "\n"

        except Exception as e:
            print("Segment failed:", str(e))

        finally:
            if segment_path and os.path.exists(segment_path):
                os.remove(segment_path)