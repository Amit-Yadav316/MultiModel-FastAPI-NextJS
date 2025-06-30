import os
import subprocess
import cv2
import numpy as np
import torch
import torchaudio

def install_ffmpeg():
    print("Starting Ffmpeg installation...")

    subprocess.check_call([sys.executable, "-m", "pip",
                          "install", "--upgrade", "pip"])

    subprocess.check_call([sys.executable, "-m", "pip",
                          "install", "--upgrade", "setuptools"])

    try:
        subprocess.check_call([sys.executable, "-m", "pip",
                               "install", "ffmpeg-python"])
        print("Installed ffmpeg-python successfully")
    except subprocess.CalledProcessError as e:
        print("Failed to install ffmpeg-python via pip")

    try:
        subprocess.check_call([
            "wget",
            "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz",
            "-O", "/tmp/ffmpeg.tar.xz"
        ])

        subprocess.check_call([
            "tar", "-xf", "/tmp/ffmpeg.tar.xz", "-C", "/tmp/"
        ])

        result = subprocess.run(
            ["find", "/tmp", "-name", "ffmpeg", "-type", "f"],
            capture_output=True,
            text=True
        )
        ffmpeg_path = result.stdout.strip()

        subprocess.check_call(["cp", ffmpeg_path, "/usr/local/bin/ffmpeg"])

        subprocess.check_call(["chmod", "+x", "/usr/local/bin/ffmpeg"])

        print("Installed static FFmpeg binary successfully")
    except Exception as e:
        print(f"Failed to install static FFmpeg: {e}")

    try:
        result = subprocess.run(["ffmpeg", "-version"],
                                capture_output=True, text=True, check=True)
        print("FFmpeg version:")
        print(result.stdout)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("FFmpeg installation verification failed")
        return False

class VideoProcessor:
    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        try:
            if not cap.isOpened():
                raise ValueError(f"Video not found: {video_path}")

            ret, frame = cap.read()
            if not ret or frame is None:
                raise ValueError(f"Video not found: {video_path}")

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            while len(frames) < 30 and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (224, 224))
                frame = frame / 255.0
                frame = (frame - np.array([0.43216, 0.394666, 0.37645])) / np.array([0.22803, 0.22145, 0.216989])
                frames.append(frame)

        except Exception as e:
            raise ValueError(f"Video error: {str(e)}")
        finally:
            cap.release()

        if (len(frames) == 0):
            raise ValueError("No frames could be extracted")

        if len(frames) < 30:
            frames += [np.zeros_like(frames[0])] * (30 - len(frames))
        else:
            frames = frames[:30]


        return torch.FloatTensor(np.array(frames)).permute(0, 3, 1, 2)

class AudioProcessor:
    def extract_features(self, video_path, max_length=300):
        audio_path = video_path.replace('.mp4', '.wav')

        try:
            subprocess.run([
                'ffmpeg',
                '-i', video_path,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                audio_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            waveform, sample_rate = torchaudio.load(audio_path)

            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)

            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_mels=64,
                n_fft=1024,
                hop_length=512
            )

            mel_spec = mel_spectrogram(waveform)

            
            mel_spec = (mel_spec - mel_spec.mean()) / mel_spec.std()

            if mel_spec.size(2) < 300:
                padding = 300 - mel_spec.size(2)
                mel_spec = torch.nn.functional.pad(mel_spec, (0, padding))
            else:
                mel_spec = mel_spec[:, :, :300]

            return mel_spec

        except subprocess.CalledProcessError as e:
            raise ValueError(f"Audio extraction error: {str(e)}")
        except Exception as e:
            raise ValueError(f"Audio error: {str(e)}")
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

class VideoUtteranceProcessor:
    def __init__(self):
        self.video_processor = VideoProcessor()
        self.audio_processor = AudioProcessor()

    def extract_segment(self, video_path, start_time, end_time, temp_dir="/tmp"):
        duration = end_time - start_time
        if duration <= 0.3:
            print(f"Skipping segment {start_time} to {end_time} due to short duration")
            return None
        os.makedirs(temp_dir, exist_ok=True)
        segment_path = os.path.join(
            temp_dir, f"segment_{start_time}_{end_time}.mp4")
        
        try:

           subprocess.run([
            "ffmpeg", "-i", video_path,
            "-ss", str(start_time),
            "-t", str(end_time - start_time),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-y",
            segment_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

           if not os.path.exists(segment_path) or os.path.getsize(segment_path) == 0:
              raise ValueError("Segment extraction failed: " + segment_path)

           return segment_path

        except Exception as e :
         print(f"Segment failed ({start_time:.2f}-{end_time:.2f}): {str(e)}")
         return None

