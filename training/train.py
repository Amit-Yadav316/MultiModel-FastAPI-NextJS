import os
import argparse
import torchaudio
import torch
from tqdm import tqdm
import json
from meld_dataset import prepare_dataloaders
from models import MultimodalSentimentModel, MultimodalTrainer
from install_ffmpeg import install_ffmpeg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--train-dir", type=str, default="./dataset/train")
    parser.add_argument("--val-dir", type=str, default="./dataset/val")
    parser.add_argument("--test-dir", type=str, default="./dataset/test")
    parser.add_argument("--model-dir", type=str, default="./saved_model")
    return parser.parse_args()


def main():
    install_ffmpeg()

    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")
    print("Available audio backends:", torchaudio.list_audio_backends())

    train_loader, val_loader, test_loader = prepare_dataloaders(
        train_csv=os.path.join(args.train_dir, 'train_sent_emo.csv'),
        train_video_dir=os.path.join(args.train_dir, 'train_splits'),
        dev_csv=os.path.join(args.val_dir, 'dev_sent_emo.csv'),
        dev_video_dir=os.path.join(args.val_dir, 'dev_splits_complete'),
        test_csv=os.path.join(args.test_dir, 'test_sent_emo.csv'),
        test_video_dir=os.path.join(args.test_dir, 'output_repeated_splits_test'),
        batch_size=args.batch_size
    )

    model = MultimodalSentimentModel().to(device)
    trainer = MultimodalTrainer(model, train_loader, val_loader)

    best_val_loss = float('inf')
    os.makedirs(args.model_dir, exist_ok=True)

    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")

        train_loss = trainer.train_epoch()
        val_loss, val_metrics = trainer.evaluate(val_loader)

        print(f"Train Loss: {train_loss['total']:.4f} | Val Loss: {val_loss['total']:.4f}")
        print(f"Val Emotion Acc: {val_metrics['emotion_accuracy']:.4f} | Sentiment Acc: {val_metrics['sentiment_accuracy']:.4f}")


        if val_loss["total"] < best_val_loss:
            best_val_loss = val_loss["total"]
            model_path = os.path.join(args.model_dir, "best_model.pth")
            torch.save(model.state_dict(), model_path)
            print(f" Saved best model to {model_path}")

    
    print("\nEvaluating on test set...")
    test_loss, test_metrics = trainer.evaluate(test_loader, phase="test")
    print(f"\nTest Emotion Accuracy: {test_metrics['emotion_accuracy']:.4f}")
    print(f"Test Sentiment Accuracy: {test_metrics['sentiment_accuracy']:.4f}")


if __name__ == "__main__":
    main()