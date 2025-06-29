from torch.utils.data import DataLoader
from models import MultimodalSentimentModel, MultimodalTrainer
import torch


def test_logging():

    mock_batch = {
        'text_inputs': {
            'input_ids': torch.ones((1, 16), dtype=torch.long),
            'attention_mask': torch.ones((1, 16), dtype=torch.long)
        },
        'video_frames': torch.randn(1, 3, 16, 112, 112),  # B x C x T x H x W
        'audio_features': torch.randn(1, 128, 128),       # B x F x T
        'emotion_label': torch.tensor([1]),               # Batch of 1
        'sentiment_label': torch.tensor([2])              # Batch of 1
    }

    mock_loader = DataLoader([mock_batch])

    model = MultimodalSentimentModel()
    trainer = MultimodalTrainer(model, mock_loader, mock_loader)

    train_losses = {
        'total': 2.5,
        'emotion': 1.0,
        'sentiment': 1.5
    }

    trainer.log_metrics(train_losses, phase="train")

    val_losses = {
        'total': 1.5,
        'emotion': 0.5,
        'sentiment': 1.0
    }
    val_metrics = {
        'emotion_precision': 0.65,
        'emotion_accuracy': 0.75,
        'sentiment_precision': 0.85,
        'sentiment_accuracy': 0.95
    }

    trainer.log_metrics(val_losses, val_metrics, phase="val")


if __name__ == "__main__":
    test_logging()