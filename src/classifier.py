from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertTokenizerFast


# ============================================================
# Configuration
# ============================================================

MAX_LEN = 64

LABEL_MAP = {
    0: "promotional",
    1: "transactional",
}


# ============================================================
# Student CNN
# This architecture must match the training notebook exactly.
# ============================================================

class SMSSequenceCNN(nn.Module):
    def __init__(
        self,
        vocab_size=30522,
        embed_dim=64,
        num_filters=64,
        filter_sizes=(2, 3, 4),
        dropout_rate=0.2,
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0,
        )

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=num_filters,
                kernel_size=k,
            )
            for k in filter_sizes
        ])

        total_conv_out = num_filters * len(filter_sizes)

        self.dropout = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(
            total_conv_out,
            64,
        )

        self.fc2 = nn.Linear(
            64,
            1,
        )

    def forward(self, input_ids):
        # [batch, sequence]
        x = self.embedding(input_ids)

        # [batch, embedding_dim, sequence]
        x = x.permute(0, 2, 1)

        conv_outputs = []

        for conv in self.convs:
            x_conv = F.relu(conv(x))

            pooled = F.adaptive_max_pool1d(
                x_conv,
                1,
            ).squeeze(-1)

            conv_outputs.append(pooled)

        features = torch.cat(
            conv_outputs,
            dim=1,
        )

        features = self.dropout(features)

        hidden = F.relu(
            self.fc1(features)
        )

        logits = self.fc2(hidden).squeeze(-1)

        return logits


# ============================================================
# SMS Classifier
# ============================================================

class SMSClassifier:

    def __init__(self, model_dir: str):

        self.model_dir = Path(model_dir)

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # ----------------------------------------------------
        # Load tokenizer
        # ----------------------------------------------------

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            self.model_dir,
            local_files_only=True,
        )

        # ----------------------------------------------------
        # Create model
        # ----------------------------------------------------

        self.model = SMSSequenceCNN(
            vocab_size=self.tokenizer.vocab_size
        )

        # ----------------------------------------------------
        # Load trained weights
        # ----------------------------------------------------

        checkpoint_path = (
            self.model_dir / "student_cnn.pt"
        )

        state_dict = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            state_dict
        )

        self.model.to(self.device)

        self.model.eval()

    # ========================================================
    # Classify one SMS
    # ========================================================

    def classify(self, message: str):

        if not isinstance(message, str):
            raise TypeError(
                "message must be a string"
            )

        message = message.strip()

        if not message:
            raise ValueError(
                "message cannot be empty"
            )

        encoding = self.tokenizer(
            message,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(
            self.device
        )

        with torch.no_grad():

            logits = self.model(
                input_ids
            )

            probability = torch.sigmoid(
                logits
            ).item()

        if probability >= 0.5:
            label_id = 1
        else:
            label_id = 0

        category = LABEL_MAP[label_id]

        confidence = (
            probability
            if label_id == 1
            else 1.0 - probability
        )

        return {
            "category": category,
            "confidence": round(
                float(confidence),
                6,
            ),
            "transactional_probability": round(
                float(probability),
                6,
            ),
            "promotional_probability": round(
                float(1.0 - probability),
                6,
            ),
        }

    # ========================================================
    # Classify multiple SMS messages
    # ========================================================

    def classify_batch(self, messages):

        if not isinstance(messages, list):
            raise TypeError(
                "messages must be a list"
            )

        return [
            self.classify(message)
            for message in messages
        ]

    # ========================================================
    # Model information
    # ========================================================

    def model_info(self):

        parameter_count = sum(
            p.numel()
            for p in self.model.parameters()
        )

        return {
            "model": "SMSSequenceCNN",
            "model_type": "1D-CNN student model",
            "parameters": parameter_count,
            "max_sequence_length": MAX_LEN,
            "labels": LABEL_MAP,
            "device": str(self.device),
        }
