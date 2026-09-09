from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer


MAX_LEN = 64

LABEL_MAP = {
    0: "promotional",
    1: "transactional",
}


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
                embed_dim,
                num_filters,
                kernel_size=k,
            )
            for k in filter_sizes
        ])

        total_features = (
            num_filters * len(filter_sizes)
        )

        self.dropout = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(
            total_features,
            64,
        )

        self.fc2 = nn.Linear(
            64,
            1,
        )

    def forward(self, input_ids):

        x = self.embedding(input_ids)

        x = x.permute(0, 2, 1)

        pooled_outputs = []

        for conv in self.convs:

            x_conv = F.relu(conv(x))

            pooled = F.adaptive_max_pool1d(
                x_conv,
                1,
            ).squeeze(-1)

            pooled_outputs.append(pooled)

        features = torch.cat(
            pooled_outputs,
            dim=1,
        )

        features = self.dropout(features)

        hidden = F.relu(
            self.fc1(features)
        )

        logits = self.fc2(hidden).squeeze(-1)

        return logits


class SMSClassifier:

    def __init__(self, model_dir):

        self.model_dir = Path(model_dir)

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # Load tokenizer.json directly
        tokenizer_path = (
            self.model_dir / "tokenizer.json"
        )

        self.tokenizer = Tokenizer.from_file(
            str(tokenizer_path)
        )

        # Create model
        self.model = SMSSequenceCNN(
            vocab_size=self.tokenizer.get_vocab_size()
        )

        # Load student CNN
        model_path = (
            self.model_dir / "student_cnn.pt"
        )

        state_dict = torch.load(
            model_path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            state_dict
        )

        self.model.to(self.device)
        self.model.eval()

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

        # Tokenize
        encoded = self.tokenizer.encode(
            message
        )

        input_ids = encoded.ids[:MAX_LEN]

        # Padding
        if len(input_ids) < MAX_LEN:
            input_ids += [
                0
            ] * (MAX_LEN - len(input_ids))

        input_ids = torch.tensor(
            [input_ids],
            dtype=torch.long,
            device=self.device,
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
                confidence,
                6,
            ),
            "transactional_probability": round(
                probability,
                6,
            ),
            "promotional_probability": round(
                1.0 - probability,
                6,
            ),
        }

    def classify_batch(self, messages):

        return [
            self.classify(message)
            for message in messages
        ]

    def model_info(self):

        parameter_count = sum(
            p.numel()
            for p in self.model.parameters()
        )

        return {
            "model": "SMSSequenceCNN",
            "parameters": parameter_count,
            "max_sequence_length": MAX_LEN,
            "labels": LABEL_MAP,
            "device": str(self.device),
        }
