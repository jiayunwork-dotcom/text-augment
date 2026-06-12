import asyncio
import json
import logging
import random
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
from collections import Counter

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.db_models import (
    TrainingExperiment, EvaluationResult, Dataset, DatasetVersion, Sample,
    TaskStatus, SampleSource, SplitType, TrainingMode, ModelBackbone
)
from ..config import MODEL_CACHE_DIR, MIN_SAMPLES_PER_CLASS, MAX_SELF_TRAINING_ITERATIONS, SELF_TRAINING_EARLY_STOP_PATIENCE

logger = logging.getLogger(__name__)


class TextDataset:
    def __init__(self, texts: list[str], labels: list[str], label2id: dict[str, int]):
        self.texts = texts
        self.labels = labels
        self.label2id = label2id

    def __len__(self):
        return len(self.texts)

    def get_numeric_labels(self) -> list[int]:
        return [self.label2id[l] for l in self.labels]


class BaseModelTrainer(ABC):
    def __init__(self, num_classes: int, label2id: dict, id2label: dict, hyperparams: dict):
        self.num_classes = num_classes
        self.label2id = label2id
        self.id2label = id2label
        self.hyperparams = hyperparams

    @abstractmethod
    async def train(self, train_dataset: TextDataset, val_dataset: TextDataset, progress_callback=None) -> dict:
        pass

    @abstractmethod
    async def evaluate(self, test_dataset: TextDataset) -> dict:
        pass

    @abstractmethod
    async def predict(self, texts: list[str]) -> tuple[list[int], list[float]]:
        pass


class DistilBertTrainer(BaseModelTrainer):
    async def train(self, train_dataset: TextDataset, val_dataset: TextDataset, progress_callback=None) -> dict:
        import torch
        from torch.utils.data import DataLoader, Dataset as TorchDataset
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=self.num_classes
        ).to(device)

        max_len = self.hyperparams.get("max_seq_length", 128)
        batch_size = self.hyperparams.get("batch_size", 16)
        epochs = self.hyperparams.get("epochs", 10)
        lr = self.hyperparams.get("learning_rate", 2e-5)
        patience = self.hyperparams.get("early_stopping_patience", 3)

        class TextTorchDataset(TorchDataset):
            def __init__(self, texts, labels, tokenizer, max_len, label2id):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_len = max_len
                self.label2id = label2id

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                encoding = self.tokenizer(
                    self.texts[idx],
                    max_length=self.max_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                return {
                    "input_ids": encoding["input_ids"].squeeze(0),
                    "attention_mask": encoding["attention_mask"].squeeze(0),
                    "labels": torch.tensor(self.label2id[self.labels[idx]]),
                }

        train_ds = TextTorchDataset(train_dataset.texts, train_dataset.labels, tokenizer, max_len, self.label2id)
        val_ds = TextTorchDataset(val_dataset.texts, val_dataset.labels, tokenizer, max_len, self.label2id)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)

        train_losses = []
        val_losses = []
        val_metrics = []
        best_val_metric = 0.0
        best_epoch = 0
        no_improve = 0
        best_state = None

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    val_loss += outputs.loss.item()
                    preds = torch.argmax(outputs.logits, dim=-1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = correct / max(total, 1)
            val_losses.append(avg_val_loss)
            val_metrics.append(val_accuracy)

            if val_accuracy > best_val_metric:
                best_val_metric = val_accuracy
                best_epoch = epoch
                no_improve = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1

            if progress_callback:
                await progress_callback(epoch + 1, epochs, avg_train_loss, avg_val_loss, val_accuracy)

            if no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

            await asyncio.sleep(0)

        if best_state:
            model.load_state_dict(best_state)

        save_path = MODEL_CACHE_DIR / f"model_{int(time.time())}"
        save_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(save_path))
        tokenizer.save_pretrained(str(save_path))

        return {
            "model_path": str(save_path),
            "train_loss_history": train_losses,
            "val_loss_history": val_losses,
            "val_metric_history": val_metrics,
            "best_epoch": best_epoch,
            "best_val_metric": best_val_metric,
            "total_epochs": epoch + 1 if 'epoch' in dir() else epochs,
        }

    async def evaluate(self, test_dataset: TextDataset) -> dict:
        import torch
        from torch.utils.data import DataLoader
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        save_path = self.hyperparams.get("_model_path", "")
        if not save_path or not Path(save_path).exists():
            return {"error": "Model not found"}

        tokenizer = AutoTokenizer.from_pretrained(save_path)
        model = AutoModelForSequenceClassification.from_pretrained(save_path).to(device)
        model.eval()

        max_len = self.hyperparams.get("max_seq_length", 128)
        batch_size = self.hyperparams.get("batch_size", 16)

        all_preds = []
        all_labels = []
        all_probs = []

        for i in range(0, len(test_dataset), batch_size):
            batch_texts = test_dataset.texts[i:i + batch_size]
            batch_labels = test_dataset.labels[i:i + batch_size]
            encodings = tokenizer(
                batch_texts, max_length=max_len, padding=True, truncation=True, return_tensors="pt"
            )
            with torch.no_grad():
                input_ids = encodings["input_ids"].to(device)
                attention_mask = encodings["attention_mask"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend([self.label2id[l] for l in batch_labels])
            all_probs.extend(probs.cpu().numpy().tolist())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)

        accuracy = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        per_class = {}
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        for label_id_str, metrics in report.items():
            if label_id_str in ("accuracy", "macro avg", "weighted avg"):
                continue
            label_name = self.id2label.get(int(label_id_str), label_id_str)
            per_class[label_name] = {
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1-score": metrics["f1-score"],
                "support": metrics["support"],
            }

        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "per_class_metrics": per_class,
        }

    async def predict(self, texts: list[str]) -> tuple[list[int], list[float]]:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        save_path = self.hyperparams.get("_model_path", "")
        if not save_path or not Path(save_path).exists():
            return [], []

        tokenizer = AutoTokenizer.from_pretrained(save_path)
        model = AutoModelForSequenceClassification.from_pretrained(save_path).to(device)
        model.eval()

        max_len = self.hyperparams.get("max_seq_length", 128)
        encodings = tokenizer(texts, max_length=max_len, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(input_ids=encodings["input_ids"].to(device), attention_mask=encodings["attention_mask"].to(device))
            probs = torch.softmax(outputs.logits, dim=-1)
            confidences, preds = torch.max(probs, dim=-1)

        return preds.cpu().numpy().tolist(), confidences.cpu().numpy().tolist()


class TextCNNTrainer(BaseModelTrainer):
    async def train(self, train_dataset: TextDataset, val_dataset: TextDataset, progress_callback=None) -> dict:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset as TorchDataset

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vocab = self._build_vocab(train_dataset.texts)
        embed_dim = 128
        num_filters = 100
        filter_sizes = [3, 4, 5]
        max_len = self.hyperparams.get("max_seq_length", 128)
        batch_size = self.hyperparams.get("batch_size", 16)
        epochs = self.hyperparams.get("epochs", 10)
        lr = self.hyperparams.get("learning_rate", 1e-3)
        patience = self.hyperparams.get("early_stopping_patience", 3)

        class TextCNNModel(nn.Module):
            def __init__(self, vocab_size, embed_dim, num_classes, num_filters, filter_sizes):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
                self.convs = nn.ModuleList([
                    nn.Conv1d(embed_dim, num_filters, k) for k in filter_sizes
                ])
                self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
                self.dropout = nn.Dropout(0.3)

            def forward(self, x):
                emb = self.embedding(x)
                emb = emb.permute(0, 2, 1)
                conv_outs = [torch.relu(conv(emb)) for conv in self.convs]
                pooled = [torch.max(c, dim=2)[0] for c in conv_outs]
                cat = torch.cat(pooled, dim=1)
                cat = self.dropout(cat)
                return self.fc(cat)

        model = TextCNNModel(len(vocab), embed_dim, self.num_classes, num_filters, filter_sizes).to(device)

        class CNDataset(TorchDataset):
            def __init__(self, texts, labels, vocab, max_len, label2id):
                self.data = []
                for text, label in zip(texts, labels):
                    tokens = text.lower().split()
                    ids = [vocab.get(t, 1) for t in tokens][:max_len]
                    ids += [0] * (max_len - len(ids))
                    self.data.append((torch.tensor(ids), label2id[label]))

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return self.data[idx]

        train_ds = CNDataset(train_dataset.texts, train_dataset.labels, vocab, max_len, self.label2id)
        val_ds = CNDataset(val_dataset.texts, val_dataset.labels, vocab, max_len, self.label2id)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        train_losses, val_losses, val_metrics = [], [], []
        best_val_metric, best_epoch, no_improve, best_state = 0, 0, 0, None

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_train_loss = total_loss / max(len(train_loader), 1)
            train_losses.append(avg_train_loss)

            model.eval()
            val_loss, correct, total = 0, 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    val_loss += criterion(outputs, labels).item()
                    preds = torch.argmax(outputs, dim=-1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            avg_val_loss = val_loss / max(len(val_loader), 1)
            val_acc = correct / max(total, 1)
            val_losses.append(avg_val_loss)
            val_metrics.append(val_acc)

            if val_acc > best_val_metric:
                best_val_metric = val_acc
                best_epoch = epoch
                no_improve = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1

            if progress_callback:
                await progress_callback(epoch + 1, epochs, avg_train_loss, avg_val_loss, val_acc)

            if no_improve >= patience:
                break
            await asyncio.sleep(0)

        if best_state:
            model.load_state_dict(best_state)

        save_path = MODEL_CACHE_DIR / f"textcnn_{int(time.time())}"
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": best_state or model.state_dict(), "vocab": vocab, "config": {
            "embed_dim": embed_dim, "num_classes": self.num_classes,
            "num_filters": num_filters, "filter_sizes": filter_sizes,
        }}, save_path / "model.pt")

        return {
            "model_path": str(save_path),
            "train_loss_history": train_losses,
            "val_loss_history": val_losses,
            "val_metric_history": val_metrics,
            "best_epoch": best_epoch,
            "best_val_metric": best_val_metric,
            "total_epochs": epoch + 1,
        }

    async def evaluate(self, test_dataset: TextDataset) -> dict:
        import torch
        from sklearn.metrics import accuracy_score, f1_score, classification_report

        save_path = self.hyperparams.get("_model_path", "")
        if not save_path or not Path(save_path).exists():
            return {"error": "Model not found"}

        checkpoint = torch.load(Path(save_path) / "model.pt", map_location="cpu")
        vocab = checkpoint["vocab"]
        from .training import TextCNNTrainer as _T
        max_len = self.hyperparams.get("max_seq_length", 128)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        import torch.nn as nn
        cfg = checkpoint["config"]

        class TextCNNModel(nn.Module):
            def __init__(self, vocab_size, embed_dim, num_classes, num_filters, filter_sizes):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
                self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, k) for k in filter_sizes])
                self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
                self.dropout = nn.Dropout(0.3)
            def forward(self, x):
                emb = self.embedding(x).permute(0, 2, 1)
                conv_outs = [torch.relu(c(emb)) for c in self.convs]
                pooled = [torch.max(c, dim=2)[0] for c in conv_outs]
                return self.fc(self.dropout(torch.cat(pooled, dim=1)))

        model = TextCNNModel(len(vocab), cfg["embed_dim"], cfg["num_classes"], cfg["num_filters"], cfg["filter_sizes"]).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        all_preds, all_labels = [], []
        for i in range(0, len(test_dataset), 32):
            batch_texts = test_dataset.texts[i:i+32]
            batch_labels = test_dataset.labels[i:i+32]
            batch_ids = []
            for text in batch_texts:
                tokens = text.lower().split()
                ids = [vocab.get(t, 1) for t in tokens][:max_len]
                ids += [0] * (max_len - len(ids))
                batch_ids.append(ids)
            with torch.no_grad():
                input_tensor = torch.tensor(batch_ids).to(device)
                preds = torch.argmax(model(input_tensor), dim=-1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend([self.label2id[l] for l in batch_labels])

        y_true, y_pred = np.array(all_labels), np.array(all_preds)
        accuracy = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        per_class = {}
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        for k, v in report.items():
            if k in ("accuracy", "macro avg", "weighted avg"):
                continue
            label_name = self.id2label.get(int(k), k)
            per_class[label_name] = {"precision": v["precision"], "recall": v["recall"], "f1-score": v["f1-score"], "support": v["support"]}

        return {"accuracy": accuracy, "macro_f1": macro_f1, "weighted_f1": weighted_f1, "per_class_metrics": per_class}

    async def predict(self, texts: list[str]) -> tuple[list[int], list[float]]:
        import torch
        save_path = self.hyperparams.get("_model_path", "")
        if not save_path or not Path(save_path).exists():
            return [], []
        checkpoint = torch.load(Path(save_path) / "model.pt", map_location="cpu")
        vocab = checkpoint["vocab"]
        max_len = self.hyperparams.get("max_seq_length", 128)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        import torch.nn as nn
        cfg = checkpoint["config"]
        class TextCNNModel(nn.Module):
            def __init__(self, vs, ed, nc, nf, fs):
                super().__init__()
                self.embedding = nn.Embedding(vs, ed, padding_idx=0)
                self.convs = nn.ModuleList([nn.Conv1d(ed, nf, k) for k in fs])
                self.fc = nn.Linear(nf * len(fs), nc)
                self.dropout = nn.Dropout(0.3)
            def forward(self, x):
                emb = self.embedding(x).permute(0, 2, 1)
                return self.fc(self.dropout(torch.cat([torch.max(torch.relu(c(emb)), dim=2)[0] for c in self.convs], dim=1)))
        model = TextCNNModel(len(vocab), cfg["embed_dim"], cfg["num_classes"], cfg["num_filters"], cfg["filter_sizes"]).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        batch_ids = []
        for text in texts:
            tokens = text.lower().split()
            ids = [vocab.get(t, 1) for t in tokens][:max_len]
            ids += [0] * (max_len - len(ids))
            batch_ids.append(ids)
        with torch.no_grad():
            logits = model(torch.tensor(batch_ids).to(device))
            probs = torch.softmax(logits, dim=-1)
            conf, preds = torch.max(probs, dim=-1)
        return preds.cpu().numpy().tolist(), conf.cpu().numpy().tolist()

    @staticmethod
    def _build_vocab(texts: list[str], max_size: int = 30000) -> dict:
        counter = Counter()
        for text in texts:
            for word in text.lower().split():
                counter[word] += 1
        vocab = {"<pad>": 0, "<unk>": 1}
        for i, (word, _) in enumerate(counter.most_common(max_size - 2)):
            vocab[word] = i + 2
        return vocab


class BiLSTMAttentionTrainer(BaseModelTrainer):
    async def train(self, train_dataset: TextDataset, val_dataset: TextDataset, progress_callback=None) -> dict:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset as TorchDataset

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vocab = TextCNNTrainer._build_vocab(train_dataset.texts)
        embed_dim = 128
        hidden_dim = 128
        max_len = self.hyperparams.get("max_seq_length", 128)
        batch_size = self.hyperparams.get("batch_size", 16)
        epochs = self.hyperparams.get("epochs", 10)
        lr = self.hyperparams.get("learning_rate", 1e-3)
        patience = self.hyperparams.get("early_stopping_patience", 3)

        class BiLSTMAttention(nn.Module):
            def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
                self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
                self.attention = nn.Linear(hidden_dim * 2, 1)
                self.fc = nn.Linear(hidden_dim * 2, num_classes)
                self.dropout = nn.Dropout(0.3)

            def forward(self, x):
                emb = self.embedding(x)
                lstm_out, _ = self.lstm(emb)
                attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
                context = torch.sum(attn_weights * lstm_out, dim=1)
                return self.fc(self.dropout(context))

        model = BiLSTMAttention(len(vocab), embed_dim, hidden_dim, self.num_classes).to(device)

        class SeqDataset(TorchDataset):
            def __init__(self, texts, labels, vocab, max_len, label2id):
                self.data = []
                for text, label in zip(texts, labels):
                    tokens = text.lower().split()
                    ids = [vocab.get(t, 1) for t in tokens][:max_len]
                    ids += [0] * (max_len - len(ids))
                    self.data.append((torch.tensor(ids), label2id[label]))
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                return self.data[idx]

        train_ds = SeqDataset(train_dataset.texts, train_dataset.labels, vocab, max_len, self.label2id)
        val_ds = SeqDataset(val_dataset.texts, val_dataset.labels, vocab, max_len, self.label2id)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        train_losses, val_losses, val_metrics = [], [], []
        best_val_metric, best_epoch, no_improve, best_state = 0, 0, 0, None

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = criterion(model(inputs), labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            train_losses.append(total_loss / max(len(train_loader), 1))

            model.eval()
            val_loss, correct, total = 0, 0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    out = model(inputs)
                    val_loss += criterion(out, labels).item()
                    correct += (torch.argmax(out, dim=-1) == labels).sum().item()
                    total += labels.size(0)
            val_acc = correct / max(total, 1)
            val_losses.append(val_loss / max(len(val_loader), 1))
            val_metrics.append(val_acc)

            if val_acc > best_val_metric:
                best_val_metric = val_acc
                best_epoch = epoch
                no_improve = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1

            if progress_callback:
                await progress_callback(epoch + 1, epochs, train_losses[-1], val_losses[-1], val_acc)
            if no_improve >= patience:
                break
            await asyncio.sleep(0)

        if best_state:
            model.load_state_dict(best_state)

        save_path = MODEL_CACHE_DIR / f"bilstm_{int(time.time())}"
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": best_state or model.state_dict(), "vocab": vocab, "config": {
            "embed_dim": embed_dim, "hidden_dim": hidden_dim, "num_classes": self.num_classes,
        }}, save_path / "model.pt")

        return {
            "model_path": str(save_path), "train_loss_history": train_losses,
            "val_loss_history": val_losses, "val_metric_history": val_metrics,
            "best_epoch": best_epoch, "best_val_metric": best_val_metric,
            "total_epochs": epoch + 1,
        }

    async def evaluate(self, test_dataset: TextDataset) -> dict:
        import torch
        from sklearn.metrics import accuracy_score, f1_score, classification_report
        save_path = self.hyperparams.get("_model_path", "")
        if not save_path or not Path(save_path).exists():
            return {"error": "Model not found"}
        checkpoint = torch.load(Path(save_path) / "model.pt", map_location="cpu")
        vocab = checkpoint["vocab"]
        max_len = self.hyperparams.get("max_seq_length", 128)
        cfg = checkpoint["config"]
        import torch.nn as nn
        class BiLSTMAttention(nn.Module):
            def __init__(self, vs, ed, hd, nc):
                super().__init__()
                self.embedding = nn.Embedding(vs, ed, padding_idx=0)
                self.lstm = nn.LSTM(ed, hd, bidirectional=True, batch_first=True)
                self.attention = nn.Linear(hd * 2, 1)
                self.fc = nn.Linear(hd * 2, nc)
                self.dropout = nn.Dropout(0.3)
            def forward(self, x):
                emb = self.embedding(x)
                out, _ = self.lstm(emb)
                w = torch.softmax(self.attention(out), dim=1)
                return self.fc(self.dropout(torch.sum(w * out, dim=1)))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BiLSTMAttention(len(vocab), cfg["embed_dim"], cfg["hidden_dim"], cfg["num_classes"]).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        all_preds, all_labels = [], []
        for i in range(0, len(test_dataset), 32):
            batch_texts = test_dataset.texts[i:i+32]
            batch_labels = test_dataset.labels[i:i+32]
            batch_ids = []
            for text in batch_texts:
                tokens = text.lower().split()
                ids = [vocab.get(t, 1) for t in tokens][:max_len]
                ids += [0] * (max_len - len(ids))
                batch_ids.append(ids)
            with torch.no_grad():
                preds = torch.argmax(model(torch.tensor(batch_ids).to(device)), dim=-1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend([self.label2id[l] for l in batch_labels])
        y_true, y_pred = np.array(all_labels), np.array(all_preds)
        accuracy = float(accuracy_score(y_true, y_pred))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        per_class = {}
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        for k, v in report.items():
            if k in ("accuracy", "macro avg", "weighted avg"):
                continue
            per_class[self.id2label.get(int(k), k)] = {"precision": v["precision"], "recall": v["recall"], "f1-score": v["f1-score"], "support": v["support"]}
        return {"accuracy": accuracy, "macro_f1": macro_f1, "weighted_f1": weighted_f1, "per_class_metrics": per_class}

    async def predict(self, texts: list[str]) -> tuple[list[int], list[float]]:
        import torch
        save_path = self.hyperparams.get("_model_path", "")
        if not save_path or not Path(save_path).exists():
            return [], []
        checkpoint = torch.load(Path(save_path) / "model.pt", map_location="cpu")
        vocab = checkpoint["vocab"]
        max_len = self.hyperparams.get("max_seq_length", 128)
        cfg = checkpoint["config"]
        import torch.nn as nn
        class BiLSTMAttention(nn.Module):
            def __init__(self, vs, ed, hd, nc):
                super().__init__()
                self.embedding = nn.Embedding(vs, ed, padding_idx=0)
                self.lstm = nn.LSTM(ed, hd, bidirectional=True, batch_first=True)
                self.attention = nn.Linear(hd * 2, 1)
                self.fc = nn.Linear(hd * 2, nc)
                self.dropout = nn.Dropout(0.3)
            def forward(self, x):
                emb = self.embedding(x)
                out, _ = self.lstm(emb)
                w = torch.softmax(self.attention(out), dim=1)
                return self.fc(self.dropout(torch.sum(w * out, dim=1)))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BiLSTMAttention(len(vocab), cfg["embed_dim"], cfg["hidden_dim"], cfg["num_classes"]).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()
        batch_ids = []
        for text in texts:
            tokens = text.lower().split()
            ids = [vocab.get(t, 1) for t in tokens][:max_len]
            ids += [0] * (max_len - len(ids))
            batch_ids.append(ids)
        with torch.no_grad():
            logits = model(torch.tensor(batch_ids).to(device))
            probs = torch.softmax(logits, dim=-1)
            conf, preds = torch.max(probs, dim=-1)
        return preds.cpu().numpy().tolist(), conf.cpu().numpy().tolist()


TRAINER_MAP = {
    ModelBackbone.distilbert: DistilBertTrainer,
    ModelBackbone.tinybert: DistilBertTrainer,
    ModelBackbone.textcnn: TextCNNTrainer,
    ModelBackbone.bilstm_attention: BiLSTMAttentionTrainer,
}


def _oversample_minority(texts: list[str], labels: list[str], min_count: int = MIN_SAMPLES_PER_CLASS) -> tuple[list[str], list[str]]:
    class_counts = Counter(labels)
    new_texts, new_labels = list(texts), list(labels)
    for label, count in class_counts.items():
        if count < min_count:
            indices = [i for i, l in enumerate(labels) if l == label]
            needed = min_count - count
            sampled = random.choices(indices, k=needed)
            for idx in sampled:
                new_texts.append(texts[idx])
                new_labels.append(labels[idx])
    return new_texts, new_labels



async def execute_training(
    session: AsyncSession,
    experiment_id: int,
) -> None:
    stmt = select(TrainingExperiment).where(TrainingExperiment.id == experiment_id)
    result = await session.execute(stmt)
    experiment = result.scalar_one_or_none()
    if not experiment:
        return

    experiment.status = TaskStatus.running
    experiment.started_at = __import__("datetime").datetime.utcnow()
    await session.commit()

    try:
        version_stmt = select(DatasetVersion).where(DatasetVersion.id == experiment.version_id)
        version = (await session.execute(version_stmt)).scalar_one()
        dataset_stmt = select(Dataset).where(Dataset.id == version.dataset_id)
        dataset = (await session.execute(dataset_stmt)).scalar_one()

        sample_stmt = select(Sample).where(
            Sample.version_id == experiment.version_id,
            Sample.is_filtered == False,
        )
        samples = (await session.execute(sample_stmt)).scalars().all()

        train_samples = [s for s in samples if s.split == SplitType.train]
        val_samples = [s for s in samples if s.split == SplitType.val]
        test_samples = [s for s in samples if s.split == SplitType.test]

        if not train_samples or not val_samples:
            raise ValueError("Insufficient train/val samples")

        all_labels = sorted(set(s.label for s in samples))
        label2id = {l: i for i, l in enumerate(all_labels)}
        id2label = {i: l for l, i in label2id.items()}

        train_texts = [s.text for s in train_samples]
        train_labels = [s.label for s in train_samples]
        val_texts = [s.text for s in val_samples]
        val_labels = [s.label for s in val_samples]

        train_texts, train_labels = _oversample_minority(train_texts, train_labels)

        hyperparams = dict(experiment.hyperparams)
        trainer_cls = TRAINER_MAP.get(experiment.backbone, DistilBertTrainer)
        trainer = trainer_cls(
            num_classes=len(all_labels),
            label2id=label2id,
            id2label=id2label,
            hyperparams=hyperparams,
        )

        train_ds = TextDataset(train_texts, train_labels, label2id)
        val_ds = TextDataset(val_texts, val_labels, label2id)

        if experiment.training_mode == TrainingMode.baseline:
            result = await trainer.train(train_ds, val_ds)

        elif experiment.training_mode == TrainingMode.augmented:
            multiplier = experiment.augmentation_multiplier
            if multiplier != 1.0:
                aug_samples = [s for s in train_samples if s.source != SampleSource.original]
                orig_samples = [s for s in train_samples if s.source == SampleSource.original]
                target_aug_count = int(len(orig_samples) * multiplier)
                if aug_samples and target_aug_count > 0:
                    sampled_aug = random.choices(aug_samples, k=min(target_aug_count, len(aug_samples) * 5))
                    train_texts = [s.text for s in orig_samples] + [s.text for s in sampled_aug[:target_aug_count]]
                    train_labels = [s.label for s in orig_samples] + [s.label for s in sampled_aug[:target_aug_count]]
                    train_texts, train_labels = _oversample_minority(train_texts, train_labels)
                    train_ds = TextDataset(train_texts, train_labels, label2id)
            result = await trainer.train(train_ds, val_ds)

        elif experiment.training_mode == TrainingMode.curriculum:
            aug_with_ppl = [(s, s.perplexity or 0) for s in train_samples if s.source != SampleSource.original]
            aug_with_ppl.sort(key=lambda x: x[1])
            thirds = len(aug_with_ppl) // 3
            batches = [
                aug_with_ppl[:thirds],
                aug_with_ppl[thirds:2*thirds],
                aug_with_ppl[2*thirds:],
            ]
            orig_train = TextDataset(
                [s.text for s in train_samples if s.source == SampleSource.original],
                [s.label for s in train_samples if s.source == SampleSource.original],
                label2id,
            )
            orig_train_texts, orig_train_labels = list(orig_train.texts), list(orig_train.labels)

            result = None
            for batch_idx, batch in enumerate(batches):
                batch_texts = [s.text for s, _ in batch]
                batch_labels = [s.label for s, _ in batch]
                combined_texts = orig_train_texts + batch_texts
                combined_labels = orig_train_labels + batch_labels
                combined_texts, combined_labels = _oversample_minority(combined_texts, combined_labels)
                combined_ds = TextDataset(combined_texts, combined_labels, label2id)
                result = await trainer.train(combined_ds, val_ds)
                if result:
                    trainer.hyperparams["_model_path"] = result["model_path"]
                    orig_train_texts = combined_texts
                    orig_train_labels = combined_labels

        elif experiment.training_mode == TrainingMode.semi_supervised:
            result = await trainer.train(train_ds, val_ds)
            if result:
                trainer.hyperparams["_model_path"] = result["model_path"]

            unlabeled_stmt = select(Sample).where(
                Sample.version_id == experiment.version_id,
                Sample.source == SampleSource.original,
                Sample.is_filtered == False,
            )
            unlabeled_samples = (await session.execute(unlabeled_stmt)).scalars().all()

            best_val_metric = result.get("best_val_metric", 0) if result else 0
            no_improve_rounds = 0

            for iteration in range(MAX_SELF_TRAINING_ITERATIONS):
                if not unlabeled_samples:
                    break

                pseudo_texts = [s.text for s in unlabeled_samples]
                pred_ids, confidences = await trainer.predict(pseudo_texts)
                high_conf_indices = [i for i, c in enumerate(confidences) if c > 0.9]
                if not high_conf_indices:
                    break

                pseudo_texts_filtered = [pseudo_texts[i] for i in high_conf_indices]
                pseudo_labels = [id2label.get(pred_ids[i], all_labels[0]) for i in high_conf_indices]

                new_train_texts = train_texts + pseudo_texts_filtered
                new_train_labels = train_labels + pseudo_labels
                new_train_texts, new_train_labels = _oversample_minority(new_train_texts, new_train_labels)
                new_train_ds = TextDataset(new_train_texts, new_train_labels, label2id)

                result = await trainer.train(new_train_ds, val_ds)
                if result:
                    trainer.hyperparams["_model_path"] = result["model_path"]
                    current_val = result.get("best_val_metric", 0)
                    if current_val > best_val_metric:
                        best_val_metric = current_val
                        no_improve_rounds = 0
                    else:
                        no_improve_rounds += 1

                if no_improve_rounds >= SELF_TRAINING_EARLY_STOP_PATIENCE:
                    break

                unlabeled_samples = [s for i, s in enumerate(unlabeled_samples) if i not in high_conf_indices]

        else:
            result = await trainer.train(train_ds, val_ds)

        if result:
            experiment.train_loss_history = result.get("train_loss_history", [])
            experiment.val_loss_history = result.get("val_loss_history", [])
            experiment.val_metric_history = result.get("val_metric_history", [])
            experiment.best_epoch = result.get("best_epoch")
            experiment.best_val_metric = result.get("best_val_metric")
            experiment.model_path = result.get("model_path")
            experiment.current_epoch = result.get("total_epochs", 0)

            if result.get("model_path"):
                trainer.hyperparams["_model_path"] = result["model_path"]
                if test_samples:
                    test_ds = TextDataset(
                        [s.text for s in test_samples],
                        [s.label for s in test_samples],
                        label2id,
                    )
                    eval_result = await trainer.evaluate(test_ds)
                    eval_row = EvaluationResult(
                        experiment_id=experiment.id,
                        accuracy=eval_result.get("accuracy", 0),
                        macro_f1=eval_result.get("macro_f1", 0),
                        weighted_f1=eval_result.get("weighted_f1", 0),
                        per_class_metrics=eval_result.get("per_class_metrics", {}),
                    )
                    session.add(eval_row)

        experiment.status = TaskStatus.completed
        experiment.completed_at = __import__("datetime").datetime.utcnow()
        await session.commit()

    except Exception as e:
        logger.exception(f"Training experiment {experiment_id} failed")
        experiment.status = TaskStatus.failed
        experiment.error_message = str(e)
        await session.commit()
