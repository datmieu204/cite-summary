from __future__ import annotations

import json
import logging
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

import joblib  # type: ignore
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

__all__ = ["BaseModel", "SciSummClassifier"]

# ---------------------------------------------------------------------------
# 1) Abstract base class – unchanged skeleton
# ---------------------------------------------------------------------------

class BaseModel(metaclass=ABCMeta):
    """Unified interface for all downstream models."""

    def __init__(
        self,
        label_col: str | None = "label",
        id_col: str | None = None,
        seed: int = 42,
        **kwargs: Any,
    ) -> None:
        self.label_col = label_col
        self.id_col = id_col
        self.seed = seed
        self._extra_cfg: Dict[str, Any] = kwargs
        self._model: Any | None = None
        set_seed(seed)

    @abstractmethod
    def train(self, *args: Any, **kwargs: Any) -> None: ...

    @abstractmethod
    def predict(self, samples: Union[pd.DataFrame, Dataset]) -> np.ndarray: ...

    def evaluate(self, samples: Union[pd.DataFrame, Dataset]) -> Dict[str, float]:
        if self.label_col is None:
            raise ValueError("Cannot evaluate when 'label_col' is None")
        if not isinstance(samples, pd.DataFrame):
            samples = pd.DataFrame(samples)
        y_true = samples[self.label_col].to_numpy()
        y_pred = self.predict(samples)
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        }

    # --------------------------------- persistence helpers --------------------------------
    def save(self, path: Union[str, Path]) -> None:
        if self._model is None:
            raise RuntimeError("Model not initialised; call 'train' first")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"cfg": self._extra_cfg, "hf_model": self._model}, path)
        LOGGER.info("Saved model artefact to %s", path)

    @classmethod
    def load(cls, path: Union[str, Path], **kwargs: Any) -> "BaseModel":
        bundle = joblib.load(path)
        instance: "BaseModel" = cls.__new__(cls)  # type: ignore[arg-type]
        super(BaseModel, instance).__init__(**kwargs)
        instance._model = bundle["hf_model"]
        instance._extra_cfg = bundle["cfg"]
        return instance

# ---------------------------------------------------------------------------
# 2) Concrete implementation – BERT + weighted cross‑entropy
# ---------------------------------------------------------------------------

class _WeightedLossTrainer(Trainer):
    """Hugging Face *Trainer* subclass with class‑weight support."""

    def __init__(self, class_weights: torch.Tensor, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    def compute_loss(self, model, inputs, return_outputs=False):  # type: ignore[override]
        outputs = model(**inputs)
        logits = outputs.get("logits")
        labels = inputs.get("labels")
        loss = self.loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
        "f1": f1_score(labels, preds, average="macro", zero_division=0),
    }


class SciSummClassifier(BaseModel):
    """Sequence‑classification model exactly as trained in *trainscisumm.ipynb*."""

    DEFAULT_CHECKPOINT = "google-bert/bert-base-uncased"

    def __init__(
        self,
        label2id: Dict[str, int] | None = None,
        checkpoint: str = DEFAULT_CHECKPOINT,
        max_length: int = 128,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if label2id is None:
            label2id = {"aim": 0, "hypothesis": 1, "implication": 2, "method": 3, "result": 4}
        self.label2id = label2id
        self.id2label = {v: k for k, v in label2id.items()}
        self.num_labels = len(label2id)

        # Hugging Face artefacts
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )
        self.max_length = max_length

    # ----------------------------------------- helpers ------------------------------------
    def _tokenize(self, examples: Dict[str, List[str]]):
        return self.tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )

    # -------------------------------------- API ------------------------------------------
    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
        epochs: int = 5,
        lr: float = 2e-5,
        weight_decay: float = 0.01,
        output_dir: str = "Label-classifier",
        batch_size: int = 8,
        push_to_hub: bool = False,
        **train_kwargs: Any,
    ) -> None:
        """Fine‑tune the BERT model with weighted loss exactly like the notebook."""
        if self.label_col is None:
            raise ValueError("'label_col' must be set for training")

        # ---------------------------------------------------------------- Data → HF Dataset
        train_ds = Dataset.from_pandas(train_df)
        val_ds = Dataset.from_pandas(val_df) if val_df is not None else None

        # Tokenise
        train_ds = train_ds.map(self._tokenize, batched=True, remove_columns=train_ds.column_names)
        if val_ds is not None:
            val_ds = val_ds.map(self._tokenize, batched=True, remove_columns=val_ds.column_names)

        # Class weights from label distribution
        label_counts = np.bincount(train_df[self.label_col], minlength=self.num_labels)
        class_weights = torch.tensor(label_counts.max() / (label_counts + 1e-9), dtype=torch.float32)

        # ---------------------------------------------------------------- Trainer
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=lr,
            weight_decay=weight_decay,
            evaluation_strategy="epoch" if val_ds is not None else "no",
            save_strategy="no",
            push_to_hub=push_to_hub,
            logging_steps=50,
        )

        trainer = _WeightedLossTrainer(
            model=self._model,
            class_weights=class_weights,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=self.tokenizer,
            compute_metrics=_compute_metrics if val_ds is not None else None,
        )

        LOGGER.info("Starting fine‑tuning – %d steps/epoch * %d epochs", len(train_ds) // batch_size, epochs)
        trainer.train()
        LOGGER.info("Finished training; best f1: %s", trainer.state.best_metric)

    # ---------------------------------------------------------------- predict / infer ----
    def predict(self, samples: Union[pd.DataFrame, Dataset]) -> np.ndarray:  # type: ignore[override]
        if isinstance(samples, pd.DataFrame):
            data = Dataset.from_pandas(samples)
        else:  
            data = samples
        data = data.map(self._tokenize, batched=True, remove_columns=data.column_names)
        logits = self._model(**{k: torch.tensor(v) for k, v in data[:].items() if k != "label"}).logits
        return np.argmax(logits.detach().cpu().numpy(), axis=-1)


# ---------------------------------------------------------------------------
# 3) CLI usage example -------------------------------------------------------
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Fine‑tune SciSumm label classifier")
    ap.add_argument("--train", required=True, help="Path to training CSV/JSON lines")
    ap.add_argument("--val", help="Optional validation CSV/JSON lines")
    ap.add_argument("--text-col", default="refer_text", help="Name of text column in CSV")
    ap.add_argument("--label-col", default="label", help="Name of label column in CSV")
    args = ap.parse_args()

    # ------------------------------- load CSVs
    train_df = pd.read_csv(args.train)
    if args.val:
        val_df = pd.read_csv(args.val)
    else:
        val_df = None
    train_df = train_df.rename(columns={args.text_col: "text", args.label_col: "label"})
    if val_df is not None:
        val_df = val_df.rename(columns={args.text_col: "text", args.label_col: "label"})

    # ------------------------------- train
    model = SciSummClassifier(label_col="label")
    model.train(train_df, val_df)
    # save artifact
    model.save(Path("label_classifier.joblib"))
