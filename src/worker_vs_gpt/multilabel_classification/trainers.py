from typing import Dict, Tuple, List, Union

import datasets
import numpy as np
import torch
from sentence_transformers.losses import CosineSimilarityLoss
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)
import wandb

from worker_vs_gpt.config import MODELS_DIR, MultilabelParams
from worker_vs_gpt.utils import get_device
from worker_vs_gpt.multilabel_classification.custom_callbacks import CustomWandbCallback


class MultiLabel:
    """Multi-label classification model."""

    def __init__(
        self, dataset: datasets.dataset_dict.DatasetDict, config: MultilabelParams
    ) -> None:
        self.config: MultilabelParams = config
        self.ckpt: str = config.ckpt
        self.batch_size: int = config.batch_size
        self.lr: float = config.lr
        self.num_epochs: int = config.num_epochs
        self.weight_decay: float = config.weight_decay
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.ckpt, problem_type="multi_label_classification"
        )
        self.dataset: datasets.dataset_dict.DatasetDict = dataset
        self.device: torch.device = get_device()
        self.num_labels: int = dataset["train"]["labels"].size()[1]

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.ckpt,
            problem_type="multi_label_classification",
            num_labels=self.num_labels,
        ).to(self.device)

    def multi_label_metrics(
        self, predictions: torch.tensor, labels: torch.tensor, threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Compute metrics for multi-label classification (F1, AUC, ACC).

        Parameters
        ----------
        predictions : torch.tensor
            Predictions from model.

        labels : torch.tensor
            True labels.

        threshold : float, optional
            Threshold for turning probabilities into integer predictions, by default 0.5

        Returns
        -------
        Dict[str, float]
            Dictionary with metrics.
        """
        # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))
        loss_fn = torch.nn.BCEWithLogitsLoss()
        # next, use threshold to turn them into integer predictions
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= threshold)] = 1
        # finally, compute metrics
        y_true = labels
        loss = loss_fn(torch.from_numpy(y_pred), torch.from_numpy(y_true))
        f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
        roc_auc = roc_auc_score(y_true, probs, average="micro")
        accuracy = accuracy_score(y_true, y_pred)
        # return as dictionary
        metrics = {
            "f1": f1_micro_average,
            "roc_auc": roc_auc,
            "accuracy": accuracy,
            "loss": loss,
        }
        return metrics

    def compute_metrics(self, p: EvalPrediction) -> Dict[str, float]:
        """
        Compute metrics for multi-label classification (F1, AUC, ACC).

        Parameters
        ----------
        p : EvalPrediction
            Evaluation prediction.

        Returns
        -------
        Dict[str, float]
            Dictionary with metrics.
        """
        preds: torch.tensor = (
            p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        )
        result: Dict[str, float] = self.multi_label_metrics(
            predictions=preds, labels=p.label_ids
        )
        return result

    def train(self) -> None:
        wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            name=f"MultiLabel_{self.ckpt}_test",
            group="multilabel",
        )

        args = TrainingArguments(
            str(MODELS_DIR / f"MultiLabel_{self.ckpt}"),
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            learning_rate=self.lr,
            load_best_model_at_end=True,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.num_epochs,
            weight_decay=self.weight_decay,
            metric_for_best_model="eval_loss",
            push_to_hub=False,
            seed=42,
        )
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"],
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[CustomWandbCallback()],
        )
        trainer.train()

        print(trainer.evaluate())
        self.trainer = trainer

    def test(self) -> None:
        prediction_output: Tuple[
            np.ndarray, np.ndarray, Dict[str, float]
        ] = self.trainer.predict(self.dataset["test"], metric_key_prefix="")

        return prediction_output  # type: ignore

    def predict(self, x: torch.Tensor) -> np.ndarray:
        return self.trainer.predict(x)
