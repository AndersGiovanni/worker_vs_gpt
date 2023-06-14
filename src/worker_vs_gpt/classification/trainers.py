from typing import Dict, Optional, List, NamedTuple

import datasets
import numpy as np
import pandas as pd
import torch
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

from worker_vs_gpt.config import MODELS_DIR, TrainerConfig
from worker_vs_gpt.utils import get_device
from worker_vs_gpt.classification.custom_callbacks import CustomWandbCallback
from worker_vs_gpt.data_processing.dataclass import DataClassWorkerVsGPT


class PredictionOutput(NamedTuple):
    """Prediction output."""

    predictions: np.ndarray
    label_ids: Optional[np.ndarray]
    metrics: Optional[Dict[str, float]]


class ExperimentTrainer:
    """Experiment Trainer."""

    def __init__(
        self,
        data: DataClassWorkerVsGPT,
        config: TrainerConfig,
    ) -> None:
        self.config: TrainerConfig = config
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.ckpt)
        self.dataset: datasets.dataset_dict.DatasetDict = data.get_data()
        self.device: torch.device = get_device()
        self.num_labels: int = len(data.labels)
        self.labels: List[str] = data.labels
        self.dataset_name: str = config.dataset

        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.ckpt,
            num_labels=self.num_labels,
        ).to(self.device)

    def _compute_metrics(
        self, predictions: np.ndarray, labels: np.ndarray, threshold: float = 0.5
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
        # first, apply softmax on predictions which are of shape (batch_size, num_labels)
        softmax = torch.nn.Softmax(dim=1)
        probs = softmax(torch.Tensor(predictions))
        loss_fn = torch.nn.CrossEntropyLoss()

        # get the index of the entry with the highest probability in each row
        max_indices = np.argmax(probs, axis=1)

        # create a new matrix with 1 on the max probability entry in each row and 0 elsewhere
        y_pred = np.zeros_like(probs)
        y_pred[np.arange(len(probs)), max_indices] = 1

        # finally, compute metrics
        y_true = labels
        loss = loss_fn(torch.from_numpy(y_pred), torch.from_numpy(y_true))
        f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average="macro")
        roc_auc = roc_auc_score(y_true, probs, average="macro")
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
        Compute metrics for classification (F1, AUC, ACC, loss).

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
        result: Dict[str, float] = self._compute_metrics(
            predictions=preds, labels=p.label_ids
        )
        return result

    def train(self) -> None:
        """Train model. We save the model in the MODELS_DIR directory and log the results to wandb."""
        wandb.init(
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            # name=f"{self.config.ckpt}_{self.config.sampling}_{self.config.augmentation_model}_combined", Used for combined training
            name=f"{self.config.ckpt}_size:{self.dataset['train'].num_rows}",  # Used for data size experiment
            group=f"{self.config.dataset}",
            config={
                "ckpt": self.config.ckpt,
                "batch_size": self.config.batch_size,
                "lr": self.config.lr,
                "num_epochs": self.config.num_epochs,
                "weight_decay": self.config.weight_decay,
                "train_size": self.dataset["train"].num_rows,
                "use_augmented_data": self.config.use_augmented_data,
                "sampling": self.config.sampling,
                "augmentation_model": self.config.augmentation_model,
            },
        )

        args = TrainingArguments(
            str(
                MODELS_DIR
                / f"{self.config.dataset}_size:{len(self.dataset['train'])}_{self.config.ckpt}"
            ),
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            learning_rate=self.config.lr,
            load_best_model_at_end=True,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_epochs,
            weight_decay=self.config.weight_decay,
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

    def test(
        self,
    ) -> PredictionOutput:
        """Test the trained model on the test set and log to wandb."""
        prediction_output: PredictionOutput = self.trainer.predict(
            self.dataset["test"], metric_key_prefix=""
        )

        metrics, y_pred_logits, y_true = (
            prediction_output.metrics,
            prediction_output.predictions,
            prediction_output.label_ids,
        )

        # Convert logits to probabilities
        softmax = torch.nn.Softmax()
        probs = softmax(torch.Tensor(y_pred_logits)).numpy()

        # get the index of the entry with the highest probability in each row
        max_indices = np.argmax(probs, axis=1)

        # create a new matrix with 1 on the max probability entry in each row and 0 elsewhere
        y_pred = np.zeros_like(probs)
        y_pred[np.arange(len(probs)), max_indices] = 1

        clf_report = classification_report(
            y_true=y_true, y_pred=y_pred, target_names=self.labels, output_dict=True
        )

        # Add prefix to metrics "test/"
        metrics = {f"test/{k[1:]}": v for k, v in metrics.items()}
        # Log results
        wandb.log(
            metrics,
        )

        df = pd.DataFrame(clf_report)
        df["metric"] = df.index
        table = wandb.Table(data=df)

        wandb.log(
            {
                "classification_report": table,
            }
        )

        return prediction_output

    def predict(self, x: torch.Tensor) -> PredictionOutput:
        """Predict"""
        return self.trainer.predict(x)
