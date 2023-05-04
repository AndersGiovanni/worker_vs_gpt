from dataclasses import dataclass
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

from setfit import SetFitModel, SetFitTrainer
from sentence_transformers.losses import CosineSimilarityLoss

from worker_vs_gpt.config import SetfitParams


class PredictionOutput(NamedTuple):
    """Prediction output."""

    predictions: np.ndarray
    label_ids: Optional[np.ndarray]
    metrics: Optional[Dict[str, float]]


class SetFitClassificationTrainer:
    """Multi-label classification model using SetFit.

    Parameters
    ----------
    dataset : datasets.dataset_dict.DatasetDict
        Dataset.

    config : ten_social_dim.config.SetfitParams
        SetFit parameters.

    Attributes
    ----------
    tokenizer : transformers.tokenization_utils_base.PreTrainedTokenizerBase
        Tokenizer.


    ckpt : str
        Model id to use.

    dataset : datasets.dataset_dict.DatasetDict
        Dataset.

    device : torch.device
        Device to use.

    num_labels : int
        Number of labels.

    model : setfit.model.SetFitModel
        SetFit model.

    trainer : transformers.trainer.Trainer
        Trainer.

    """

    def __init__(
        self, dataset: datasets.dataset_dict.DatasetDict, config: SetfitParams
    ) -> None:
        self.config: SetfitParams = config
        self.ckpt = config.ckpt
        self.dataset = dataset
        self.device = get_device()
        self.text_selection = config.text_selection
        self.labels: List[str] = [
            "social_support",
            "conflict",
            "trust",
            "neutral",
            "fun",
            "respect",
            "knowledge",
            "power",
            "similarity_identity",
        ]
        self.num_classes = len(self.labels)
        self.model = SetFitModel.from_pretrained(
            self.ckpt,
            use_differentiable_head=True,
            head_params={"out_features": self.num_classes},
        ).to(self.device)

    def train(self, evaluate_test_set: bool = True) -> None:
        """Trainer for SetFit model.

        Parameters
        ----------
        batch_size : int, optional
            Batch size, by default 4
        lr : _type_, optional
            Learning rate, by default 2e-5
        num_iterations : int, optional
            Number of pais to generate for contrastive learning, by default 2
        number_epochs : int, optional
            Number of epochs to use for contrastive learning, by default 2
        """

        trainer = SetFitTrainer(
            model=self.model,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["validation"],
            test_dataset=self.dataset["test"],
            loss_class=CosineSimilarityLoss,
            batch_size=self.config.batch_size,
            num_iterations=self.config.num_iterations,
            num_epochs=self.config.num_epochs_body,
            column_mapping={self.config.text_selection: "text", "labels": "label"},
            metric=self.metrics_setfit,
            wandb_project=self.config.wandb_project,
            wandb_entity=self.config.wandb_entity,
            text_selection=self.config.text_selection,
        )

        # Initialize W&B
        # trainer.init_wandb(model_id=self.ckpt)

        # Freeze the model and train only the body with contrastive learning
        print("Training body...")
        trainer.freeze()
        trainer.train(
            body_learning_rate=self.config.lr_body,
            learning_rate=self.config.lr_head,
            num_epochs=self.config.num_epochs_body,
            batch_size=self.config.batch_size,
        )
        print("Body training finished.")

        # # Unfreeze model and train the head
        print("Training head...")
        trainer.unfreeze(keep_body_frozen=True)
        trainer.train(
            learning_rate=self.config.lr_head,
            body_learning_rate=self.config.lr_body,
            num_epochs=self.config.num_epochs_head,
            batch_size=self.config.batch_size,
            l2_weight=self.config.weight_decay,
        )
        print("Head training finished.")

        # Set trainer to trainer object
        self.trainer = trainer

        if self.device == torch.device("mps"):
            self.model.to("cpu")
        self.model._save_pretrained(str(MODELS_DIR / f"SetFitMultiLabel_{self.ckpt}"))

        # Evaluate on test set
        if evaluate_test_set:
            test_metrics: Dict[str, float] = trainer.evaluate(use_test_set=True)
            # prefix test/ to metrics
            test_metrics = {f"test/{k}": v for k, v in test_metrics.items()}
            wandb.log(test_metrics)

    def test(self) -> None:
        preds = self.model.predict_proba(self.dataset["test"][self.text_selection])
        y_true = torch.tensor(self.dataset["test"]["labels"], dtype=torch.float32)
        print("---------Test set results---------")
        test_restuls: Dict[str, float] = self.metrics_setfit(
            preds, y_true, is_test=True
        )

        # get the index of the entry with the highest probability in each row
        max_indices = torch.argmax(preds, dim=1)

        # create a new matrix with 1 on the max probability entry in each row and 0 elsewhere
        y_pred = torch.zeros_like(preds)
        y_pred[torch.arange(preds.size(0)), max_indices] = 1

        clf_report = classification_report(
            y_true=y_true.cpu(),
            y_pred=y_pred.cpu(),
            target_names=self.labels,
            output_dict=True,
        )

        # Add prefix to metrics "test/"
        metrics = {f"test/{k}": v for k, v in test_restuls.items()}
        # Log results
        if wandb.run is not None:
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

        print(test_restuls)

    def predict(self, x: List[str]) -> np.ndarray:
        return self.trainer.model.predict_proba(x)

    def metrics_setfit(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        threshold: float = 0.5,
        is_test: bool = False,
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
        loss_fn = torch.nn.CrossEntropyLoss()

        if is_test:
            probs = predictions
        else:
            # first, apply softmax on predictions which are of shape (batch_size, num_labels)
            softmax = torch.nn.Softmax(dim=1)
            probs = softmax(torch.Tensor(predictions))

        # get the index of the entry with the highest probability in each row
        max_indices = torch.argmax(probs, dim=1)

        # create a new matrix with 1 on the max probability entry in each row and 0 elsewhere
        y_pred = torch.zeros_like(probs)
        y_pred[torch.arange(probs.size(0)), max_indices] = 1

        # finally, compute metrics
        y_true = labels
        y_true, y_pred, probs = y_true.cpu(), y_pred.cpu(), probs.cpu()
        loss = loss_fn(probs, y_true)
        f1_ = f1_score(y_true=y_true, y_pred=y_pred, average="macro")
        try:
            roc_auc = roc_auc_score(
                y_true.numpy(), probs.detach().numpy(), average="macro"
            )
        except ValueError:  # Of
            roc_auc = 0.0
        accuracy = accuracy_score(y_true, y_pred)

        # return as dictionary
        metrics = {
            "f1": f1_,
            "roc_auc": roc_auc,
            "accuracy": accuracy,
            "loss": loss.item() / len(y_true),
        }
        return metrics
