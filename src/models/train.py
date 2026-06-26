from __future__ import annotations

from pathlib import Path
from typing import Any
import copy
import json

import mlflow
import mlflow.pytorch
import numpy as np
import torch

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.ann import ANNModel
from src.data.build_dataset import MyDataset
from src.models.engine import train_one_epoch
from src.models.evaluate import (
    collect_predictions,
    evaluate,
    get_classification_report,
    get_confusion_matrix,
    tune_threshold,
)
from src.utils.seed import seed_everything
from src.utils.split_data import split_data
from src.utils.helpers import log_metrics, make_json_serializable


def train_model(
    hidden_dims: list[int] | None = None,
    dropouts: list[float] | None = None,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    batch_size: int = 8,
    epochs: int = 80,
    threshold: float = 0.5,
    random_state: int = 41,
    experiment_name: str = "telco-churn-ann",
) -> dict[str, Any]:
    """
    Train, validate, tune threshold, evaluate on test data, and log everything to MLflow.
    """

    seed_everything(random_state)

    x_train, x_val, x_test, y_train, y_val, y_test = split_data(
        random_state=random_state,
    )

    train_set = MyDataset(x_train, y_train)
    val_set = MyDataset(x_val, y_val)
    test_set = MyDataset(x_test, y_test)

    generator = torch.Generator()
    generator.manual_seed(random_state)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if hidden_dims is None:
        hidden_dims = [32, 64, 128, 64, 32]

    if dropouts is None:
        dropouts = [0.1, 0.1, 0.1, 0.2, 0.1]

    model = ANNModel(
        input_dim=x_train.shape[1],
        hidden_dims=hidden_dims,
        dropouts=dropouts,
    ).to(device)

    loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    mlflow.set_experiment(experiment_name)

    best_val_binary_f1 = 0.0
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0

    with mlflow.start_run():
        mlflow.log_params(
            {
                "input_dim": x_train.shape[1],
                "hidden_dims": str(hidden_dims),
                "dropouts": str(dropouts),
                "lr": lr,
                "weight_decay": weight_decay,
                "batch_size": batch_size,
                "epochs": epochs,
                "fixed_threshold": threshold,
                "random_state": random_state,
                "optimizer": "Adam",
                "loss_fn": "BCEWithLogitsLoss",
                "device": str(device),
                "train_size": len(train_set),
                "val_size": len(val_set),
                "test_size": len(test_set),
            }
        )

        progress_bar = tqdm(
            range(epochs),
            desc="Training ANN",
            leave=True,
        )

        for epoch in progress_bar:
            train_metrics = train_one_epoch(
                model=model,
                loader=train_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device,
                threshold=threshold,
            )

            val_metrics = evaluate(
                model=model,
                loader=val_loader,
                loss_fn=loss_fn,
                device=device,
                threshold=threshold,
            )

            log_metrics("train", train_metrics, step=epoch)
            log_metrics("val", val_metrics, step=epoch)

            progress_bar.set_postfix(
                {
                    "train_loss": f"{train_metrics['loss']:.4f}",
                    "train_acc": f"{train_metrics['accuracy']:.4f}",
                    "val_acc": f"{val_metrics['accuracy']:.4f}",
                    "val_f1": f"{val_metrics['binary_f1']:.4f}",
                }
            )

            if val_metrics["binary_f1"] > best_val_binary_f1:
                best_val_binary_f1 = val_metrics["binary_f1"]
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch + 1

        if best_state is not None:
            model.load_state_dict(best_state)

        mlflow.log_metric("best_epoch", best_epoch)
        mlflow.log_metric("best_val_binary_f1_at_0_5", best_val_binary_f1)

        val_metrics_fixed = evaluate(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            threshold=threshold,
        )

        test_metrics_fixed = evaluate(
            model=model,
            loader=test_loader,
            loss_fn=loss_fn,
            device=device,
            threshold=threshold,
        )

        log_metrics("final_val_fixed_threshold", val_metrics_fixed)
        log_metrics("final_test_fixed_threshold", test_metrics_fixed)

        val_probs, val_targets = collect_predictions(
            model=model,
            loader=val_loader,
            device=device,
        )

        best_threshold, best_threshold_val_f1 = tune_threshold(
            probs=val_probs,
            targets=val_targets,
        )

        mlflow.log_metric("best_threshold", best_threshold)
        mlflow.log_metric("best_threshold_val_binary_f1", best_threshold_val_f1)

        test_metrics_tuned = evaluate(
            model=model,
            loader=test_loader,
            loss_fn=loss_fn,
            device=device,
            threshold=best_threshold,
        )

        log_metrics("final_test_tuned_threshold", test_metrics_tuned)

        test_probs, test_targets = collect_predictions(
            model=model,
            loader=test_loader,
            device=device,
        )

        report_fixed = get_classification_report(
            probs=test_probs,
            targets=test_targets,
            threshold=threshold,
        )

        report_tuned = get_classification_report(
            probs=test_probs,
            targets=test_targets,
            threshold=best_threshold,
        )

        cm_fixed = get_confusion_matrix(
            probs=test_probs,
            targets=test_targets,
            threshold=threshold,
        )

        cm_tuned = get_confusion_matrix(
            probs=test_probs,
            targets=test_targets,
            threshold=best_threshold,
        )

        final_summary = {
            "best_epoch": best_epoch,
            "fixed_threshold": threshold,
            "best_threshold": best_threshold,
            "best_val_binary_f1_at_0_5": best_val_binary_f1,
            "best_threshold_val_binary_f1": best_threshold_val_f1,
            "final_val_fixed_threshold": val_metrics_fixed,
            "final_test_fixed_threshold": test_metrics_fixed,
            "final_test_tuned_threshold": test_metrics_tuned,
        }

        final_summary = make_json_serializable(final_summary)

        artifacts_dir = Path("artifacts")
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        fixed_report_path = artifacts_dir / "classification_report_fixed_threshold.txt"
        tuned_report_path = artifacts_dir / "classification_report_tuned_threshold.txt"
        fixed_cm_path = artifacts_dir / "confusion_matrix_fixed_threshold.json"
        tuned_cm_path = artifacts_dir / "confusion_matrix_tuned_threshold.json"
        metrics_path = artifacts_dir / "final_metrics.json"

        fixed_report_path.write_text(report_fixed)
        tuned_report_path.write_text(report_tuned)

        fixed_cm_path.write_text(
            json.dumps(
                make_json_serializable(cm_fixed),
                indent=4,
            )
        )

        tuned_cm_path.write_text(
            json.dumps(
                make_json_serializable(cm_tuned),
                indent=4,
            )
        )

        metrics_path.write_text(
            json.dumps(
                final_summary,
                indent=4,
            )
        )

        mlflow.log_artifact(str(fixed_report_path))
        mlflow.log_artifact(str(tuned_report_path))
        mlflow.log_artifact(str(fixed_cm_path))
        mlflow.log_artifact(str(tuned_cm_path))
        mlflow.log_artifact(str(metrics_path))

        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)

        model.eval()

        model_to_log = copy.deepcopy(model).cpu()
        model_to_log.eval()

        model_path = models_dir / "ann_customer_churn.pt"
        torch.save(model_to_log.state_dict(), model_path)

        mlflow.log_artifact(str(model_path))

        input_example = torch.tensor(
            x_train.iloc[:1].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )

        mlflow.pytorch.log_model(
            pytorch_model=model_to_log,
            name="ann_model",
            input_example=input_example,
            serialization_format="pickle",
        )

        print("\n================ FIXED THRESHOLD TEST REPORT ================")
        print(f"Threshold: {threshold}")
        print(report_fixed)

        print("\n================ TUNED THRESHOLD TEST REPORT ================")
        print(f"Best threshold from validation: {best_threshold:.2f}")
        print(report_tuned)

        print("\n================ FINAL TEST METRICS ================")

        print("\nFixed threshold:")
        for metric_name, value in test_metrics_fixed.items():
            print(f"{metric_name}: {value:.4f}")

        print("\nTuned threshold:")
        for metric_name, value in test_metrics_tuned.items():
            print(f"{metric_name}: {value:.4f}")

        return final_summary


if __name__ == "__main__":
    train_model(
        hidden_dims=[32, 64, 128, 64, 32],
        dropouts=[0.1, 0.1, 0.1, 0.2, 0.1],
        lr=1e-4,
        weight_decay=0.0,
        batch_size=8,
        epochs=80,
        threshold=0.5,
        random_state=41,
        experiment_name="telco-churn-ann",
    )
    