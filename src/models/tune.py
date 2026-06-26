from __future__ import annotations

import mlflow
import optuna

from src.models.train import train_model


def objective(trial: optuna.Trial) -> float:
    hidden_dims = trial.suggest_categorical(
        "hidden_dims",
        [
            [32, 64, 32],
            [64, 128, 64],
            [32, 64, 128, 64, 32],
            [128, 64, 32],
            [256, 128, 64],
        ],
    )

    dropout = trial.suggest_float("dropout", 0.05, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    threshold = trial.suggest_float("threshold", 0.3, 0.7)

    score = train_model(
        hidden_dims=hidden_dims,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        epochs=80,
        threshold=threshold,
        experiment_name="telco-churn-ann-optuna",
    )

    return score


def tune_model(n_trials: int = 30):
    mlflow.set_experiment("telco-churn-ann-optuna")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print("Best validation F1:", study.best_value)
    print("Best parameters:")
    print(study.best_params)


if __name__ == "__main__":
    tune_model(n_trials=30)