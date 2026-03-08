import os
os.environ["SM_FRAMEWORK"] = "tf.keras"

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    CSVLogger,
    LearningRateScheduler,
    EarlyStopping,
)

import segmentation_models as sm
sm.set_framework("tf.keras")

from deepd3.model import DeepD3_Model
from deepd3.training.stream import DataGeneratorStream

import numpy as np
import random

from config_utils import load_config, cfg_get


# ==========================
# Learning rate scheduler
# ==========================
def build_scheduler(decay_start_epoch: int, decay_rate: float):
    def schedule(epoch, lr):
        if epoch < decay_start_epoch:
            return lr
        else:
            return lr * tf.math.exp(-decay_rate)
    return schedule


# ==========================
# Model builder
# ==========================
def build_model(filters: int, learning_rate: float):
    m = DeepD3_Model(filters=int(filters))
    m.compile(
        Adam(learning_rate=float(learning_rate)),
        [sm.losses.dice_loss, "mse"],
        metrics=["acc", sm.metrics.iou_score],
    )
    return m


# ==========================
# Main
# ==========================
if __name__ == "__main__":

    cfg = load_config("config.yaml")

    # Docker専用ルート
    docker_base_path = cfg_get(cfg, ["docker_base_path"], None)
    if not docker_base_path:
        raise KeyError("config.yaml must define: deepd3_training.docker_base_path")

    repeats = int(cfg_get(cfg, ["repeats"], 3))
    epochs_per_shuffle = int(cfg_get(cfg, ["deepd3_training", "epochs_per_shuffle"], 20))
    batch_size = int(cfg_get(cfg, ["deepd3_training", "batch_size"], 8))
    target_resolution = float(cfg_get(cfg, ["deepd3_training", "target_resolution"], 0.0353))
    learning_rate = float(cfg_get(cfg, ["deepd3_training", "learning_rate"], 0.0005))
    filters = int(cfg_get(cfg, ["deepd3_training", "filters"], 32))
    min_content = int(cfg_get(cfg, ["deepd3_training", "min_content"], 50))
    early_patience = int(cfg_get(cfg, ["deepd3_training", "early_stopping_patience"], 5))
    lr_decay_start = int(cfg_get(cfg, ["deepd3_training", "lr_decay_start_epoch"], 15))
    lr_decay_rate = float(cfg_get(cfg, ["deepd3_training", "lr_decay_rate"], 0.1))
    base_seed = int(cfg_get(cfg, ["deepd3_training", "base_seed"], 42))

    scheduler = build_scheduler(lr_decay_start, lr_decay_rate)

    for i in range(repeats):

        print(f"\n========== Training on shuffle_{i} ==========")

        tf.keras.backend.clear_session()

        seed = base_seed + i
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        training_data_path = os.path.join(
            docker_base_path,
            "Training_dataset",
            f"shuffle_{i}",
            f"shuffle_{i}.d3set",
        )

        validation_data_path = os.path.join(
            docker_base_path,
            "Validation_dataset",
            f"shuffle_{i}",
            f"shuffle_{i}.d3set",
        )

        dg_training = DataGeneratorStream(
            training_data_path,
            batch_size=batch_size,
            target_resolution=target_resolution,
            min_content=min_content,
        )

        dg_validation = DataGeneratorStream(
            validation_data_path,
            batch_size=batch_size,
            target_resolution=target_resolution,
            min_content=min_content,
            augment=False,
            shuffle=False,
        )

        model = build_model(filters, learning_rate)
        model.summary()

        ckpt_path = os.path.join(docker_base_path, f"DeepD3_model_s{i}.h5")
        log_path = os.path.join(docker_base_path, f"DeepD3_training_s{i}.csv")

        mc = ModelCheckpoint(
            ckpt_path,
            save_best_only=True,
            monitor="val_loss",
            mode="min",
        )

        csv = CSVLogger(log_path, append=False)

        lrs = LearningRateScheduler(scheduler)

        es = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=early_patience,
            restore_best_weights=True,
        )

        model.fit(
            dg_training,
            epochs=epochs_per_shuffle,
            validation_data=dg_validation,
            callbacks=[mc, csv, lrs, es],
            shuffle=True,
        )

        final_path = os.path.join(docker_base_path, f"DeepD3_final_model_s{i}.h5")
        model.save(final_path)
