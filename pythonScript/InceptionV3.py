import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from keras import applications, layers, models, optimizers, regularizers
from modules.data_augmentation import (
    get_test_image_data_generator,
    get_train_image_data_generator,
)
from modules.data_preprocessing import ImageType, load_images_from_folder
from modules.wandb_integration import get_sweep_run_name, log_evaluation, log_image
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from wandb.integration.keras import WandbMetricsLogger

import wandb

base_model = applications.InceptionV3(
    weights="imagenet", include_top=False, input_tensor=None
)
base_model.trainable = False


def build_model(dropout, learning_rate):
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.01))(
        x
    )
    x = layers.BatchNormalization(name="batch_norm_1")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
    x = layers.BatchNormalization(name="batch_norm_2")(x)
    predictions = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs=base_model.input, outputs=predictions)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":

    sweep_id: str = sys.argv[1]

    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    def sweep_agent():
        with wandb.init() as run:
            config = wandb.config

            run.name = get_sweep_run_name(
                config.learning_rate, config.batch_size, config.dropout
            )

            # Load Dataset
            artifact = run.use_artifact(
                "silvan-wiedmer-fhgr/VisionTransformer/swissimage-10cm-preprocessing:v1",
                type="dataset",
            )

            artifact_dir = artifact.download()

            # Get Training Data
            training_artifact = np.load(
                os.path.join(artifact_dir, "training-preprocessing.npy")
            )
            training_images = training_artifact["images"]
            training_labels = training_artifact["labels"]

            # Get Validation Data
            validation_artifact = np.load(
                os.path.join(artifact_dir, "validation-preprocessing.npy")
            )
            validation_images = validation_artifact["images"]
            validation_labels = validation_artifact["labels"]

            # Get Test Data
            test_artifact = np.load(
                os.path.join(artifact_dir, "test-preprocessing.npy")
            )
            test_images = test_artifact["images"]
            test_labels = test_artifact["labels"]

            train_datagen = get_train_image_data_generator()
            validation_datagen = get_test_image_data_generator()

            train_generator = train_datagen.flow(
                training_images, training_labels, batch_size=config.batch_size
            )
            validation_generator = validation_datagen.flow(
                validation_images, validation_labels, batch_size=config.batch_size
            )

            model = build_model(config.dropout, config.learning_rate)

            history = model.fit(
                train_generator,
                steps_per_epoch=len(training_images) // config.batch_size,
                batch_size=config.batch_size,
                epochs=20,
                validation_data=validation_generator,
                validation_steps=len(validation_images) // config.batch_size,
                callbacks=[WandbMetricsLogger()],
            )

            test_loss, test_acc = model.evaluate(test_images, test_labels)

            model_prediction = (model.predict(test_images) > 0.5).astype(int)

            f1 = f1_score(test_labels, model_prediction)

            log_evaluation(test_loss, test_acc, f1)

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].plot(history.history["accuracy"], label="Train")
            axes[0].plot(history.history["val_accuracy"], label="Validation")
            axes[0].set_title("Model Accuracy")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Accuracy")
            axes[0].legend(loc="upper left")

            axes[1].plot(history.history["loss"], label="Train")
            axes[1].plot(history.history["val_loss"], label="Validation")
            axes[1].set_title("Model Loss")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Loss")
            axes[1].legend(loc="upper left")

            plt.suptitle("Model Training - InceptionV3", fontsize=16)
            plt.tight_layout()
            log_image("training_plot", plt)
            # wandb.log({"training_plot": wandb.Image(fig)})
            # plt.close(fig)

            predictions = (model.predict(test_images) > 0.5).astype(int)
            cm = confusion_matrix(test_labels, predictions)
            fig, ax = plt.subplots(figsize=(10, 7))
            sns.heatmap(
                cm,
                annot=True,
                cmap="Blues",
                fmt="d",
                xticklabels=["Wartelinie", "keine Wartelinie"],
                yticklabels=["Wartelinie", "keine Wartelinie"],
            )
            ax.set_xlabel("Predicted Labels")
            ax.set_ylabel("True Labels")
            ax.set_title("Confusion Matrix - InceptionV3 Model")
            log_image("Confusion Matrix", plt)
            # wandb.log({"confusion_matrix": wandb.Image(fig)})
            # plt.close(fig)

    wandb.agent(sweep_id, sweep_agent)

    wandb.finish()
