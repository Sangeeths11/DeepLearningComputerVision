import os
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from keras import callbacks, layers, models, optimizers, preprocessing
from modules.data_augmentation import (
    get_test_image_data_generator,
    get_train_image_data_generator,
)
from modules.data_preprocessing import ImageType, load_images_from_folder
from modules.wandb_integration import (
    get_sweep_run_name,
    log_evaluation,
    log_image,
    log_model_artifact,
)
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from wandb.integration.keras import WandbMetricsLogger

import wandb

matplotlib.use("Agg")

# Path Configuration
DATA_PATH = os.path.join(".", "data")


class ImageClassifier:
    def __init__(
        self, image_size=(250, 250), batch_size=32, project_name="VisionTransformer"
    ):
        self.image_size = image_size
        self.batch_size = batch_size
        self.model = None
        wandb.init(project=project_name)

    def prepare_data(self, path_with_sign, path_without_sign):
        images_with_sign, labels_with_sign = load_images_from_folder(
            path_with_sign, 1, self.image_size, ImageType.ORIGINAL
        )
        images_without_sign, labels_without_sign = load_images_from_folder(
            path_without_sign, 0, self.image_size, ImageType.ORIGINAL
        )

        all_images = (
            np.concatenate((images_with_sign, images_without_sign), axis=0) / 255.0
        )
        all_labels = np.concatenate((labels_with_sign, labels_without_sign), axis=0)

        train_images, test_images, train_labels, test_labels = train_test_split(
            all_images, all_labels, test_size=0.2, random_state=42
        )
        train_images, validation_images, train_labels, validation_labels = (
            train_test_split(train_images, train_labels, test_size=0.2, random_state=42)
        )

        return (
            train_images,
            test_images,
            validation_images,
            train_labels,
            test_labels,
            validation_labels,
        )

    def build_model(self, config):
        model = models.Sequential(
            [
                layers.Conv2D(
                    32,
                    (3, 3),
                    activation="relu",
                    input_shape=(self.image_size[0], self.image_size[1], 3),
                ),
                layers.BatchNormalization(),
                layers.MaxPooling2D(2, 2),
                layers.Conv2D(64, (3, 3), activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D(2, 2),
                layers.Conv2D(128, (3, 3), activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D(2, 2),
                layers.Conv2D(256, (3, 3), activation="relu"),
                layers.BatchNormalization(),
                layers.MaxPooling2D(2, 2),
                layers.Flatten(),
                layers.Dense(128, activation="relu"),
                layers.Dropout(config.dropout),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer=optimizers.Adam(learning_rate=config.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        self.model = model

    def train(self, train_images, train_labels, validation_images, validation_labels):
        train_datagen = get_train_image_data_generator()
        validation_datagen = get_test_image_data_generator()

        train_generator = train_datagen.flow(
            train_images, train_labels, batch_size=self.batch_size
        )
        validation_generator = validation_datagen.flow(
            validation_images, validation_labels, batch_size=self.batch_size
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=3, min_lr=0.00001
        )
        early_stopping = callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )

        history = self.model.fit(
            train_generator,
            steps_per_epoch=len(train_images) // self.batch_size,
            epochs=20,
            validation_data=validation_generator,
            validation_steps=len(validation_images) // self.batch_size,
            callbacks=[WandbMetricsLogger(), reduce_lr, early_stopping],
        )
        return history

    def evaluate(self, test_images, test_labels):
        test_loss, test_acc = self.model.evaluate(test_images, test_labels)

        model_prediction = (self.model.predict(test_images) > 0.5).astype(int)

        f1 = f1_score(test_labels, model_prediction)

        log_evaluation(test_loss, test_acc, f1)

        return test_loss, test_acc

    def plot_training(self, history):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history.history["accuracy"])
        plt.plot(history.history["val_accuracy"])
        plt.title("Model Accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Validation"], loc="upper left")

        plt.subplot(1, 2, 2)
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("Model Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Validation"], loc="upper left")

        plt.suptitle("Model Training - Basic CNN")
        plt.tight_layout()
        log_image("training_plot.png", plt)
        # plt.savefig("training_plot.png")  # Save the plot instead of showing it
        # wandb.log({"training_plot": wandb.Image("training_plot.png")})
        # plt.close()  # Close the plot to avoid tkinter errors

    def predict_and_report(self, test_images, test_labels):
        predictions = (self.model.predict(test_images) > 0.5).astype(int)
        for i in range(len(test_images)):
            wandb.log(
                {
                    "image": [
                        wandb.Image(
                            test_images[i],
                            caption=f"Prediction: {predictions[i]}, True: {test_labels[i]}",
                        )
                    ],
                    "prediction": predictions[i],
                    "true_label": test_labels[i],
                }
            )

        # Log confusion matrix
        cm = confusion_matrix(test_labels, predictions)
        sns.heatmap(
            cm,
            annot=True,
            cmap="Blues",
            fmt="d",
            xticklabels=["Wartelinie", "keine Wartelinie"],
            yticklabels=["Wartelinie", "keine Wartelinie"],
        )
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix - Basic CNN Model")
        log_image("confusion_matrix.png", plt)
        """
        plt.savefig("confusion_matrix.png")
        wandb.log({"confusion_matrix": wandb.Image("confusion_matrix.png")})
        plt.close()
        """

    def save_model(self):
        self.model.save("cnn_model.h5")
        log_model_artifact("cnn-model", "cnn_model.h5")


if __name__ == "__main__":

    sweep_id: str = sys.argv[1]

    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    def sweep_agent():
        with wandb.init() as run:
            config = wandb.config

            run.name = get_sweep_run_name(
                config.learning_rate, config.batch_size, config.dropout
            )

            classifier = ImageClassifier(batch_size=config.batch_size)

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

            """
            path_with_sign = os.path.join(DATA_PATH, "y")
            path_without_sign = os.path.join(DATA_PATH, "n")

            (
                train_images,
                test_images,
                validation_images,
                train_labels,
                test_labels,
                validation_labels,
            ) = classifier.prepare_data(path_with_sign, path_without_sign)
            """

            classifier.build_model(config)
            history = classifier.train(
                training_images, training_labels, validation_images, validation_labels
            )

            classifier.evaluate(test_images, test_labels)

            classifier.plot_training(history)

            classifier.predict_and_report(test_images, test_labels)

            classifier.save_model()

    wandb.agent(sweep_id, sweep_agent)

    wandb.finish()
