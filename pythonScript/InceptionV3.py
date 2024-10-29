import os
import cv2
import numpy as np
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from wandb.integration.keras import WandbMetricsLogger
from image_preprocessors import apply_canny, apply_morphology, black_and_white

wandb.init(project="VisionTransformer")

image_size = (250, 250)
path_with_sign = os.path.join('..', 'data', 'y')
path_without_sign = os.path.join('..', 'data', 'n')

def load_images_from_folder(folder, label, target_size, img_type='normal'):
    images, labels = [], []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            img = cv2.imread(img_path)
            if img_type == 'canny':
                img = apply_canny(img, image_size)
            elif img_type == 'morphology':
                img = apply_morphology(img, target_size)
            elif img_type == 'normal':
                img = black_and_white(img, target_size)
            else:
                raise ValueError(f"Unknown image type '{img_type}'")
            images.append(img)
            labels.append(label)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
    return np.array(images), np.array(labels)

images_with_sign_canny, labels_with_sign_canny = load_images_from_folder(path_with_sign, 0, image_size, img_type='canny')
images_without_sign_canny, labels_without_sign_canny = load_images_from_folder(path_without_sign, 1, image_size, img_type='canny')
images_with_sign_morphology, labels_with_sign_morphology = load_images_from_folder(path_with_sign, 0, image_size, img_type='morphology')
images_without_sign_morphology, labels_without_sign_morphology = load_images_from_folder(path_without_sign, 1, image_size, img_type='morphology')
images_with_sign_normal, labels_with_sign_normal = load_images_from_folder(path_with_sign, 0, image_size, img_type='normal')
images_without_sign_normal, labels_without_sign_normal = load_images_from_folder(path_without_sign, 1, image_size, img_type='normal')

images_with_sign = np.concatenate((images_with_sign_canny, images_with_sign_morphology, images_with_sign_normal), axis=-1)
images_without_sign = np.concatenate((images_without_sign_canny, images_without_sign_morphology, images_without_sign_normal), axis=-1)
labels_with_sign = np.array(labels_with_sign_canny)
labels_without_sign = np.array(labels_without_sign_canny)

all_images = np.concatenate((images_with_sign, images_without_sign), axis=0)
all_labels = np.concatenate((labels_with_sign, labels_without_sign), axis=0)
all_images = all_images.astype('float32') / 255.0

train_images, test_images, train_labels, test_labels = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator()
train_generator = train_datagen.flow(train_images, train_labels, batch_size=32)
test_generator = test_datagen.flow(test_images, test_labels, batch_size=32)

base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=None)
base_model.trainable = False

def build_model(dropout, learning_rate):
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model

sweep_config = {
    "method": "bayes",
    "metric": {"name": "val_accuracy", "goal": "maximize"},
    "parameters": {
        "learning_rate": {"min": 0.00001, "max": 0.001},
        "dropout": {"min": 0.3, "max": 0.6},
        "batch_size": {"values": [16, 32, 64]}
    }
}

sweep_id = wandb.sweep(sweep_config, project="VisionTransformer")

def train_sweep():
    with wandb.init() as run:
        config = wandb.config
        model = build_model(config.dropout, config.learning_rate)

        history = model.fit(
            train_images, train_labels,
            batch_size=config.batch_size,
            epochs=20,
            validation_split=0.10,
            callbacks=[WandbMetricsLogger()]
        )

        test_loss, test_acc = model.evaluate(test_images, test_labels)
        wandb.log({"test_loss": test_loss, "test_acc": test_acc})

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(history.history['accuracy'], label='Train')
        axes[0].plot(history.history['val_accuracy'], label='Validation')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend(loc='upper left')

        axes[1].plot(history.history['loss'], label='Train')
        axes[1].plot(history.history['val_loss'], label='Validation')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend(loc='upper left')

        plt.suptitle('Model Training - InceptionV3', fontsize=16)
        plt.tight_layout()
        wandb.log({"training_plot": wandb.Image(fig)})
        plt.close(fig)

        predictions = (model.predict(test_images) > 0.5).astype(int)
        cm = confusion_matrix(test_labels, predictions)
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=['Wartelinie', 'keine Wartelinie'], yticklabels=['Wartelinie', 'keine Wartelinie'])
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title('Confusion Matrix - InceptionV3 Model')
        wandb.log({"confusion_matrix": wandb.Image(fig)})
        plt.close(fig)

wandb.agent(sweep_id, train_sweep)
wandb.finish()
