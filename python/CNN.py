import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import wandb
from wandb.integration.keras import WandbMetricsLogger


class ImageClassifier:
    def __init__(self, image_size=(250, 250), batch_size=32, project_name="VisionTransformer"):
        self.image_size = image_size
        self.batch_size = batch_size
        self.model = None

        wandb.init(project=project_name)

    def load_images_from_folder(self, folder, label):
        images, labels = [], []
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            try:
                img = load_img(img_path, target_size=self.image_size)
                img_array = img_to_array(img)
                images.append(img_array)
                labels.append(label)
            except Exception as e:
                print(f"Fehler beim Laden des Bildes {img_path}: {e}")
        return np.array(images), np.array(labels)

    def prepare_data(self, path_with_sign, path_without_sign):
        images_with_sign, labels_with_sign = self.load_images_from_folder(path_with_sign, 1)
        images_without_sign, labels_without_sign = self.load_images_from_folder(path_without_sign, 0)
        
        all_images = np.concatenate((images_with_sign, images_without_sign), axis=0) / 255.0
        all_labels = np.concatenate((labels_with_sign, labels_without_sign), axis=0)
        
        train_images, test_images, train_labels, test_labels = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)
        train_images, validation_images, train_labels, validation_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)
        
        return train_images, test_images, validation_images, train_labels, test_labels, validation_labels

    def build_model(self):
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(self.image_size[0], self.image_size[1], 3)),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
        self.model = model
        print(self.model.summary())

    def train(self, train_images, train_labels, validation_images, validation_labels):
        train_datagen = ImageDataGenerator(
            rotation_range=30, width_shift_range=0.2, height_shift_range=0.2,
            shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        validation_datagen = ImageDataGenerator()

        train_generator = train_datagen.flow(train_images, train_labels, batch_size=self.batch_size)
        validation_generator = validation_datagen.flow(validation_images, validation_labels, batch_size=self.batch_size)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = self.model.fit(
            train_generator,
            steps_per_epoch=len(train_images) // self.batch_size,
            epochs=50,
            validation_data=validation_generator,
            validation_steps=len(validation_images) // self.batch_size,
            callbacks=[WandbMetricsLogger(), reduce_lr, early_stopping]
        )
        return history

    def evaluate(self, test_images, test_labels):
        test_loss, test_acc = self.model.evaluate(test_images, test_labels)
        print(f"Testgenauigkeit: {test_acc:.4f}, Testverlust: {test_loss:.4f}")
        wandb.log({'test_loss': test_loss, 'test_acc': test_acc})
        return test_loss, test_acc

    def plot_training(self, history):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')

        plt.suptitle('Model Training - Basic CNN')
        plt.tight_layout()
        plt.show()

    def predict_and_report(self, test_images, test_labels):
        predictions = (self.model.predict(test_images) > 0.5).astype(int)
        print(classification_report(test_labels, predictions))

        namelabels = {0: 'Wartelinie', 1: 'keine Wartelinie'}
        cm = confusion_matrix(test_labels, predictions)
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=namelabels.values(), yticklabels=namelabels.values())
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix - Basic CNN Model')
        wandb.log({"confusion-matrix": wandb.Image(plt)})
        plt.show()


if __name__ == "__main__":
    classifier = ImageClassifier()
    
    path_with_sign = 'data/y'
    path_without_sign = 'data/n'

    train_images, test_images, validation_images, train_labels, test_labels, validation_labels = classifier.prepare_data(
        path_with_sign, path_without_sign
    )

    classifier.build_model()
    history = classifier.train(train_images, train_labels, validation_images, validation_labels)

    classifier.evaluate(test_images, test_labels)

    classifier.plot_training(history)

    classifier.predict_and_report(test_images, test_labels)

    wandb.finish()