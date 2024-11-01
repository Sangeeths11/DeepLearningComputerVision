from tensorflow.keras.preprocessing.image import ImageDataGenerator


def get_train_image_data_generator():
    return ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )


def get_test_image_data_generator():
    return ImageDataGenerator()
