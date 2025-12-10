import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras.preprocessing import image

train_gen = ImageDataGenerator(rescale=0.001, validation_split=0.2)

train_data = train_gen.flow_from_directory(
    "dataset/",
    target_size=(150,150),
    batch_size=16,
    class_mode="binary",
    subset="training"
)

val_data = train_gen.flow_from_directory(
    "dataset/",
    target_size=(150,150),
    batch_size=16,
    class_mode="binary",
    subset="validation"
)

model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(150,150,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

test_img = image.load_img("test_image.jpg", target_size=(150,150))
test_img = image.img_to_array(test_img) / 255.0
test_img = np.expand_dims(test_img, axis=0)

prediction = model.predict(test_img)

if prediction[0] > 0.5:
    print("Nəticə: PNEVMONİYA ehtimalı yüksəkdir.")
else:
    print("Nəticə: NORMAL ehtimalı yüksəkdir.")
