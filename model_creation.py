import os
import tensorflow as tf
from keras import layers
import keras
from keras import Sequential
import matplotlib.pyplot as plt


dataset_path = os.getenv('DATASET_PATH')

training_subset = tf.keras.utils.image_dataset_from_directory(
  dataset_path,
  validation_split=0.1,
  subset="training",
  seed=123,
  image_size=(240, 240),
  batch_size=32)

validation_subset = tf.keras.utils.image_dataset_from_directory(
  dataset_path,
  validation_split=0.1,
  subset="validation",
  seed=123,
  image_size=(240, 240),
  batch_size=32)

AUTOTUNE = tf.data.AUTOTUNE

training_subset = training_subset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_subset = validation_subset.cache().prefetch(buffer_size=AUTOTUNE)


data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(240,
                                  240,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)



model = Sequential([
  data_augmentation,
  layers.Rescaling(1./255),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


model.summary()

training_process = model.fit(training_subset, epochs=10, validation_data=validation_subset)

acc = training_process.history['accuracy']
val_acc = training_process.history['val_accuracy']

loss = training_process.history['loss']
val_loss = training_process.history['val_loss']

epochs_range = range(10)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


model.save(os.getenv('MODEL_PATH'))
