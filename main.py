import tensorflow_datasets as tfds
import tensorflow as tf


# Load EMNIST Balanced dataset
(ds_train, ds_test), ds_info = tfds.load(
    'emnist/balanced',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)


# augment = tf.keras.Sequential([
#     tf.keras.layers.RandomRotation(0.1)  # Rotates ±10% of 360° → ~±36°
# ])

# def augment(image, label):
#     # Random rotate by 20-degree intervals
#     image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
#     image = tf.keras.layers.RandomRotation(0.1)
#     return image, label

# Normalize and batch the data
def preprocess(image, label):

    # Rotate 90 CW and invert horizontally
    image = tf.transpose(image, perm=[1, 0, 2])

    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, -1)  # Add channel dimension
    return image, label

batch_size = 128

ds_train = ds_train.map(preprocess).shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Show an example
# import matplotlib.pyplot as plt
# plt.figure(figsize=(5, 5))
# plt.imshow(ds_train.take(1).as_numpy_iterator().next()[0][0].squeeze(), cmap='gray')
# plt.show()

# Model definition
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(ds_info.features['label'].num_classes, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model.fit(ds_train, epochs=5, validation_data=ds_test)

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the .tflite model
with open("emnist_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Model trained and exported as emnist_model.tflite")
