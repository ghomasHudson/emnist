import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from matplotlib import pyplot as plt


# Load EMNIST Balanced dataset
(ds_train, ds_test), ds_info = tfds.load(
    'emnist/balanced',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

# Load dataset from dirs
dataset_dir = "dataset"

# Load training dataset without resizing
ds_train2 = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.1,
    subset="training",
    seed=123,
    image_size=(28, 28),               # Still required, but won't resize if already 28x28
    batch_size=None,
    color_mode='grayscale'             # Important for EMNIST-style data
)

# Load validation dataset
ds_test2 = tf.keras.utils.image_dataset_from_directory(
    dataset_dir,
    validation_split=0.1,
    subset="validation",
    seed=123,
    image_size=(28, 28),
    batch_size=None,
    color_mode='grayscale'
)


class DummyFeatures:
    def __init__(self, class_names):
        self.label = self.Label(class_names)

    class Label:
        def __init__(self, class_names):
            self.num_classes = len(class_names)
            self.names = class_names

class DummyDSInfo:
    def __init__(self, class_names):
        self.features = {'label': DummyFeatures(class_names).label}

class_names = ds_train2.class_names
ds_info2 = DummyDSInfo(class_names)


def transpose_image(image, label):
    # Rotate 90 CW and invert horizontally
    return tf.transpose(image, perm=[1, 0, 2]), label

# Normalize and batch the data
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, -1)  # Add channel dimension
    return image, label

batch_size = 128

ds_train = ds_train.map(transpose_image).map(preprocess).shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.map(transpose_image).map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)

ds_train2 = ds_train2.map(preprocess).shuffle(1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
ds_test2 = ds_test2.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Show an example
# import matplotlib.pyplot as plt

def show_example(ds, target_label):
    for image_batch, label_batch in ds.unbatch():
        if label_batch.numpy() == target_label:
            image = image_batch.numpy().squeeze()
            plt.figure(figsize=(5, 5))
            plt.imshow(image, cmap='gray')
            plt.title(f"Found label: {target_label}")
            plt.show()
            break

# target_label = 17
# show_example(ds_train, target_label)
# show_example(ds_train2, target_label)
# import sys;sys.exit()


# Callbacks
checkpoint_cb = ModelCheckpoint(
    "best_model.h5",  # Save path
    save_best_only=True,
    monitor="val_accuracy",  # You can use 'val_loss' too
    mode="max",
    verbose=1
)

earlystop_cb = EarlyStopping(
    monitor="val_accuracy",
    patience=3,  # Stop after 3 epochs without improvement
    mode="max",
    restore_best_weights=True,  # Restore weights of the best epoch
    verbose=1
)

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.1)  # Rotate up to ±10% of 180° (i.e., ±18°)
])


# Model definition
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    data_augmentation,  # <- applies only during training
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
    metrics=['accuracy'],
)


# Load the best model from checkpoint
#model.load_weights("emnist_trained.h5")

# Train the model
model.fit(ds_train, epochs=40, validation_data=ds_test2
        , callbacks=[checkpoint_cb, earlystop_cb])

model.fit(ds_train2, epochs=40, validation_data=ds_test2
        , callbacks=[checkpoint_cb, earlystop_cb])

# Load the best model from checkpoint
best_model = tf.keras.models.load_model("best_model.h5")

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
tflite_model = converter.convert()

# Save the .tflite model
with open("enmist_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Model trained and exported as emnist_model.tflite")
