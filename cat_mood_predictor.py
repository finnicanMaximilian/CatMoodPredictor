import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os
import numpy as np

# Get the folder where this script lives
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build full path to CatMoods folder right next to the script
data_dir = os.path.join(script_dir, "CatMoods")
print(f"Looking for CatMoods folder at: {data_dir}")

img_height, img_width = 128, 128
batch_size = 8

# Load Dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names
print("Cat mood categories:", class_names)

for mood in class_names:
    path = os.path.join(data_dir, mood)
    pics = os.listdir(path)
    print(f"{mood} folder has {len(pics)} pictures: {pics}")

# Normalize images
train_ds = train_ds.map(lambda x, y: (x / 255.0, y))

# Build Model
model = models.Sequential([
    layers.Conv2D(16, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train Model
history = model.fit(train_ds, epochs=5)

# Plot Accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# Save Model
model.save("cat_mood_model.h5")
print("Model saved! Now it knows when to ignore you.")

# Path to the folder with your new test cat images
test_folder = "submissions"

# Define your mood categories (adjust to match your model’s training labels)
moods = ["Curious", "Eepy", "Grumpy", "Happy", "Zoomies"]

# Loop through images in submissions folder
for img_name in os.listdir(test_folder):
    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        img_path = os.path.join(test_folder, img_name)
        img = image.load_img(img_path, target_size=(128, 128))  # adjust to your model input
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0  # normalize

        preds = model.predict(x)
        predicted_mood = moods[np.argmax(preds)]

        print(f"{img_name} → {predicted_mood}")