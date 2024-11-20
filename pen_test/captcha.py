import os
import glob
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


# Step 1: Load and preprocess the dataset
def load_data(captcha_image_folder, image_size=(64, 64)):
    image_files = glob.glob(os.path.join(captcha_image_folder, "*.png"))
    images = []
    labels = []

    for image_file in image_files:
        img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, image_size)  # Resize to consistent size
        img = img / 255.0  # Normalize the pixel values to [0, 1]
        images.append(img)

        # Extract label (based on the file name, if the filename follows a pattern)
        label = int(os.path.basename(image_file).split('_')[2])  # Assuming 'captcha_1_char_2.png' format
        labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    # Reshape images for the CNN: (num_samples, height, width, channels)
    images = images.reshape(images.shape[0], images.shape[1], images.shape[2], 1)
    labels = to_categorical(labels, num_classes=36)  # Assuming we have 36 classes (0-9, A-Z)

    return images, labels


# Step 2: Define the CNN model
def build_model(input_shape=(64, 64, 1), num_classes=36):
    model = Sequential()

    # First convolutional block
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))

    # Second convolutional block
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    # Flatten the 2D feature maps
    model.add(Flatten())

    # Dense layer for classification
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))  # Output layer for multi-class classification

    return model


# Step 3: Compile and train the model
def train_model(captcha_image_folder):
    # Load data
    X, y = load_data(captcha_image_folder)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the model
    model = build_model(input_shape=X_train.shape[1:], num_classes=y_train.shape[1])

    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Save the model
    model.save('captcha_character_recognition_model.h5')


# Step 4: Evaluate the model
def evaluate_model(model_path='captcha_character_recognition_model.h5', captcha_image_folder='captcha_images'):
    # Load the trained model
    model = keras.models.load_model(model_path)

    # Load data
    X, y = load_data(captcha_image_folder)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")


# Example Usage
captcha_image_folder = "path_to_captcha_images"
train_model(captcha_image_folder)
evaluate_model()