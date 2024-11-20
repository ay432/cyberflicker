import os
from keras.preprocessing.image import ImageDataGenerator
from mesonet_classifier import Meso4  # Import Meso4 from mesonet_classifier

# Load the MesoNet model with pre-trained weights
classifier = Meso4()
classifier.load('mesonet_weights/Meso4_DF')

# Set up ImageDataGenerator for rescaling images
valDataGenerator = ImageDataGenerator(rescale=1./255)

# Create a data generator for loading images from a directory
# Assuming images are in the 'mesonet_test_images' directory
val_generator = valDataGenerator.flow_from_directory(
    'mesonet_test_images/',  # Path to your image directory
    target_size=(224, 224),  # Resize images to match MesoNet's input size
    batch_size=32,
    class_mode=None,  # No labels because we are only predicting
    shuffle=False  # Do not shuffle for ordered predictions
)

# Get the next batch of images and their labels (if available)
X = val_generator.next()

# Predict the class for each image in the batch
predictions = classifier.predict(X)

# Interpret the predictions (output is a probability for 'real' class)
# You can threshold this probability to classify as 'real' or 'fake'
pred_labels = ['real' if p > 0.5 else 'fake' for p in predictions]

# Print the predicted labels for each image
for filename, pred in zip(val_generator.filenames, pred_labels):
    print(f'{filename}: {pred}')