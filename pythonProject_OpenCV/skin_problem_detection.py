import cv2
import cvlib as cv
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os
import numpy as np
from pip._internal.req.req_file import preprocess
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical



# @author Pallavi Biswas
# @author Ananya Guntur

def load_images_from_folder(folder):
    images = []
    labels = []
    for subdir in os.listdir(folder):
        subfolder_path = os.path.join(folder, subdir)
        if os.path.isdir(subfolder_path):
            label = subdir
            for filename in os.listdir(subfolder_path):
                file_path = os.path.join(subfolder_path, filename)
                # Skip files with .DS_Store extension
                if filename.endswith('.DS_Store'):
                    continue
                if os.path.isfile(file_path):
                    # Attempt to load the image
                    try:
                        image = cv2.imread(file_path)
                        if image is not None:
                            # Preprocess the image (resize, normalize, etc.)
                            image = cv2.resize(image, (224, 224))
                            image = image / 255.0  # Normalize pixel values to [0, 1]
                            images.append(image)
                            labels.append(label)
                        else:
                            print("Error: Failed to load image '{}'".format(file_path))
                    except Exception as e:
                        print("Error:", e)
                        print("Error loading image '{}'".format(file_path))
    return np.array(images), np.array(labels)


# Specify the directory containing the images
folder = 'Derm_Database'

# Load images and corresponding labels
images, labels = load_images_from_folder(folder)
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_one_hot = to_categorical(labels_encoded)
num_classes = len(set(labels))

print("Number of images:", len(images))
print("Number of labels:", len(labels))


def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model


# Step 3: Compile your model
model = build_model(input_shape=(224, 224, 3), num_classes=num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 4: Train your model
model.fit(images, labels_one_hot, epochs=10, batch_size=32, validation_split=0.2)

labels = ['Blackheads', 'Cold Sore', 'Cysts', 'Eczema', 'Hives', 'Mole', 'Psoriasis', 'Rosacea', 'Scar', 'Vitiligo',
          'Whiteheads']

accuracy = model.evaluate(images, labels_one_hot)
print("Test Accuracy:", accuracy)

# Initialize the video capture object
video_0 = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame0 = video_0.read()

    # Check if the frame is captured successfully
    if not ret:
        print("Error: Unable to capture frame from webcam.")
        break

    # Perform face detection
    faces, confidences = cv.detect_face(frame0)

    # Loop through detected faces
    for face, conf in zip(faces, confidences):
        # Get the coordinates of the face
        (startX, startY) = face[0], face[1]
        (endX, endY) = face[2], face[3]

        # Crop the face region
        face_region = frame0[startY:endY, startX:endX]

        # Perform skin condition analysis on the face region

        # Example: Drawing bounding box around the detected face
        cv2.rectangle(frame0, (startX, startY), (endX, endY), (0, 255, 0), 2)

    preprocessed_frame = cv2.resize(frame0, (224, 224)) / 255.0

    # Make predictions using the pre-trained model
    prediction = model.predict(np.expand_dims(preprocessed_frame, axis=0))
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = labels[predicted_class_index]

    # Overlay the predicted label on the frame
    cv2.putText(frame0, predicted_class_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Skin Condition Recognition', frame0)

    # Break the loop if 'a' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

# Release the webcam and close all OpenCV windows
