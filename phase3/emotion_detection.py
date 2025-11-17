import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers,models
train_dir='phase3/dataset/train'
test_dir='phase3/dataset/test'

IMG_SiZE=48
BATCH=64

train_datagen=ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
)

test_datagen=ImageDataGenerator(rescale=1./255)

train_data=train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SiZE,IMG_SiZE),
    batch_size=BATCH,
    color_mode='grayscale',
    class_mode='categorical'
)

test_data=test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SiZE,IMG_SiZE),
    batch_size=BATCH,
    color_mode='grayscale',
    class_mode='categorical'
)

model=models.Sequential([
    layers.Conv2D(32,(3,3),activation='relu',input_shape=(48,48,1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64,(3,3),activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128,(3,3),activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),

    layers.Dense(128,activation='relu'),
    layers.Dropout(0.3),

    layers.Dense(7,activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history=model.fit(
    train_data,
    epochs=25,
    validation_data=test_data
)

model.save("emotion_cnn_model.h5")

import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("emotion_cnn_model.h5")

emotion_labels = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

def predict_emotion(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48,48))
    img = img.reshape(1,48,48,1)/255.0
    
    predictions = model.predict(img)
    emotion = emotion_labels[np.argmax(predictions)]
    
    print("Predicted Emotion:", emotion)

predict_emotion("test.jpg")
