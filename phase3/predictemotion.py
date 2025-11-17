import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('emotion_cnn_model.h5')

emotion_labels = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

def predict_emotion(img_path):
    # Read image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Show original image
    cv2.imshow("Input Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Preprocess
    img_resized = cv2.resize(img, (48,48))
    img_reshaped = img_resized.reshape(1,48,48,1) / 255.0

    # Predict
    predictions = model.predict(img_reshaped)
    emotion = emotion_labels[np.argmax(predictions)] 

    print("Predicted Emotion:", emotion)

predict_emotion("phase3/dataset/test/happy/PrivateTest_95094.jpg")