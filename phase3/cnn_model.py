import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,models

model=models.Sequential([
    layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)),
#Takes a 32×32×3 image
# Applies 32 filters of size 3×3
# Extracts low-level features like edges, corners, color blobs.

    layers.MaxPooling2D((2,2)),
# Reduces size 32×32 → 16×16
# Keeps only the strongest features
# Reduces computation and overfitting

    layers.Conv2D(64,(3,3),activation='relu'),
# Applies 64 filters of size 3×3
# Learns more complex patterns
# like curves, shapes, textures
# Output: 16×16 → 14×14 (if no padding)

    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation='relu'),

    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(10,activation='softmax')
])

model.summary()

