import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN , Dense 
model=Sequential([
    SimpleRNN(128,input_shape=(50,1)),
    Dense(1)
])
model.compile(optimizer='adam',loss='mse')
model.summary()