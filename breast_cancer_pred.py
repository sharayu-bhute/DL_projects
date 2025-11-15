import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf 
tf.random.set_seed(2)
from tensorflow import keras


breast_cancer_dataset=sklearn.datasets.load_breast_cancer()
df=pd.DataFrame(breast_cancer_dataset.data,columns=breast_cancer_dataset.feature_names)
df['target']=breast_cancer_dataset.target
x=df.drop(columns='target',axis=1)
y=df['target']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
#standardization the data
scaler=StandardScaler()
x_train_standard=scaler.fit_transform(x_train)
x_test_standard=scaler.transform(x_test)

model=keras.Sequential([
    keras.layers.Flatten(input_shape=(30,)),
    keras.layers.Dense(20,activation='relu'),
    keras.layers.Dense(2,activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train_standard,y_train,validation_split=0.1,epochs=10)

# plt.plot(model.history.history['accuracy'])
# plt.plot(model.history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train','val'],loc='lower left')
# plt.show()
loss,accuracy=model.evaluate(x_test_standard,y_test)
print(f'Test accuracy: {accuracy*100:.2f}%')

y_pred=model.predict(x_test_standard)
y_pred_labels=[np.argmax(i) for i in y_pred]
input_data=( 14.97, 19.69, 98.24, 711.0, 0.09648, 0.08187, 0.06664, 0.04781, 0.1885, 0.05766, 0.3613, 1.137, 2.616, 27.38, 0.007193, 0.03123, 0.03456, 0.01487, 0.01994, 0.004303, 16.11, 25.41, 108.7, 880.3, 0.1328, 0.1984, 0.2436, 0.1283, 0.2977, 0.07259)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
input_data_standard=scaler.transform(input_data_reshaped)
prediction=model.predict(input_data_standard)
print(prediction)
prediction_label=[np.argmax(prediction)]
print(prediction_label)
if prediction_label[0]==0:
    print('The breast cancer is malignant')
else:
    print('The breast cancer is benign')





