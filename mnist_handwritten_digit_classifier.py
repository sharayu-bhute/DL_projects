import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


#importing the dependencies
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import tensorflow as tf
tf.random.set_seed(3)
from tensorflow import keras
from keras.datasets import mnist
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

#loading the dataset
(X_train,Y_train),(X_test,Y_test)=mnist.load_data()
X_train=X_train/255
X_test=X_test/255
#setting up the layers of the nn
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(50,activation='relu'),
    keras.layers.Dense(50,activation='relu'),
    keras.layers.Dense(10,activation='sigmoid')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train,Y_train,epochs=10)

#accuracy on test data
loss,accuracy=model.evaluate(X_test,Y_test)
print(f'Test accuracy: {accuracy*100:.2f}%')
y_pred=model.predict(X_test)
y_pred_label=[np.argmax(i) for i in y_pred]
conf_mat=confusion_matrix(Y_test,y_pred_label)
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues')
plt.show()

#predicteing on input data
input_image_path=input("insert the image_path for the number image:")
input_image=cv2.imread(input_image_path)
if input_image is None:
    print("Image not found. Check the path!")
    exit()
cv2.imshow("input_image",input_image)
cv2.waitKey(0)         
cv2.destroyAllWindows()
print(input_image.shape)
grayscale = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
input_image_resize=cv2.resize(grayscale,(28,28))
print(input_image_resize.shape)
input_image_resize=input_image_resize/255
image_reshape=np.reshape(input_image_resize,[1,28,28])
input_prediction=model.predict(image_reshape)
input_pred_label=np.argmax(input_prediction)
print( "The number in the number in the image: ",input_pred_label)


