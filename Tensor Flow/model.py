import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist
from keras import optimizers
import keras
import cv2

#loading data from mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()

#filtering data - it should contain onlt 0 and 1
train_filter=np.logical_or(y_train==0,y_train==1) #consists of true(for 0 and 1) and false(otherwise) values
test_filter=np.logical_or(y_test==0,y_test==1)

#applying the above filter so that the new training and testing set contain only 0 and 1
x_train,y_train=x_train[train_filter],y_train[train_filter]
x_test,y_test=x_test[test_filter],y_test[test_filter]

y_train = 1 - y_train
y_test = 1 - y_test

#flattening the images and normalizing them
x_train=x_train.reshape(-1,784)/255.0
x_test=x_test.reshape(-1,784)/255.0

#building the neural network with three layers consisting of 25, 15 and 1 neuron each
model=Sequential([
    Dense(25,activation='relu',input_shape=(784,)),
    Dense(15,activation='relu'),
    Dense(1,activation='sigmoid')
])
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=['accuracy']
)
model.fit(
    x_train,y_train,
    epochs=20,
    batch_size=32,
    validation_data=(x_test,y_test)
)

#static prediction
prediction=model.predict(x_test[0].reshape(1,784))
digit=(1 if prediction>=0.5 else 0)
print("Prediction: ",digit)

prediction=model.predict(x_test[99].reshape(1,784))
digit=(1 if prediction>=0.5 else 0)
print("Prediction: ",digit)

#user-input / dynamic prediction
drawing=False
img=np.zeros((280,280),dtype=np.uint8)

def draw(event,x,y,flags,param):
    global drawing
    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
    elif event==cv2.EVENT_MOUSEMOVE and drawing:
        cv2.circle(img,(x,y),15,255,-1)
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False

window_name = "Draw Digit (0 or 1)"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, draw)

while True:
    cv2.imshow("Draw Digit (0 or 1)", img)
    key = cv2.waitKey(1)

    if key == ord('q'):   # press q to quit
        break

cv2.destroyAllWindows()
img_resize=cv2.resize(img,(28,28))
img_resize=255-img_resize
img_resize=img_resize/255.0
img_flat = img_resize.reshape(1,784)

prediction=model.predict(img_flat)
print("Raw prediction: ",prediction)

digit=1 if prediction>=0.5 else 0
print("Predicted digit: ",digit)