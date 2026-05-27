import numpy as np
import keras
import cv2
from keras.layers import Dense

#loading data from mnist
(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()

#flattening the images
x_train=x_train.reshape(-1,784)/255.0
x_test=x_test.reshape(-1,784)/255.0

model=keras.Sequential([
    Dense(25,activation='relu',input_shape=(784,)),
    Dense(15,activation='relu'),
    Dense(10,activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001), 
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)
model.fit(
    x_train,y_train,
    epochs=20,
    batch_size=32,
    validation_data=(x_test,y_test)
)

y_pred=model.predict(x_test[0].reshape(-1,784))
print("Predicted digit:", np.argmax(y_pred))
print("Actual digit:", y_test[0])

drawing = False
img = np.zeros((280, 280), dtype=np.uint8)

def draw(event, x, y, flags, param):
    global drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.circle(img, (x, y), 8, 255, -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

window = "Draw Digit (0-9), press q when done"
cv2.namedWindow(window)
cv2.setMouseCallback(window, draw)

while True:
    cv2.imshow(window, img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
# Crop digit
coords = np.column_stack(np.where(img > 0))
if len(coords) > 0:
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    img = img[y0:y1, x0:x1]

# Resize to 20x20
img = cv2.resize(img, (20, 20))

# Pad to 28x28
pad = 4
img = np.pad(img, ((pad, pad), (pad, pad)), mode='constant')

# Invert colors (MNIST style)
img = 255 - img

# Normalize
img = img / 255.0

# Flatten
img = img.reshape(1, 784)

# Predict
prediction = model.predict(img)
digit = np.argmax(prediction)

print("Predicted digit:", digit)
