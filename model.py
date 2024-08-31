import cv2
import os
import keras
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import normalize, to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense

images_directory = '/content/drive/MyDrive/ PROJECT'
no_tumor_images = os.listdir('/content/drive/MyDrive/ PROJECT/no')
yes_tumor_images = os.listdir('/content/drive/MyDrive/ PROJECT/yes')
dataset = []
label = []
INPUT_SIZE = 150

import matplotlib.pyplot as plt

for i in range(5):
  plt.imshow(plt.imread(images_directory + '/no/' + no_tumor_images[i]), cmap='gray')
  plt.show()
  plt.imshow(plt.imread(images_directory + '/yes/' + yes_tumor_images[i]), cmap='gray')
  plt.show()



print(len(no_tumor_images))
print(len(yes_tumor_images))

for i , image_name in enumerate(no_tumor_images):
  if(image_name.split('.')[1]=='jpg'):
    image = cv2.imread(images_directory + '/no/' + image_name)
    image = Image.fromarray(image , 'RGB')
    image = image.resize((INPUT_SIZE , INPUT_SIZE))
    dataset.append(np.array(image))
    label.append(0)

for i , image_name in enumerate(yes_tumor_images):
  if(image_name.split('.')[1]=='jpg'):
    image = cv2.imread(images_directory +'/yes/' + image_name)
    image = Image.fromarray(image , 'RGB')
    image = image.resize((INPUT_SIZE , INPUT_SIZE))
    dataset.append(np.array(image))
    label.append(1)

print(len(dataset))
print(len(label))

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 5, figsize=(10, 10))
for i, ax in enumerate(axes):
  ax.imshow(dataset[i], cmap='gray')
  ax.set_title(f"Label: {label[i]}")
  ax.axis("off")
plt.show()

import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 5, figsize=(10, 10))
for i, ax in enumerate(axes[0]):
  ax.imshow(dataset[i], cmap='gray')
  ax.set_title(f"Label: {label[i]}")
  ax.axis("off")

for i, ax in enumerate(axes[1]):
  ax.imshow(dataset[len(no_tumor_images) + i], cmap='gray')
  ax.set_title(f"Label: {label[len(no_tumor_images) + i]}")
  ax.axis("off")

plt.show()

for image in dataset:
    print(image)

dataset = np.array(dataset)
label = np.array(label)

x_train , x_test , y_train , y_test = train_test_split(dataset , label , test_size = 0.2 , train_size=0.8
                                                       , random_state=0)

print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)

x_train = normalize(x_train  , axis=1)
x_test = normalize(x_train  , axis=1)

y_train = to_categorical(y_train , num_classes=2)
y_test = to_categorical(y_test , num_classes=2)

import numpy as np
import matplotlib.pyplot as plt
from keras.activations import sigmoid
from keras.backend import flatten
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Conv2D(32 , (3,3) , input_shape = (INPUT_SIZE , INPUT_SIZE , 3))) # step -1 Convolution
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2 , 2))) # step -2  Pooling

#Adding second convolutional layer
model.add(Conv2D(32 , (3,3) , kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2 , 2)))

#Adding third convolutional layer

model.add(Conv2D(64 , (3,3) , kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2 , 2)))

model.add(Flatten()) # step -3 :  Flattening (hidden layer)
model.add(Dense(64)) #
model.add(Activation('relu')) # step - 4 : full connection

model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('sigmoid')) # step - 5 : output layer

model.compile(loss='categorical_crossentropy' , optimizer = 'adam' ,  metrics=['accuracy'])
res = model.fit(np.array(x_train), np.array(y_train), verbose=1, epochs=10 ,batch_size=16,
         shuffle = False)
model.save('Brain_Tumor_Detection_Model.h5')

import matplotlib.pyplot as plt
print(res.history.keys())
print()
# loss
plt.plot(res.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

print()
print()
print("Training accuracy")
print()
print()
# accuracy

plt.plot(res.history['accuracy'])
plt.title('Model Accuracy')
plt.legend(['Training Accuracy'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

import cv2
from keras.models import load_model
from PIL import Image
import numpy as np
model = load_model('Brain_Tumor_Detection_Model.h5')

image = cv2.imread('/content/drive/MyDrive/FINAL YEAR PROJECT 24/pred/pred12.jpg')
img = Image.fromarray(image)
img = img.resize((64 , 64))
img = np.array(img)
print(img)

import cv2
from keras.models import load_model
from PIL import Image
import numpy as np
model = load_model('Brain_Tumor_Detection_Model.h5')
image = cv2.imread('/content/drive/MyDrive/FINAL YEAR PROJECT 24/pred/pred12.jpg')
img = Image.fromarray(image)
img = img.resize((150 , 150))
img = np.array(img)

input_img = np.expand_dims(img , axis=0)
predict_x=model.predict(input_img)
result=np.argmax(predict_x,axis=1)

if result==1:
  print(f' This patient is affected by tumor  :: {result} ')
else:
  print(f'this patient is not affected by tumor :: {result} ')
