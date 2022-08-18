import tensorflow
import keras
import cv2
import flask
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1.0/255,shear_range = 0.2, zoom_range=0.2, horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1)
x_train = train_datagen.flow_from_directory('fruit-dataset/train',target_size = (64,64),batch_size=24, class_mode='categorical')
x_test = train_datagen.flow_from_directory('fruit-dataset/test',target_size = (64,64),batch_size=24, class_mode='categorical')
print(x_train.class_indices)
train_label_map = (x_train.class_indices)
print(train_label_map)
test_label_map = (x_test.class_indices)
print(test_label_map)
model = Sequential()
model.add(Convolution2D(32,(3,3),input_shape = (64,64,3),activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.summary()
model.add(Dense(6,activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = "adam", metrics = ["accuracy"])
model.fit_generator(x_train, steps_per_epoch=len(x_train), validation_data=x_test, validation_steps=len(x_test), epochs = 10)
model.save("fruit.h5")

model.summary()
