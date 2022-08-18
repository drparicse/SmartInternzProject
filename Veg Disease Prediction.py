from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1.0/255,shear_range = 0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1)
x_train = train_datagen.flow_from_directory('Veg-dataset/train',target_size = (64,64),batch_size=16, class_mode='categorical')
x_test = train_datagen.flow_from_directory('Veg-dataset/test',target_size = (64,64),batch_size=16, class_mode='categorical')

model = Sequential()
model.add(Convolution2D(32,(3,3),input_shape = (64,64,3),activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.summary()
model.add(Dense(9,activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = "adam", metrics = ["accuracy"])

model.fit_generator(x_train, steps_per_epoch = 89, epochs = 20, validation_data =x_test, validation_steps = 27)

model.save("vegetable.h5")

model.summary()


