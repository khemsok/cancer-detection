from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.applications.resnet50 import ResNet50


test_dir = './dataset/test'
train_dir = './dataset/train'

model = Sequential()

model.add(Conv2D(32, (2, 2), input_shape=(96, 96, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# from keras.applications import InceptionResNetV2

# conv_base = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(96,96,3))
# model = Sequential()
# model.add(conv_base)
# model.add(Flatten())
# model.add(Dense(64, activation='relu'))
# model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(96, 96),
    batch_size=16, class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(96, 96),
    batch_size=16, class_mode='binary')

  
model.fit_generator(train_generator, 
    steps_per_epoch = 64000 // 16, 
    epochs = 25, validation_data = test_generator, 
    validation_steps = 16000 // 16) 

model.save('model2v2.h5')

model.save('model2v2')
