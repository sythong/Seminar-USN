# Libraries
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import VGG_MNIST_EVALUATE


# Dimension of the images
img_width, img_height = 28, 28

# Dataset path
train_data_dir = '/home/chrisander/datasets/MNIST/training'
validation_data_dir = '/home/chrisander/datasets/MNIST/testing'

# Dataset
nb_train_samples = 60000
nb_validation_samples = 10000

# Config
epoch = 1
batch_size = 32
num_classes = 10

# Saving model
save_model = True

# Channel input
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

print('Input shape = ', input_shape)

# Creating a small convolutional neural network (VGG architecture)
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape, data_format='channels_last'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# First block
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second block
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.05))

# Fully connected layers / Classifier
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.40))
model.add(Dense(num_classes, activation='softmax'))

# Compiles the model with loss and optimization function
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# This is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# We use no augmentation configuration for validation:
# only rescaling the RGB to float between 0 and 1.
validation_datagen = ImageDataGenerator(rescale=1. / 255)

# Reading the training images from the given path
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# Reading the validation images from the given path
validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# Training the model
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epoch,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

# Saving the model/weights
if save_model:
    model.save('VGG_MNIST.model')
    print('Model saved')
else:
    print('Model not saved')

VGG_MNIST_EVALUATE.EvaluatingModel()

del model
K.clear_session()
