
import os
import csv
import numpy as np
import scipy.misc

import glob

X_train = []
X_train_id = []
y_train = []
for i in range(0,7):
    path = os.path.join('images', 'Training', str(i) , '*.png')
    files = sorted(glob.glob(path))
    for fl in files:
        flbase = os.path.basename(fl)
        img = scipy.misc.imread(fl)
        X_train.append(img)
        X_train_id.append(flbase)
        y_train.append(i)

X_val = []
X_val_id = []
y_val = []
for i in range(0,7):
    path = os.path.join('images', 'PublicTest', str(i) , '*.png')
    files = sorted(glob.glob(path))
    for fl in files:
        flbase = os.path.basename(fl)
        img = scipy.misc.imread(fl)
        X_val.append(img)
        X_val_id.append(flbase)
        y_val.append(i)

X_test = []
X_test_id = []
y_test = []
for i in range(0,7):
    path = os.path.join('images', 'PrivateTest', str(i) , '*.png')
    files = sorted(glob.glob(path))
    for fl in files:
        flbase = os.path.basename(fl)
        img = scipy.misc.imread(fl)
        X_test.append(img)
        X_test_id.append(flbase)
        y_test.append(i)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

batch_size = 16

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
	rotation_range=10,
	width_shift_range=0.1,
	height_shift_range=0.1,	
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow(np.array(X_train)[:,:,:,0:1],to_categorical(y_train),
        batch_size=batch_size)  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow(np.array(X_val)[:,:,:,0:1],to_categorical(y_val),
        batch_size=batch_size)

from keras.models import load_model

model = load_model('emotion.hd5')

model.fit_generator(
        train_generator,
        steps_per_epoch=5000 // batch_size,
        epochs=500,
        validation_data=validation_generator,
        validation_steps=5000 // batch_size)

model.save('emotion.hd5')
