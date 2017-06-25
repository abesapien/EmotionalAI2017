
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

from keras.utils import to_categorical

from keras.models import load_model

for i in range(0,50):
	model = load_model('emotion_l2_' + str(i) + '.hd5')
	print(str(i))
	res=model.evaluate(np.array(X_test)[:,:,:,0:1]/255.0,to_categorical(y_test),verbose=0)
	print(res)
	res=model.evaluate(np.array(X_val)[:,:,:,0:1]/255.0,to_categorical(y_val),verbose=0)
	print(res)
	print('\n')
