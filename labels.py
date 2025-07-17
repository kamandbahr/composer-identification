import os
import numpy as np


DIR_in = '/Users/kamand/preprocessing/mfcc_features'  

max_time = 216       # number of time steps
num_mfcc = 20        # number of MFCC coefficients

x_train_pad = []
y_train = []

label_dict = {}
label_counter = 0

for filename in os.listdir(DIR_in):
    if filename.endswith('.npy'):
        # extract label from filename as the labels are name of the files
        composer = filename.split(',')[0].strip()

        # map composer to an integer label added to y_train
        if composer not in label_dict:
            label_dict[composer] = label_counter
            label_counter += 1
        mfcc = np.load(os.path.join(DIR_in, filename))  
        # making shapes the same padding zeros here!
        source = mfcc.T  
        padded = np.zeros((max_time, num_mfcc))
        length = min(source.shape[0], max_time)
        padded[:length] = source[:length]

        x_train_pad.append(padded)
        y_train.append(label_dict[composer])

#numpy
x_train_pad = np.array(x_train_pad)         # shape: (6547, 216, 40)
y_train = np.array(y_train)                 # shape: (6547,)

# add channel dimension for Conv2D least 3 dimension
x_train_pad = x_train_pad[..., np.newaxis]  # shape: (6547, 216, 40, 1)

print("x_train_pad shape:", x_train_pad.shape)
print("y_train shape:", y_train.shape)
print("Label mapping:", label_dict)
