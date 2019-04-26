#%% [markdown]
# ## Warning this code will not run as is
# 
# ### This file has been used with google colab in order to save weights, in order to run this notebook locally you must download the dataset and change the path variable to the path for your download

#%%
import numpy as np
import keras
from glob import glob
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import sklearn
from sklearn.preprocessing import LabelEncoder

from keras.utils import to_categorical
from google.colab import drive
drive.mount('/content/drive')

img_size = 64


#%%

#print(os.listdir('C:/Users/Owner/4thYear/MainProject/ImageRecNN/input/fruits-360'))
path = '/content/drive/My Drive/Training/*'


training_fruit_img = []
training_label = []
for i in glob(path):
    img_label = i.split("/")[-1]
    for img_path in glob(os.path.join(i, "*.jpg")):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_size, img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        training_fruit_img.append(img)
        training_label.append(img_label)
training_fruit_img = np.array(training_fruit_img).astype('float32')
training_fruit_img /= 255.0
training_label = np.array(training_label)
len(np.unique(training_label))




#%%

label_to_id = {v : k for k, v in enumerate(np.unique(training_label))}
id_to_label = {v : k for k, v in label_to_id.items()}


training_label_id = np.array([label_to_id[i] for i in training_label])

training_label_id


#%%

model = keras.Sequential()
model.add(Conv2D(16, (3, 3), input_shape = (img_size, img_size, 3), padding = "same", activation = "relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(32, (3, 3), padding = "same", activation = "relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(64, (3, 3), padding = "same", activation = "relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(128, (3, 3), padding = "same", activation = "relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))


model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dense(20, activation = "softmax"))

model.compile(loss = "sparse_categorical_crossentropy", optimizer = 'adam', metrics = ['accuracy'])



model.fit(training_fruit_img, training_label_id, batch_size = 128, epochs = 10, verbose=1)

# create on Colab directory
model.save('model.h5')    
model_file = drive.CreateFile({'title' : 'model.h5'})
model_file.SetContentFile('model.h5')
model_file.Upload()

drive.CreateFile({'id': model_file.get('id')})
   
        #save the weights 
model.save_weights('model_weights.h5')
weights_file = drive.CreateFile({'title' : 'model_weights.h5'})
weights_file.SetContentFile('model_weights.h5')
weights_file.Upload()
drive.CreateFile({'id': weights_file.get('id')})


#%%



#%%



