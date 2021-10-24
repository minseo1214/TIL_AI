
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
import PIL.Image as Image
from tensorflow import keras
os.listdir()
p = '/content/drive/My Drive'
os.chdir(p)

def resize_img(img,size):
    return img.resize(size)

def load_img(file_path):
  data = []
  print(p + file_path[1:] + '/')
  for f in os.listdir(file_path):
    data.append(resize_img(Image.open(p + file_path[1:] + '/' + f) , (64,64)))
  return data

train_malocclusion = load_img('./치아 폴더/부정교합/train') # img_data_list / element is image not nparray
test_malocclusion = load_img('./치아 폴더/부정교합/test')
val_malocclusion = load_img('./치아 폴더/부정교합/val')

train_occlusion = load_img('./치아 폴더/부정교합이 아닌것/train')
test_occlusion = load_img('./치아 폴더/부정교합이 아닌것/test')
val_occlusion = load_img('./치아 폴더/부정교합이 아닌것/val')
def img_to_array(img):
    return np.array(img, dtype = 'float32')/255.0

#del train_occlusion_arr
#del test_occlusion_arr
#del val_occlusion_arr
#del train_malocclusion_arr
#del test_malocclusion_arr
#del val_malocclusion_arr

train_occlusion_arr,train_occlusion_sol = np.array([img_to_array(occlusion) for occlusion in train_occlusion]),np.array([1]*len(train_occlusion))
test_occlusion_arr,test_occlusion_sol = np.array([img_to_array(occlusion) for occlusion in test_occlusion]),np.array([1]*len(test_occlusion))
val_occlusion_arr,val_occlusion_sol = np.array([img_to_array(occlusion) for occlusion in val_occlusion]),np.array([1]*len(val_occlusion))

train_malocclusion_arr,train_malocclusion_sol = np.array([img_to_array(malocclusion) for malocclusion in train_malocclusion]),np.array([0]*len(train_malocclusion))
test_malocclusion_arr,test_malocclusion_sol = np.array([img_to_array(malocclusion) for malocclusion in test_malocclusion]),np.array([0]*len(test_malocclusion))
val_malocclusion_arr,val_malocclusion_sol = np.array([img_to_array(malocclusion) for malocclusion in val_malocclusion]),np.array([0]*len(val_malocclusion))

train_img,train_sol = np.concatenate((train_occlusion_arr,train_malocclusion_arr)),np.concatenate((train_occlusion_sol,train_malocclusion_sol))
test_img,test_sol = np.concatenate((test_occlusion_arr,test_malocclusion_arr)),np.concatenate((test_occlusion_sol,test_malocclusion_sol))
val_img,val_sol = np.concatenate((val_occlusion_arr,val_malocclusion_arr)),np.concatenate((val_occlusion_sol,val_malocclusion_sol))
# train model
model = keras.models.Sequential()
model.add(keras.layers.BatchNormalization(input_shape=train_img.shape[1:]))
model.add(keras.layers.Conv2D(filters = 64, kernel_size = (5,5),padding = 'same', activation= 'elu', input_shape = (64,64,3)))
model.add(keras.layers.MaxPooling2D(pool_size = (2,2), strides = (2,2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.BatchNormalization(input_shape=train_img.shape[1:]))
model.add(keras.layers.Conv2D(128, (5,5), padding = 'same', activation='elu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.BatchNormalization(input_shape=train_img.shape[1:]))
model.add(keras.layers.Conv2D(256,(5,5), padding = 'same', activation = 'elu'))
model.add(keras.layers.MaxPooling2D(pool_size = (2,2), strides=(2,2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(256,activation = 'elu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(128,activation = 'elu'))
model.add(keras.layers.Dense(18,activation = 'elu'))
model.add(keras.layers.Dense(1,activation = 'sigmoid'))

model.compile(loss='binary_crossentropy', optimizer = keras.optimizers.Adam(lr = 1e-3,) , metrics=['accuracy'])
model.summary()
model.fit(x = train_img, y = train_sol, batch_size = 128, epochs = 30,verbose=1,
          validation_data = (val_img,val_sol))
model.evaluate.(test_img,test_sol,batch_size = 128)
