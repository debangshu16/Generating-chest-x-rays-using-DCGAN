import keras.backend as K
import matplotlib.pyplot as plt
from keras.models import load_model


'''generator = load_model('generated_models/Generator_model_3000')

noise = np.random.normal(0,1,(1,100))
generated_img = generator.predict(noise)

print (generated_img.shape)
'''

'''fig,axs = plt.subplots(1,1,figsize=(12,8))
axs.imshow(generated_img[0,:,:,0],cmap='bone')
plt.show()'''

import pandas as pd
import os
import h5py
from keras.utils.io_utils import HDF5Matrix
import numpy as np

disease_vec_labels = ['Atelectasis','Cardiomegaly','Consolidation','Edema','Effusion','Emphysema','Fibrosis',
 'Hernia','Infiltration','Mass','Nodule','Pleural_Thickening','Pneumonia','Pneumothorax']
disease_vec = []
with h5py.File('chest_xray.h5','r') as h_data:
    all_fields = list(h_data.keys())
    for c_key in all_fields:
        print (c_key,h_data[c_key].shape)
    for c_key in disease_vec_labels:
        disease_vec+=[h_data[c_key][:]]

    disease_vec = np.stack(disease_vec,1)
    print ('Disease_vec',disease_vec.shape)

img_ds = HDF5Matrix('chest_xray.h5','images')
split_index = int(0.8 * disease_vec.shape[0])
train_ds = HDF5Matrix('chest_xray.h5','images',end=split_index)
test_ds = HDF5Matrix('chest_xray.h5','images',start=split_index)

train_vec = disease_vec[0:split_index]
test_vec = disease_vec[split_index:]
print('Train Shape', train_ds.shape, 'test shape', test_ds.shape)

from keras.applications.mobilenet import MobileNet
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, BatchNormalization, AveragePooling2D
raw_model = MobileNet(input_shape=(None, None, 1), include_top = False, weights = None)
full_model = Sequential()
full_model.add(AveragePooling2D((2,2), input_shape = img_ds.shape[1:]))
full_model.add(BatchNormalization())
full_model.add(raw_model)
full_model.add(Flatten())
full_model.add(Dropout(0.5))
full_model.add(Dense(64))
full_model.add(Dense(disease_vec.shape[1], activation = 'sigmoid'))
full_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['acc'])
full_model.summary()

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau,TensorBoard
from time import time
file_path="weights.best.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=3)
tensorboard = TensorBoard(log_dir = "logs/{}".format(time()))
callbacks_list = [checkpoint, early,tensorboard] #early


full_model.fit(train_ds,train_vec,validation_data=(test_ds,test_vec),epochs=5,verbose=True,shuffle='batch',
callbacks=callbacks_list)
