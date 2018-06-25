
import pandas as pd
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
from glob import glob

#reading our mini_dataset
from keras.utils.io_utils import HDF5Matrix

'''disease_vec_labels = ['Atelectasis','Cardiomegaly','Consolidation','Edema','Effusion','Emphysema','Fibrosis',
 'Hernia','Infiltration','Mass','Nodule','Pleural_Thickening','Pneumonia','Pneumothorax']
disease_vec=[]
with h5py.File('chest_xray.h5','r') as h5_data:
    all_fields = list(h5_data.keys())

    for c_key in disease_vec_labels:
        disease_vec += [h5_data[c_key][:]]

    disease_vec = np.stack(disease_vec,1)
'''    #print ('Disease Vec:',disease_vec.shape)

disease = 'Atelectasis'
h5_path = '{}_mini.h5'.format(disease)

img_ds = HDF5Matrix(h5_path,'images',normalizer = lambda x:x/127.5-1)
#print (img_ds[:1].shape)

from keras.models import Model,Sequential
from keras.layers import Dense,Flatten,Dropout,Activation,Lambda,Reshape
from keras.layers.convolutional import Conv2D,Deconv2D,ZeroPadding2D,UpSampling2D
from keras.layers import Input,merge
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
import keras.backend as K
from keras.optimizers import Adam


class DCGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(100,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 32 * 32, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((32, 32, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(32, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50,last_model_point=0):

        # Load the dataset
        X_train = img_ds

        # Rescale -1 to 1
        #X_train = X_train / 127.5 - 1.
        #X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images

            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = np.array([X_train[i] for i in idx])

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch,last_model_point)

        self.generator.save('generated_models/Generator_model_{}_{}'.format(disease,epoch+last_model_point+1))
        self.discriminator.save('generated_models/Discriminator_model_{}_{}'.format(disease,epoch+last_model_point+1))


    def save_imgs(self, epoch,last_model_point):
        pass
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='bone')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("generated_images/{}_{}.png".format(disease,epoch+last_model_point))
        plt.close()

from keras.models import load_model

def find_last_model_checkpoint():
    last_model_point=0
    for f in (glob('generated_models/Generator_model_{}_*'.format(disease))):
        file = f.split('/')[-1]
        checkpoint_no = int(file.split('_')[-1])
        if checkpoint_no > last_model_point:
            last_model_point = checkpoint_no

    return int(last_model_point)

if __name__ == '__main__':
    dcgan = DCGAN()
    last_model_point=find_last_model_checkpoint()
    print ("Last checkpoint number = %d" %last_model_point)

    if os.path.exists('generated_models/Generator_model_{}_{}'.format(disease,last_model_point)):
        dcgan.generator = load_model('generated_models/Generator_model_{}_{}'.format(disease,last_model_point))
        dcgan.discriminator = load_model('generated_models/Discriminator_model_{}_{}'.format(disease,last_model_point))
        optimizer = Adam(0.0002, 0.5)
        z = Input(shape=(100,))
        img = dcgan.generator(z)

        # For the combined model we will only train the generator
        dcgan.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = dcgan.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        dcgan.combined = Model(z, valid)
        dcgan.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
    dcgan.train(epochs=2000, batch_size=32, save_interval=100,last_model_point=last_model_point)
