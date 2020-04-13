
# Large amount of credit goes to:
# https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
# which I've used as a reference for this implementation

from __future__ import print_function, division

import os
import cv2
from model import *

# from keras.datasets import mnist
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Embedding
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from functools import partial

import keras.backend as K

import matplotlib.pyplot as plt
import sys
import numpy as np

class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((64, 1, 1, 1)) # default minval=0.0, maxval=1.0 in keras.backend.random_uniform()
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class WGANGP():
    def __init__(self, restore_path=None, post_fix=None):
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 36
        self.latent_dim = 300

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        optimizer = RMSprop(lr=0.00005)

        # Build the generator and critic
       
        self.generator = self.build_2Dgenerator()
        self.critic = self.build_critic()
       
        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------

        # Freeze generator's layers while training critic
        self.generator.trainable = False

        # Image input (real sample)
        real_img = Input(shape=self.img_shape)

        # Noise input
        z_disc = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        # Generate image based of noise (fake sample)
        fake_img = self.generator([z_disc, label])

        # Discriminator determines validity of the real and fake images
        fake = self.critic([fake_img, label])
        valid = self.critic([real_img, label])

        # Construct weighted average between real and fake images (which is a tensor)
        interpolated_img = RandomWeightedAverage()([real_img, fake_img]) # same shape with real_img
        # Determine validity of weighted sample
        validity_interpolated = self.critic([interpolated_img, label])

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                          averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic_model = Model(inputs=[real_img, z_disc, label],
                            outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                              self.wasserstein_loss,
                                              partial_gp_loss],
                                        optimizer=optimizer,
                                        loss_weights=[1, 1, 10])
        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.generator.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(self.latent_dim,))
        label_gen = Input(shape=(1,))
        # Generate images based of noise
        img = self.generator([z_gen, label_gen])
        # Discriminator determines validity
        valid = self.critic([img, label_gen])
        # Defines generator model
        self.generator_model = Model([z_gen, label_gen], valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0] 
		
        gradients_sqr = K.square(gradients)
        
        gradients_sqr_sum = K.sum(gradients_sqr,axis=np.arange(1, len(gradients_sqr.shape))) 
        
        gradient_l2_norm = K.sqrt(gradients_sqr_sum) 
        
        gradient_penalty = K.square(1 - gradient_l2_norm) 
        
        return K.mean(gradient_penalty)


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_2Dgenerator(self):

        model = Sequential()

        model.add(Dense(128 * 8 * 8, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((8, 8, 128)))

        model.add(UpSampling2D()) # 16*16*128
        model.add(Conv2D(128, kernel_size=3, padding="same")) # 16*16*128
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling2D()) # 32*32*128
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling2D())  # 64*64*64
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(UpSampling2D())  # 128*128*64
        model.add(Conv2D(32, kernel_size=3, padding="same")) # 128*128*32
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))

        model.add(Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')  # 0 ~ 9
        # embed label portion into multi-dimension
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))

        # how to make noise with label for cgan :
        model_input = multiply([noise, label_embedding])
        img = model(model_input)  # generator's output

        return Model([noise, label], img)

    def build_critic(self):

        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        odel.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))

        

        img = Input(shape=self.img_shape)
        label = Input(shape=(1,), dtype='int32')
        # embe label portion into the flatted size of img shape
        label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.img_shape))(label))
        flat_img = Flatten()(img)

        # how to make input for discriminator with label for cgan :
        model_input = multiply([flat_img, label_embedding])
        # print("debug1", flat_img) # Tensor("flatten_3/Reshape:0", shape=(?, ?), dtype=float32)
        # print("debug1", label_embedding) # Tensor("flatten_2/Reshape:0", shape=(?, ?), dtype=float32)

        validity = model(model_input)

        return Model([img, label], validity)


    def train(self, X_train, y_train, epochs, batch_size=128, sample_interval=50, save_path=None, img_save_path=None):
        
        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake =  np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1)) # Dummy gt for gradient penalty
        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs, labels = X_train[idx], y_train[idx]
                # Sample generator input
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                # Train the critic
                d_loss = self.critic_model.train_on_batch([imgs, noise, labels],
                                                                [valid, fake, dummy])

            # ---------------------
            #  Train Generator
            # ---------------------
            sampled_labels = np.random.randint(0, self.num_classes, batch_size).reshape(-1, 1)
            g_loss = self.generator_model.train_on_batch([noise, sampled_labels], valid)

            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))
            # print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
            # print (epoch, d_loss, g_loss) # 0 [21.782318, 0.5213826, -0.15711007, 2.1418045] 0.67338276
            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                if img_save_path is not None:
                    self.sample_images(epoch, img_save_path)
                else:
                    self.sample_images(epoch)
                

    def sample_images(self, epoch, save_path=None):
        r, c = 6, 6
        latent_val = self.latent_dim
        fig, axs = plt.subplots(r, c)

        noise = np.random.normal(0, 1, (r * c, latent_val))  # (36, 100)
        sampled_labels = np.arange(0, self.num_classes).reshape(-1, 1)  # (36, 1)

        gen_imgs = self.generator.predict([noise, sampled_labels])  # (36, ...)
        print("gen_imgs", gen_imgs.shape)  # (36, 95, 79, 1)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        cnt = 0
        for i in range(r):

            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:,:], cmap='gray')

                axs[i,j].set_title("Slice: %d" % sampled_labels[cnt])
                axs[i,j].axis('off')
                cnt += 1
        if save_path is not None:
            fig.savefig(save_path+"/%d.png" % epoch)
        else :
            fig.savefig("./images/%d.png" % epoch)
        plt.close()

    def save_model(self, save_path, post_fix):

        generator_save_path = os.path.join(save_path, "gen_model_" + post_fix + ".h5")
        discriminator_save_path = os.path.join(save_path, "disc_model_" + post_fix + ".h5")

        self.generator.save(generator_save_path)
        self.discriminator.save(discriminator_save_path)
        return

    def restore_model(self, save_path, post_fix):

        generator_save_path = os.path.join(save_path, "gen_model_" + post_fix + ".h5")
        discriminator_save_path = os.path.join(save_path, "disc_model_" + post_fix + ".h5")

        optimizer = RMSprop(lr=0.00005)

        # Build and compile the discriminator
        print("[!] Initiate discriminator")
        self.discriminator = load_model(discriminator_save_path)
        self.discriminator.compile(loss=['binary_crossentropy'],
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        print("[!] Initiate generator")
        self.generator = load_model(generator_save_path)
        
        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])  # generator's input : noise + label

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([img, label])  # discriminator's input : img + label

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=['binary_crossentropy'],
                              optimizer=optimizer)

    def generate_imgs(self):

        noise = 0
        label = 0

        r, c = 6, 6
        latent_val = self.latent_dim
        # noise = np.random.normal(0, 1, (r * c, latent_val))  # (36, 100) #  random sample from a disribution; mean : 0, sd : 1
        # noise_ = [np.random.normal(0, 1, (self.latent_dim))]
        # noise = np.concatenate([noise_ for _ in range(self.num_classes)])
        # print("debug", noise.shape) # (36, 300)

        iter = np.array(list(range(-10, 10, 1))) * 0.1
        print("debug", iter)
        for work_ in iter:

            fig, axs = plt.subplots(r, c)

            # noise[:,100] = work_
            # noise = np.random.normal(0, 1, (r * c, latent_val))  # (36, 100)
            noise_ = [np.random.normal(0, 1, (self.latent_dim))]
            noise = np.concatenate([noise_ for _ in range(self.num_classes)])
            sampled_labels = np.arange(0, self.num_classes).reshape(-1, 1)  # (36, 1)

            gen_imgs = self.generator.predict([noise, sampled_labels])  # (36, ...)
            print("gen_imgs", gen_imgs.shape)  # (36, 95, 79, 1)
            print("gen_imgs max, min", np.max(gen_imgs), np.min(gen_imgs))  # 1.0, -1.0

            # Rescale images 0 - 1
            gen_imgs = 0.5 * gen_imgs + 0.5
            # gen_imgs = 127.5 * gen_imgs + 127.5

            # # resize images
            # gen_imgs = np.array([cv2.resize(np.array(x), (128, 128)) for x in gen_imgs]) # (36, 128, 128)
            # print("resized gen_imgs", gen_imgs.shape)
            # gen_imgs = np.expand_dims(gen_imgs, axis=3)

            cnt = 0

            for i in range(r):

                for j in range(c):
                    axs[i, j].imshow(gen_imgs[cnt, :, :, :], cmap='gray')

                    axs[i, j].set_title("Slice: %d" % sampled_labels[cnt])
                    axs[i, j].axis('off')
                    cnt += 1

            # plt.show()
            fig.savefig("working_space/%d.png" % (work_ * 10 + 10))
            plt.close()

        return

    def img_aug_by_generator(self, num_augment):
        """        
        :param save_path: 
        :param post_fix: 
        :param num_augment: 
        :return: totally num_augment*num_classes datas are going to be returned
        """

        bin = []
        for _ in range(num_augment):
            # noise[:,1]+=1
            noise = np.random.normal(0, 1, (self.num_classes, self.latent_dim))  # (36, 100)
            sampled_labels = np.arange(0, self.num_classes).reshape(-1, 1)  # (36, 1)

            gen_imgs = self.generator.predict([noise, sampled_labels])  # (36, ...)
            # print("gen_imgs", gen_imgs.shape)  # (36, 128, 128, 3)

            # Rescale images 0 - 1
            gen_imgs = 127.5 * gen_imgs + 127.5

            # # resize images
            gen_imgs = np.array([cv2.resize(np.array(x), (224, 224)) for x in gen_imgs])
            # print("resized gen_imgs", gen_imgs.shape) # (36, 224, 224, 3)
            # gen_imgs = np.expand_dims(gen_imgs, axis=3)
            bin.append(gen_imgs)

        return np.concatenate(bin)

if __name__ == '__main__':
    wgan = WGANGP()
    wgan.train(epochs=30000, batch_size=32, sample_interval=100)
