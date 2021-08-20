
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import initializers
import cv2


# Let Keras know that we are using tensorflow as our backend engine
os.environ["KERAS_BACKEND"] = "tensorflow"
# To make sure that we can reproduce the experiment and get the same results
np.random.seed(10)
# The dimension of our random noise vector.
random_dim = 100

def load_minst_data():
    # load the data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # покажем что накачали
    print(
        f'x_train length {len(x_train)} y_train length {len(y_train)} x_test length {len(x_test)} y_test length {len(y_test)}')
    print(f'x- train 1 shape {x_train[1].shape}')
    h_pic = x_train[0].copy()
    v_pic = np.zeros((1, 28*50))
    # cv2.imshow('v_pic', v_pic)
    # cv2.waitKey()
    j=1
    for i in range(19):
        h_pic = x_train[i*j]
        for j in range(49):
            h_pic = np.hstack((h_pic, x_train[j*i]))
        # v_pic = h_pic
        # cv2.imshow('h_pic', h_pic)
        # cv2.waitKey()
        # print(f'v_pic.shape {v_pic.shape} i== {i}')
        v_pic = np.vstack((v_pic, h_pic))
        # cv2.imshow('v_pic'+ str(i), v_pic)
        # cv2.waitKey()
    print(f'y_train[0] {y_train[0]}')
    print(f'y_train[1] {y_train[1]}')
    print(f'y_train[0-30] {y_train[0:30]}')

    for i in range (10):
        print(f'y_train {y_train[i*10:(i*10)+10]} :{i}')
    cv2.imshow('X _tr', v_pic)
    cv2.waitKey(1)
    # y_train - это аннотации


    # normalize our inputs to be in the range[-1, 1]
    x_train = (x_train.astype(np.float32) - 127.5)/127.5
    # convert x_train with a shape of (60000, 28, 28) to (60000, 784) so we have
    # 784 columns per row
    x_train = x_train.reshape(60000, 784)
    # xx = input('Press to continue.')
    return (x_train, y_train, x_test, y_test)

def get_optimizer():
    return Adam(lr=0.0002, beta_1=0.5)

def get_generator(optimizer):
    generator = Sequential()
    generator.add(Dense(256, input_dim=random_dim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(512))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(1024))
    generator.add(LeakyReLU(0.2))

    generator.add(Dense(10000, activation='tanh'))
    generator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return generator

def get_discriminator(optimizer):
    discriminator = Sequential()
    discriminator.add(Dense(10000, input_dim=10000, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(1000))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(256))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))

    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
    return discriminator
