# -*- coding: utf-8 -*-

import random
import numpy as np

import tensorflow as tf
from keras.layers import Input
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import plot_model
# from keras import backend as K
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers



def siamese_model_sider_adr_emb(dim_X=1024,lr=0.0001):

    # Generate the initial random embedding
    embedding_dim = 512
    vocab_size = 18
    std_dev = 0.01 

    adr_term_embeddings_values = np.random.randn(vocab_size, embedding_dim) #np.random.rand(vocab_size, embedding_dim)
    noisy_embeddings = adr_term_embeddings_values + np.random.normal(scale=std_dev, size=adr_term_embeddings_values.shape)

    embedding_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim, 
        embeddings_initializer=tf.keras.initializers.Constant(noisy_embeddings),
        trainable=True
    )

    left_input = Input(shape=(dim_X,))
    right_input = Input(shape=(1,), dtype="int32") 
    # The input is the label of the ADR category, which is subsequently converted into a 512-dimensional vector through the embedding layer


    model1 = tf.keras.models.Sequential([
        tf.keras.layers.Dense(dim_X,input_shape=(dim_X,),  activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(128, activation='relu'),

    ])

    model2 = tf.keras.models.Sequential([
        tf.keras.layers.Dense(512,input_shape=(512,),  activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(128, activation='relu'),
    ])

    right_input_ = embedding_layer(right_input)
    right_input_ = tf.reshape(right_input_, shape=(-1, 512,1))

    encoded_l = model1(left_input)
    encoded_r = model2(right_input_)

    L1 = tf.keras.layers.Lambda (lambda x: K.abs(x[0]-x[1]))([encoded_l, encoded_r])

    L1_D = tf.keras.layers.Dropout(0.1)(L1)
    L2 = tf.keras.layers.Dense(64, activation='relu')(L1_D)
    L2_D = tf.keras.layers.Dropout(0.1)(L2)
    L3 = tf.keras.layers.Dense(32, activation='relu')(L2_D)
    L3_D = tf.keras.layers.Dropout(0.1)(L3)
    L4 = tf.keras.layers.Dense(16, activation='relu')(L3_D)
    L4_D = tf.keras.layers.Dropout(0.1)(L4)
    L5 = tf.keras.layers.Dense(8, activation='relu')(L4_D)

    prediction = tf.keras.layers.Dense(1, activation='sigmoid')(L5)

    siamese_net = tf.keras.Model([left_input, right_input], prediction)

    optimizer= Adam(learning_rate=lr) # 0.001
    siamese_net.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=["accuracy", tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    return siamese_net

def siamese_model_sider_adr_GPTemb(dim_X=1024, adr_term_embeddings=None,lr=0.0001):
    
    if adr_term_embeddings is not None:
        adr_term_embeddings_values = adr_term_embeddings # 18*1536
        embedding_layer = tf.keras.layers.Embedding(input_dim=18, output_dim=1536,
                                                    embeddings_initializer=tf.keras.initializers.Constant(adr_term_embeddings_values), 
                                                    trainable=True)
        
    left_input = Input(shape=(dim_X,))
    right_input = Input(shape=(1,), dtype="int32")


    model1 = tf.keras.models.Sequential([
        tf.keras.layers.Dense(dim_X,input_shape=(dim_X,),  activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
    ])

    model2 = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1536,input_shape=(1536,),  activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation='relu'),
    ])
    
    right_input_ = embedding_layer(right_input)
    right_input_ = tf.reshape(right_input_, shape=(-1, 1536,1)) # (batch_size,512)

    encoded_l = model1(left_input)
    encoded_r = model2(right_input_)

    L1 = tf.keras.layers.Lambda (lambda x: K.abs(x[0]-x[1]))([encoded_l, encoded_r])


    L1_D = tf.keras.layers.Dropout(0.2)(L1)
    L2 = tf.keras.layers.Dense(64, activation='relu')(L1_D)
    L2_D = tf.keras.layers.Dropout(0.2)(L2)
    L3 = tf.keras.layers.Dense(32, activation='relu')(L2_D)
    L3_D = tf.keras.layers.Dropout(0.2)(L3)
    L4 = tf.keras.layers.Dense(16, activation='relu')(L3_D)
    L4_D = tf.keras.layers.Dropout(0.2)(L4)
    L5 = tf.keras.layers.Dense(8, activation='relu')(L4_D)

    prediction = tf.keras.layers.Dense(1, activation='sigmoid')(L5)

    siamese_net = tf.keras.Model([left_input, right_input], prediction)

    optimizer= Adam(learning_rate=lr)
    siamese_net.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=["accuracy", tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])


    return siamese_net
