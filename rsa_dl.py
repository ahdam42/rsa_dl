import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import multiprocessing as mp
import numpy as np
from rsa_utils import BIT_NUMBER, BATCH_SIZE, RSA_NUM, RSA_NUM_VEC, vec_to_int, generate_rsa_pairs

keras.backend.set_floatx('float64')

model = keras.Sequential(
    [
        layers.Dense(BIT_NUMBER, activation='relu', input_shape=(BIT_NUMBER,)),
        layers.Dropout(.15),
        layers.Reshape((1, BIT_NUMBER)),
        layers.LSTM(BIT_NUMBER),
        layers.Dropout(.25),
        layers.Dense(BIT_NUMBER, activation='relu'),
        layers.Dropout(.15),
        layers.Dense(BIT_NUMBER // 2, activation='sigmoid'),
        layers.Dropout(.15)
    ]
)

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
model.build()

model.summary();
epochs = 1
while True:
    epochs *= 2
    x_train, y_train = [], []
    pool = mp.Pool(mp.cpu_count())
    results = [pool.apply_async(generate_rsa_pairs) for _ in  range(BATCH_SIZE)]
    pool.close();
    output = [p.get() for p in results]
    for x, y in output:
        x_train.append(x)
        y_train.append(y)

    model.fit(tf.convert_to_tensor(x_train), tf.convert_to_tensor(y_train), epochs = epochs, batch_size = BATCH_SIZE)

    predictedNumber = vec_to_int(model.predict(tf.convert_to_tensor([RSA_NUM_VEC]))[0])

    if (RSA_NUM % predictedNumber) == 0:
        print(predictedNumber)
        model.save("final_model")
        break;


