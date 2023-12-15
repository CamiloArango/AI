import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import tensorflow_datasets as tfds
import resource
import os

datos, metadatos = tfds.load('mnist', as_supervised=True, with_info=True)

test, train = datos['test'], datos['train']

def normalizar(imagenes, etiquetas):
  imagenes = tf.cast(imagenes, tf.float32)
  imagenes /= 255
  return imagenes, etiquetas

train = train.map(normalizar)
test = test.map(normalizar)

train = train.cache()
test = test.cache()

modelo = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), input_shape=(28,28,1), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=100, activation='relu'),
    tf.keras.layers.Dense (10, activation='softmax')
])

modelo.compile(
    optimizer='Adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

train_quantity = metadatos.splits['train'].num_examples
test_quantity = metadatos.splits['test'].num_examples

TAMANO_LOTE=32

train = train.repeat().shuffle(train_quantity).batch(TAMANO_LOTE)
test = test.batch(TAMANO_LOTE)

historial = modelo.fit(
    train,
    epochs=60,
    steps_per_epoch=math.ceil(train_quantity/TAMANO_LOTE)
)

modelo.save('numeritos.keras')