import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

# База данных
df = pd.read_csv('http://storage.googleapis.com/'
                 'download.tensorflow.org/data/ecg.csv', header=None)
# В массиве labels сохраняем значения последнего столбца (1-здоров, 0-болен)
# В массиве data сохраняем все координаты точек на графике
data = df.iloc[:,:-1].values
labels = df.iloc[:,-1].values

# Готовим данные для обучения и тестирования
train_data, test_data, train_labels, test_labels = (
    train_test_split(data, labels, test_size=0.2, random_state=21))

# Ищем минимум, максимум для обучения
min = tf.reduce_min(train_data)
max = tf.reduce_max(train_data)

# Отформатируем данные, чтобы их значения были от [0 : 1]
train_data = (train_data - min)/(max - min)
test_data = (test_data - min)/(max - min)

train_data = tf.cast(train_data, dtype=tf.float32)
test_data = tf.cast(test_data, dtype=tf.float32)

train_labels = train_labels.astype(bool)
test_labels = test_labels.astype(bool)

normal_train_data = train_data[train_labels]
normal_test_data = test_data[test_labels]

anomalous_train_data = train_data[~train_labels]
anomalous_test_data = test_data[~test_labels]

# визуализация аномального и обычного экг
"""
plt.grid()
plt.plot(np.arange(140), normal_train_data[0])
plt.title("A Normal ECG")
plt.show()

plt.grid()
plt.plot(np.arange(140), anomalous_train_data[0])
plt.title("An Anomalous ECG")
plt.show()
"""

class AnomalyDetector(Model):
  def __init__(self):
    super(AnomalyDetector, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(32, activation="relu"),
      layers.Dense(16, activation="relu"),
      layers.Dense(8, activation="relu")])

    self.decoder = tf.keras.Sequential([
      layers.Dense(16, activation="relu"),
      layers.Dense(32, activation="relu"),
      layers.Dense(140, activation="sigmoid")])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = AnomalyDetector()
autoencoder.compile(optimizer='adam', loss='mae')
history = autoencoder.fit(normal_train_data, normal_train_data,
          epochs=20,
          batch_size=256,
          validation_data=(test_data, test_data),
          shuffle=True)

# Результат работы
encoded_imgs = autoencoder.encoder(normal_test_data).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

plt.plot(normal_test_data[0],'b')
plt.plot(decoded_imgs[0],'r')
plt.fill_between(np.arange(140), decoded_imgs[23], anomalous_test_data[50],
                 color='lightcoral')
plt.legend(labels=["Input", "Reconstruction", "Error"])
plt.show()

