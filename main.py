import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model

dataframe = (
    pd.read_csv(f'http://storage.googleapis.com/'
                'download.tensorflow.org/data/ecg.csv', header=None))

raw_data = dataframe.values
print(dataframe.head())
