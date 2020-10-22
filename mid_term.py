import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

raw_dataset = pd.read_csv('insurance.csv', na_values='?', sep=',', skipinitialspace=True, header=0)
dataset = raw_dataset.copy()
# print(dataset.tail())
# print(dataset.shape)
# dataset['region'] = dataset['region'].map({'southwest': 1, 'southeast': 2, 'northwest': 3, 'northeast': 4})
dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')  # 이렇게 하면 one-hot-encoding 이 자동으로 됨.
# print(dataset.tail())
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('charges')
test_labels = test_features.pop('charges')
# print(train_features.columns)
#
train_mean = train_features.mean(axis=0)
train_std = train_features.std(axis=0)
# print(train_mean)
# print(train_std)

normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))
print(normalizer.mean.numpy())


def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,  # 첫 layer 는 normalization layer
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)  # 출력은 charges 하나.
    ])
    model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(0.001))
    return model


dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()

history = dnn_model.fit(
    train_features, train_labels,
    validation_split=0.2,
    verbose=0, epochs=100)


def plot_loss(pre_history):
    plt.plot(pre_history.history['loss'], label='loss')
    plt.plot(pre_history.history['val_loss'], label='val_loss')
    plt.ylim([0,20000])
    plt.xlabel('Epoch')
    plt.ylabel('Error [charges]')
    plt.legend()
    plt.grid(True)


plot_loss(history)  # 예측된 평균 오차.
plt.show()

test_results = dnn_model.evaluate(test_features, test_labels, verbose=0)
print(test_results)

test_predictions = dnn_model.predict(test_features).flatten()  # test data 를 예측.

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [charges]')
plt.ylabel('Predictions [charges]')
lims = [0, 50000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()

error = test_predictions - test_labels  # 에러 계산.
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [charges]')
_ = plt.ylabel('Count')
plt.show()