import tensorflow as tf
from tensorflow import keras
from lesson_6.HandleInput import *

import numpy as np

from matplotlib import pyplot as plt

print(tf.__version__)


def create_model():
    model = keras.Sequential([
        keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same',
                            input_shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNEL)),
        # 사이즈가 3인 커널이 총 32개. padding = 'same' 은 padding 을 하겠다는 의미. input_shape 는 첫번째 layer 한테만 전달.
        keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        # 보통 pool_size 가 2라면 strides 도 2로 따라감. padding = 'same' 을 전달해줌으로써 홀수일 경우 padding 을 해 1/2로 줄여질 수 있도록 함.
        keras.layers.Conv2D(64, kernel_size=3, activation='relu', padding='same'),
        keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        keras.layers.Conv2D(128, kernel_size=(2, 2), activation='relu', padding='same'),
        keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        keras.layers.Flatten(),
        keras.layers.Dense(512, activation=tf.nn.relu),
        keras.layers.Dropout(0.2),
        # Dropout 은 overfitting 을 억제하기 위한 방법. 오로지 training 할 때만 작동. test 할 때는 작동하지 않음.
        keras.layers.Dense(NUM_CLASS, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    return model


def train(model, train_features, train_labels, val_features, val_labels):
    checkpoint_path = "training_cnn/cp-{epoch:04d}.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1,
                                                     period=5)
    model.fit(train_features, train_labels, epochs=50,
              velidation_data=(val_features, val_labels),
              claabacks=[cp_callback])


def train_from_scratch():
    class_names = ['cat', 'cow', 'dog', 'pig', 'sheep']
    train_features, train_labels, test_features, test_labels = load_all_data()
    model = create_model()
    train(model, train_features, train_labels, test_features, test_labels)

    test_loss, test_acc = model.evaluate(test_features, test_labels)
    print('Test accuracy:', test_acc)

    predictions = model.predict(test_features)
    print(predictions[0])
    print(np.argmax(predictions[0]))
    print(test_labels[0])


def load_weights_and_predict():
    model = create_model()
    model.load_weights('training_1/cp.ckpt')

    my_test_img = load_image('test_image01.jpeg')
    my_test_img = np.reshape(my_test_img, [1, IMG_HEIGHT, IMG_WIDTH, 3])
    print(my_test_img.shape)
    my_prediction = model.predict(my_test_img)
    print(my_prediction[0])
    print(np.argmax(my_prediction[0]))


if __name__ == '__main__':
    train_from_scratch()
    # load_weights_and_predict()