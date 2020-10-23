import tensorflow as tf
from tensorflow import keras
from lesson_6.HandleInput import *

import numpy as np

from matplotlib import pyplot as plt

class_names = ['cat', 'cow', 'dog', 'pig', 'sheep']
train_features, train_labels, test_features, test_labels = load_all_data()

# 이 단락은 이미지가 제대로 읽혀져왔는지 test 해본 것. -> 이미지가 정상적으로 보이진 않을 것.
plt.figure()
plt.imshow(train_features[0])
plt.colorbar()
plt.grid(False)
plt.show()

plt.figure(figsize=(10, 10))
for i in range(25):  # 총 25장의 이미지를 읽고 제대로 읽혀졌는지 확인 -> 이미지가 정상적으로 보이지 않음.
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_features[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[test_labels[i]])

plt.show()


def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Flatten(input_shape=(IMG_HEIGHT,IMG_WIDTH,NUM_CHANNEL)),  # 다차원 배열을 1차원 배열로 만들어 주는 단계.
        keras.layers.Dense(512, activation=tf.keras.activations.relu),  # 512는 노드의 갯수, 그 다음은 어떤 activation 함수를 사용할 것인가.
        keras.layers.Dense(NUM_CLASS,activation=tf.keras.activations.softmax)  # 출력 노드의 갯수, 출력 layer 에서의 activation 함수.
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(),  # AdapOptimizer 는 learning rate 를 adaptive 하게 조절해주는 gradient descent 알고리즘.
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  # 정답과 출력의 차이. 우리가 직접 one-hot-encode 했다면 sparse 를 붙여줄 필요 없음. classification 에서만 crossentropy
                  # 를 사용하고 아니라면 loss = 'mse' 를 사용.
                  metrics=['accuracy'])

    return model

checkpoint_path = "traing_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create checkpoint call back
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 period=5)

model = create_model()
model.summary()

model.fit(train_features, train_labels, epochs=100,
          validation_data=(test_features, test_labels),
          callbacks= [cp_callback])
