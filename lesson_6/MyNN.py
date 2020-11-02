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
        keras.layers.Flatten(input_shape=(IMG_HEIGHT, IMG_WIDTH, NUM_CHANNEL)),  # 다차원 배열을 1차원 배열로 만들어 주는 단계.
        keras.layers.Dense(512, activation=tf.keras.activations.relu),  # 512는 노드의 갯수, 그 다음은 어떤 activation 함수를 사용할 것인가.
        keras.layers.Dense(NUM_CLASS, activation=tf.keras.activations.softmax)  # 출력 노드의 갯수, 출력 layer 에서의 activation 함수.
    ])

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  # AdapOptimizer 는 learning rate 를 adaptive 하게 조절해주는 gradient descent 알고리즘.
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
          callbacks=[cp_callback])

model.save_weights('training/final_weight.ckpt')  # weight 를 따로 저장. -> 새로 트레이닝.
# model.load_weights('./training/final_weight.ckpt')  # 저장된 weight 를 불러옴. -> 새로 트레이닝이 아닌 트레이닝 된 것을 가져옴.

test_loss, test_acc = model.evaluate(test_features, test_labels)
# test 데이터를 이용해서 학습된 모델을 평가해주는 함수.
print('Test accuracy:', test_acc)

predictions = model.predict(test_features)
# 데이터의 예상 값을 출력 함. 그래서 정답률이 중요한것이 아니기에 이미지만 주면 됨.
print(predictions[0])
print(np.argmax(predictions[0]))  # 최대값의 인덱스를 구해줌.
print(test_labels[0])

my_test_img = load_image('test_image01.jpeg')
my_test_img = (np.expand_dims(my_test_img, 0))  # 3차원 배열인 my_test_img 를 4차원 배열로 만듦.
print(my_test_img.shape)
my_prediction = model.predict(my_test_img)
print(my_prediction[0])
print(np.argmax(my_prediction[0]))

# def denormalize(img):
#     img = img - np.main(img)
#     img = img * 255 / np.max(img)
#     return img.astype(np.uint8)
#
#
# def plot_image(i, predictions_array, true_label, img):
#     predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
#     plt.grid(False)
#     plt.xticks([])
#     plt.yticks([])
#
#     plt.imshow(denormalize(img))
#
#     predicted_label = np.argmax(predictions_array)
#     if predicted_label == true_label:
#         color = 'blue'
#     else:
#         color = 'red'
#
#     plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
#                                          100 * np.max(predictions_array),
#                                          class_names[true_label]), color=color)
