import numpy as np
import os
import cv2

IMG_HEIGHT = 40
IMG_WIDTH = 60
NUM_CHANNEL = 3  # color image 이기 떄문에 3(R,G,B)
NUM_CLASS = 5  # 5종류로 분류하기 위함.
IMAGE_DIR_BASE = '../animal_images'


def load_image(addr):
    img = cv2.imread(addr)
    # img = cv2.imread(addr, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = (img - np.mean(img)) / np.std(img)
    return img


def load_data_set(set_name):  # set name is either 'train' or 'test'
    data_set_dir = os.path.join(IMAGE_DIR_BASE, set_name)  # os.path.join 을 통해 경로를 합칠 수 있음. ex) ../animal_images/train
    image_dir_list = os.listdir(data_set_dir)  # 파일에 있는 목록들을 list 형식처럼 뽑아주는것이 os.listdir -> ['cat','pig','dog',
    # 'sheep', 'cow']
    image_dir_list.sort()  # 정렬을 반드시 안해도 되지만 하면 좋음(실수 방지)
    features = []  # 각 폴더에 있는(cat, pig...) 사진들을 모두 읽어와서 저장.
    labels = []  # 각 이미지 마다 이것이 cat 인지 pig 인지 판별하기 위한 label 들을 저장하는 list 0-cat 1-cow 2-dog 3-pig 4-sheep
    # 밑에 3 for 문들은 모두 동일 내용이며 features 안에는 사진을 넣고 labels 안에는 label 을 넣은 역할을 함.

    # for dir_name in image_dir_list:
    #     image_list = os.listdir(os.path.join(data_set_dir, dir_name))  # ../animal_images/train/cat 과 같은 경로에 있는 이미지들을저장.
    #     for file_name in image_list:
    #         image = load_image(os.path.join(data_set_dir, dir_name, file_name))  # ../animal_images/train/cat/image5 같은 경로.
    #         features.append(image)
    #         if 'cat' in dir_name:
    #             labels.append(0)
    #         elif 'cow' in dir_name:
    #             labels.append(1)
    #         elif 'dog' in dir_name:
    #             labels.append(2)
    #         elif 'pig' in dir_name:
    #             labels.append(3)
    #         elif 'sheep' in dir_name:
    #             labels.append(4)
    #         else:
    #             print('something wrong')

    # for cls_index in range(5):  # 위의 주석 for 문과 동일 내용.
    #     image_list = os.listdir(os.path.join(data_set_dir, image_dir_list[cls_index]))
    #     for file_name in image_list:
    #         image = load_image(os.path.join(data_set_dir, image_dir_list[cls_index], file_name))
    #         features.append(image)
    #         labels.append(cls_index)
    #
    for cls_index, dir_name in enumerate(image_dir_list):  # 위의 주석 for 문과 동일.
        image_list = os.listdir(os.path.join(data_set_dir, dir_name))
        for file_name in image_list:
            if 'png' in file_name or 'jpg' in file_name or 'jpeg' in file_name:  # 이미지 파일이 아닌 파일이 섞일 까봐 넣은 조건.
                image = load_image(os.path.join(data_set_dir, dir_name, file_name))
                features.append(image)
                labels.append(cls_index)

    idxs = np.arange(0, len(features))
    np.random.shuffle(idxs)  # 데이터를 랜덤하게 섞음. -> GradientDescentMethod 를 사용하기 위함.
    features = np.array(features)
    labels = np.array(labels)
    shuf_features = features[idxs]  # features 배열 안에 아까 랜덤하게 만든 idxs 를 넣으면 idxs 순 대로 랜덤하게 셔플 됨.
    shuf_labels = labels[idxs]  # features 와 label 를 동일한 랜덤 순으로 셔플 하는 이유는 둘을 따로 랜덤 셔플하게 되면
    # 이미지와 label 이 안맞을 수 도 있기 때문.
    return shuf_features, shuf_labels


def load_all_data():
    train_images, train_labels = load_data_set('train')
    test_images, test_labels = load_data_set('test')
    return train_images, train_labels, test_images, test_labels


if __name__ == '__main__':
    train_images, train_labels, test_images, test_labels = load_all_data()
    print(train_images.shape)