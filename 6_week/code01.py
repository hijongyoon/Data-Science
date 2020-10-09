import numpy as np
import cv2


img = cv2.imread("test_img4.jpg", cv2.IMREAD_COLOR)
"""cv2.imshow("Test Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()"""

# cv2.imwrite('original.jpg', img) # 파이참 파일에다가 저장되는 사진

img = cv2.resize(img, (500, 800), interpolation=cv2.INTER_CUBIC)
# 이미지를 일정한 크기로 바꾸기 위해 사용. resize 함수에서 두번째 인자는 (너비, 높이)의 형태이지만 shape로 불러올땐 (높이,너비)의 형태.
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# cv2.imwrite('converted.jpg', img)

from matplotlib import pyplot as plt

plt.imshow(img)
plt.show()

img2 = img.reshape(200,100,3)
flattened_image = img.ravel()
# 이미지를 1차원 배열로 만들어 주는 메소드
img_f32 = np.float32(img)
# 신경망 안에서 이뤄지는 모든 연산은 실수연산이기에 데이터 타입을 바꿔줌 그래야 실수 연산이 이루어질 수 있음
# img_f32 = flattened_image.astype(np.float32) 이것은 위의 표현과 동일

normalized_img32 = img_f32/255.
# 이렇게 배열에다가 255.를 나누면 배열의 모든 원소들에 대하여 /255. 연산을 진행함
zero_centered_img = (img_f32 - np.mean(img_f32))/np.std(img_f32)
# np.mean(배열)은 배열안의 값에 대한 평균을 구함 np.std(img_f32)는 이미지에 대한 표준 편차임

print(zero_centered_img)