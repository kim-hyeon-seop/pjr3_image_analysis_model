from PIL import Image # pillow 설치
import glob
import numpy as np
from sklearn.model_selection import train_test_split

img_dir = '../datasets/UTKFace/'
categories = ['male', 'female']

image_w = 64
image_h = 64

pixel = image_h * image_w * 3
X = []
Y = []
files = None
for idx, category in enumerate(categories):
    # ../datasets/cat_dog/train/cat*.jpg
    files = glob.glob(img_dir + '*' + category + '*.jpg')
    for i, f in enumerate(files):
       try:
            img = Image.open(f)
            img = img.convert('RGB')
            img = img.resize((image_w, image_h))
            data = np.asarray(img)
            X.append(data)
            Y.append(idx)
            if i % 300 == 0:
                print(category, ':', f)
       except:
           print(category, i, '')
X = np.array(X)
Y = np.array(Y)
X = X / 200
print(X[0])
print(Y[:5])
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1)
xy = (X_train, X_test, Y_train, Y_test)
np.save('../datasets/binary_image_data.npy', xy)

# -------------------------------------------------------
# # 샘이 주신코드 수정중입니다.`~~~~~~~~~~~~~~~~~~~~~~~~
# paths = []
# for i in range(117):
#     temp = glob.glob('../datasets/for_classification/for_classification/UTKFace/UTKFace/{}_0*'.format(i))
#     for j in range(len(temp)):
#         temp[j] = temp[j].replace('{}_0'.format(i), '{}_male'.format(i))
#     paths = paths + temp
# print(paths)
#
# paths = []
# for i in range(117):
#     temp = glob.glob('../datasets/for_classification/for_classification/UTKFace/UTKFace/{}_1*'.format(i))
#     for j in range(len(temp)):
#         temp[j] = temp[j].replace('{}_0'.format(i), '{}_female'.format(i))
#     paths = paths + temp
# print(paths)
#
# #
# #
# #
# # paths0 = []
# # for i in range(117):
# #     temp = glob.glob('../datasets/for_classification/for_classification/UTKFace/{}_0*'.format(i))
# #     for j in range(len(temp)):
# #         temp[j] = temp[j].replace('{}_0'.format(i), '{}_male'.format(i))
# #     paths0 = paths0 + temp
# # print(paths0)
# #
# # paths1 = []
# # for i in range(117):
# #     temp = glob.glob('../datasets/for_classification/for_classification/UTKFace/{}_1*'.format(i))
# #     for j in range(len(temp)):
# #         temp[j] = temp[j].replace('{}_1'.format(i), '{}_female'.format(i))
# #     paths = paths1 + temp
# # print(paths1)
# #
# # exit()
# # -------------------------
