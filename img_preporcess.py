# Copy Right Kairos03 2017. All Right Reserved.

import pandas as pd
import numpy as np
import cv2
import pickle
import data_input


# generate the training data
# create 3 bands having HH, HV and avg of both
def gen_new_data(data_list, is_flip=False):
    def get_more_images(imgs):
        more_images = []
        vert_flip_imgs = []
        hori_flip_imgs = []

        for i in range(0, imgs.shape[0]):
            a = imgs[i, :, :, 0]
            b = imgs[i, :, :, 1]
            c = imgs[i, :, :, 2]

            av = cv2.flip(a, 1)
            ah = cv2.flip(a, 0)
            bv = cv2.flip(b, 1)
            bh = cv2.flip(b, 0)
            cv = cv2.flip(c, 1)
            ch = cv2.flip(c, 0)

            vert_flip_imgs.append(np.dstack((av, bv, cv)))
            hori_flip_imgs.append(np.dstack((ah, bh, ch)))

        v = np.array(vert_flip_imgs)
        h = np.array(hori_flip_imgs)

        more_images = np.concatenate((imgs, v, h))

        return more_images

    def getHigh(img, length=1):
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)

        rows = np.size(img, 0)  # taking the size of the image
        cols = np.size(img, 1)
        crow, ccol = int(rows / 2), int(cols / 2)

        fshift[crow - length:crow + length, ccol - length:ccol + length] = 0
        f_ishift = np.fft.ifftshift(fshift)

        img_back = np.power(np.abs(np.fft.ifft2(f_ishift)), 2)  ## shift for centering 0.0 (x,y)

        img_back = (img_back - np.mean(img_back)) / np.std(img_back)
        img_back = (img_back - np.min(img_back)) / (np.max(img_back) - np.min(img_back))

        return img_back

    # Generate the training data
    data_list.inc_angle = data_list.inc_angle.replace('na', 0)
    idx_meaningful = np.where(data_list.inc_angle > 0)

    # Create 3 bands having HH, HV and avg of both
    X_band_1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in data_list["band_1"]])
    X_band_2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in data_list["band_2"]])
    X_band_3 = np.array([np.multiply(abs(X_band_1[i, :, :]), abs(X_band_2[i, :, :])) for i in range(len(X_band_1))])

    # Apply GetHigh
    X_band_1 = np.array([getHigh(X_band_1[i, :, :]) for i in range(len(X_band_1))])
    X_band_2 = np.array([getHigh(X_band_2[i, :, :]) for i in range(len(X_band_1))])
    X_band_3 = np.array([getHigh(X_band_3[i, :, :]) for i in range(len(X_band_1))])

    X_train = np.concatenate(
        [X_band_1[:, :, :, np.newaxis], X_band_2[:, :, :, np.newaxis], X_band_3[:, :, :, np.newaxis]],
        # [X_band_1[:, :, :, np.newaxis], X_band_2[:, :, :, np.newaxis], ((X_band_1+X_band_2)/2)[:, :, :, np.newaxis]],
        axis=-1)

    X_train = X_train[idx_meaningful[0], ...]
    Y_train = train['is_iceberg']
    Y_train = Y_train[idx_meaningful[0]]

    # Flips
    if is_flip:
        X_train = get_more_images(X_train)
        Y_train = np.concatenate((Y_train, Y_train, Y_train))

    # one hot
    Y_train = np.eye(2)[np.asarray(Y_train)]

    # index
    index = data_list['id']

    return X_train, Y_train, index


if __name__ == '__main__':
    # load the data set
    train = pd.read_json(data_input.origin_path)
    # test = pd.read_json(data_input.test_path)

    # train
    x, label, idx = gen_new_data(train, is_flip=True)
    # d_test = gen_new_data(test)

    pickle.dump(x, open(data_input.pp_x_path, 'wb'))
    pickle.dump(label, open(data_input.pp_label_path, 'wb'))
    pickle.dump(idx, open(data_input.pp_idx_path, 'wb'))
    # pickle.dump(x, open(data_input.pp_test_path, 'wb'))
    # pickle.dump(idx, open(data_input.pp_test_idx_path, 'wb'))
