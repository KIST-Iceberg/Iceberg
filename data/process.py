"""
Copyright kairos03, dalbom. All Right Reserved.

Data Processing module
"""
import os
import pickle

import numpy as np
import pandas as pd
import cv2

DATA_PATH = 'data/'
PROCESSED_PATH = 'processed/'
ROOT_PATH = DATA_PATH + PROCESSED_PATH
IMAGE_PATH = 'image.pkl'
LABEL_PATH = 'label.pkl'
ANGLE_PATH = 'angle.pkl'


def gen_combined_channel(images):
    """
    Generate 3 channel image from 2 channel with hh, hv
    :param images: images, ndarray, dims like [num of images, width, height, 2]
    :return: new_img: images, ndarray, dims like [num of images, width, height, 3]
    """
    # input check
    assert type(images) == np.ndarray
    assert images.shape[3] == 2

    # get channel
    hh = images[:, :, :, 0]
    hv = images[:, :, :, 1]
    combined = np.multiply(abs(hh), abs(hv))

    new_img = np.stack((hh, hv, combined), axis=3)

    # shape check
    assert new_img.shape[0:3] == images.shape[0:3]
    assert new_img.shape[3] == 3

    # value check
    assert new_img[3, 2, 1, 2] == np.multiply(abs(images[3, 2, 1, 0]), abs(images[3, 2, 1, 1]))

    return new_img


def high_filter_and_gamma_correction(images, length=1):
    """
    High filter and Gamma correction
    :param images: images, ndarray, dims like [num of images, width, height, channel]
    :param length: filter length
    :return: high pass filtered image
    """
    assert type(images) == np.ndarray
    assert type(length) == int

    f = np.fft.fft2(images)
    fshift = np.fft.fftshift(f)

    rows, cols = images.shape[1:3]  # taking the size of the image

    crow, ccol = rows // 2, cols // 2  # center point

    fshift[crow - length:crow + length, ccol - length:ccol + length] = 0
    f_ishift = np.fft.ifftshift(fshift)

    img_back = np.power(np.abs(np.fft.ifft2(f_ishift)), 2)  # Gamma correction

    if np.std(img_back) > 0.05:
        img_back = np.power(img_back, 2)

    img_back = (img_back - np.mean(img_back)) / np.std(img_back)
    img_back = (img_back - np.min(img_back)) / (np.max(img_back) - np.min(img_back))

    assert images.shape == img_back.shape

    return img_back


def rotate_img(images, labels, angles):
    """
    Rotate image in 30 degrees
    :param images: images, ndarray
    :param labels: images labels
    :param angles: images angles
    :return rotated_images: rotated images, ndarray
    :return rotated_labels: augmented label
    :return rotated_angles: augmented angle
    """
    # input check
    assert type(images) == np.ndarray
    assert images.shape[0] == labels.shape[0]
    assert images.shape[0] == angles.shape[0]

    # Testing with combined channel
    rows, cols = images.shape[1:3]

    rotated_images = []
    rotated_labels = []
    rotated_angles = []

    for i in range(images.shape[0]):
        # For 360 degree
        for j in range(12):
            # Rotation matrix
            matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), 30 * j, 1)

            # Apply matrix
            dst = cv2.warpAffine(images[i], matrix, (cols, rows))

            # Fill the blank with original pixels
            rotated = np.where(dst == 0, images[i], dst)

            rotated_images.append(rotated)
            rotated_labels.append(labels[i])
            rotated_angles.append(angles[i])

    rotated_images = np.stack(rotated_images)
    rotated_labels = np.stack(rotated_labels)
    rotated_angles = np.stack(rotated_angles)

    # size check
    assert rotated_images.shape[0] == images.shape[0] * 12
    assert rotated_images.shape[1:] == images.shape[1:]

    assert rotated_images.shape[0] == rotated_labels.shape[0]
    assert rotated_images.shape[0] == rotated_angles.shape[0]
    assert rotated_labels.size == labels.size * 12

    # image, label match check
    assert rotated_labels[128] == labels[128 // 12]
    assert rotated_angles[456] == angles[456 // 12]

    return rotated_images, rotated_labels, rotated_angles


def flip_image(images, labels, angles, direction='both'):
    """
    Flip vertical and horizontal
    :param images: images, ndarray
    :param labels: images labels
    :param angels: images angels
    :param direction: dirrection to flip, one of these 'both', 'vertical' 'horizontal'
    :return flipped_images, ndarray
    :return flipped_labels: augmented label
    :return flipped_angels: augmented angel
    """
    # input check
    assert type(images) == np.ndarray
    assert images.shape[0] == labels.shape[0]
    assert images.shape[0] == angles.shape[0]
    assert direction in ['both', 'vertical', 'horizontal']

    flipped_images = []
    flipped_labels = []
    flipped_angles = []

    for i in range(images.shape[0]):

        items = [images[i]]

        if direction == 'both' or direction == 'vertical':
            fh = cv2.flip(images[i], 0)
            items.append(fh)
        if direction == 'both' or direction == 'horizontal':
            fv = cv2.flip(images[i], 1)
            items.append(fv)

        for item in items:
            flipped_images.append(item)
            flipped_labels.append(labels[i])
            flipped_angles.append(angles[i])

    flipped_images = np.stack(flipped_images)
    flipped_labels = np.stack(flipped_labels)
    flipped_angels = np.stack(flipped_angles)

    # size check
    multiplyer = 3 if direction == "both" else 2
    assert flipped_images.shape[0] == images.shape[0] * multiplyer
    assert flipped_images.shape[1:] == images.shape[1:]

    assert flipped_images.shape[0] == flipped_labels.shape[0]
    assert flipped_images.shape[0] == flipped_angels.shape[0]

    # image, label match check
    assert flipped_labels[128] == labels[128 // multiplyer]
    assert flipped_angels[456] == angles[456 // multiplyer]

    return flipped_images, flipped_labels, flipped_angels


def save_to_pickle(images, labels, angles):
    """
    Make pickle from each data

    :param images: images, ndarray dims like [number of images, width, height, channel]
    :param labels: lagels of images
    :param angles: angles of iamges
    """

    # directory check
    if PROCESSED_PATH[:-1] not in os.listdir(DATA_PATH):
        os.makedirs(PROCESSED_PATH[:-1])

    pickle.dump(images, open(ROOT_PATH + IMAGE_PATH, 'wb'), protocol=4)
    print("IMAGE DUMPED", ROOT_PATH + IMAGE_PATH)
    pickle.dump(labels, open(ROOT_PATH + LABEL_PATH, 'wb'), protocol=4)
    print("LABEL DUMPED", ROOT_PATH + LABEL_PATH)
    pickle.dump(angles, open(ROOT_PATH + ANGLE_PATH, 'wb'), protocol=4)
    print("ANGEL DUMPED", ROOT_PATH + ANGLE_PATH)
    print("Data ALL Saved.")

def load_from_pickle():
    """
    Load pickle data of image, label, angel

    :return: images, labels, angels data; ndarray
    """

    # file check
    flist = os.listdir(ROOT_PATH)
    if not set([IMAGE_PATH, LABEL_PATH, ANGLE_PATH]).issubset(flist):
        raise FileNotFoundError

    images = pickle.load(open(ROOT_PATH + IMAGE_PATH, 'rb'))
    print("IMAGE LOADED", ROOT_PATH + IMAGE_PATH)
    labels = pickle.load(open(ROOT_PATH + LABEL_PATH, 'rb'))
    print("LABEL LOADED", ROOT_PATH + LABEL_PATH)
    angles = pickle.load(open(ROOT_PATH + ANGLE_PATH, 'rb'))
    print("ANGLE LOADED", ROOT_PATH + ANGLE_PATH)
    print("Data ALL Loaded.")

    return images, labels, angles


def pre_process_data():  # TODO Train, Test로 나눠서 가능하도록 변경
    """
    Data Pre-Process

    Cast pd.DataFrame image to np.ndarray,
    Angle's 'na' to 0.0,
    Generate combine channel,
    High filter and Gamma correction

    :returns: processed image, labels, angles
    """
    # load origin data frame
    train = pd.read_json("data/origin/train.json")
    print("Origin Data Loaded.")

    # make list to ndarray
    img_dict = {}

    for band in ["band_1", "band_2"]:
        samples = train.loc[:, band].values
        for i in range(samples.size):
            samples[i] = np.reshape(samples[i], (75, 75))

        img_dict[band] = np.stack(samples)

    origin_images = np.stack((img_dict["band_1"], img_dict["band_2"]), axis=3)

    # convert check
    assert origin_images[720, 64, 32, 0] == train.loc[720, "band_1"][64, 32]
    assert origin_images[123, 6, 71, 1] == train.loc[123, "band_2"][6, 71]
    assert origin_images[123, 6, 71, 1] != train.loc[123, "band_1"][6, 71]
    print("Image to Numpy Done.")

    # labels
    labels = train.is_iceberg.values

    # angles
    angles = train.inc_angle.values
    # convert 'na' to 0.0
    angles = np.asarray([angle if angle != 'na' else 0.0 for angle in angles])

    # generate combined channel
    combined = gen_combined_channel(origin_images)
    print("Combined.", combined.shape)

    # pass high filter
    filtered = high_filter_and_gamma_correction(combined)
    print("Filtered.", filtered.shape)

    return filtered, labels, angles


def argumentate_data(images, labels, angles):
    """
    Data Argumentate

    Add Rotate images with 30 degree,
    Add Flip images with horizontally, vertically or both

    :param images: images data
    :param labels: images labels
    :param angels: images angels:

    :returns: argumentated images, labels, angels
    """
    # rotate
    images, labels, angles = rotate_img(images, labels, angles)
    print("Rotated.", images.shape)

    # flip
    images, labels, angles = flip_image(images, labels, angles, direction='vertical')
    print("Flipped.", images.shape)

    return images, labels, angles


def main():  # TODO Train, Test 받아서 처리 하도록 TODO load, save 처리
    print("Data Processing Start")

    # data pre-processing
    print("Pre-Processing Start")
    images, labels, angles = pre_process_data()

    # data argumentation
    print('Argumentation Start')
    images, labels, angles = argumentate_data(images, labels, angles)

    # save to pickle
    save_to_pickle(images, labels, angles)

    # load from pickle
    i, l, a = load_from_pickle()
    print('image', i)
    print('label', l)
    print('angle', a)

    print("Data Processing Done.")

if __name__ == '__main__':
    main()
