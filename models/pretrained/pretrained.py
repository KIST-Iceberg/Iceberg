# Copyright Kairos03 2018. All Right Reserved.
"""
pre-trained vgg16 keras model
"""
import sys
import os

import pandas as pd
import numpy as np

from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Input, concatenate, GlobalMaxPooling2D

from keras.applications import VGG16, VGG19, Xception, ResNet50, InceptionResNetV2, MobileNet, InceptionV3
from keras.layers import Input

sys.path.append(os.path.abspath(os.path.join('.')))
from data import process

TOTAL_EPOCH = 300

# from data import data_input
input_tensor = Input(shape=(75, 75, 3))
factory = {
    # 'vgg16': lambda: VGG16(include_top=False, input_tensor=input_tensor, weights='imagenet'),
    'vgg19': lambda: VGG19(include_top=False, input_tensor=input_tensor, weights='imagenet'),
    'xception': lambda: Xception(include_top=False, input_tensor=input_tensor, weights='imagenet'),
    'InceptionV3': lambda: InceptionV3(include_top=False, input_tensor=input_tensor, weights='imagenet'),
    'InceptionResNetV2': lambda: InceptionResNetV2(include_top=False, input_tensor=input_tensor, weights='imagenet')
}

def get_model(name='simple',train_base=True,use_angle=False,dropout=0.8,layers=(512,256)):

    base = factory[name]()
    inputs = [base.input]
    x = GlobalMaxPooling2D()(base.output)

    if use_angle:
        angle_in = Input(shape=(5,))
        angle_x = Dense(5, activation='relu')(angle_in)
        inputs.append(angle_in)
        x = concatenate([x, angle_x])

    for l_sz in layers:
        x = Dense(l_sz, activation='relu')(x)
        x = Dropout(dropout)(x)

    x = Dense(1, activation='sigmoid')(x)

    for l in base.layers:
        l.trainable = train_base

    return Model(inputs=inputs, outputs=x)

def train():
    # train = pd.read_json('data/origin/train.json')
    test  = pd.read_json("data/origin/test.json")

    keys = list(factory.keys())
    keys.sort()

    # train data
    X, Y, angle = process.load_from_pickle()
    T_X, T_Y, T_angle = process.load_from_pickle(is_test=True)

    for model_name in keys:
        print(model_name)
        
        # model
        model = get_model(model_name, train_base=False, use_angle=True)
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4), metrics=['accuracy'])

        # fit
        history = model.fit([X, angle], Y, shuffle=True, verbose=1, epochs=TOTAL_EPOCH)

        # submit
        predicted_test = model.predict([T_X, T_angle])
        
        print(predicted_test)

        submission = pd.DataFrame()
        submission['id'] = test['id']
        submission['is_iceberg'] = predicted_test.reshape((predicted_test.shape[0]))
        submission.to_csv("submissions/"+model_name+'_ep_'+str(TOTAL_EPOCH)+'.csv', index=False)


def ensemble():
    # ensemble
    print('ENSEMBLE')
    files = os.listdir('submissions')
    subs = []
    for file in files:
        path = 'submissions/'+file
        subs.append(pd.read_csv(path))
        print('[READ] ', 'submissions/'+file)

    final = pd.DataFrame()
    final['id'] = subs[0]['id']
    final['is_iceberg'] = np.exp(np.mean(
        [sub['is_iceberg'].apply(lambda x: np.log(x)) for sub in subs], axis=0))
    final.to_csv('submissions/ensamble.csv', index=False, float_format='%.6f')

    del files


if __name__ == '__main__':
    # train()
    ensemble()
