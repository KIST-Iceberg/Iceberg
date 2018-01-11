# Copyright Kairos03 2018. All Right Reserved.
"""
pre-trained vgg16 keras model
"""
import pandas as pd
import numpy as np

from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Input, concatenate, GlobalMaxPooling2D

from keras.applications import VGG16, VGG19, Xception, ResNet50, InceptionResNetV2
from keras.layers import Input

# from data import data_input

input_tensor = Input(shape=(75, 75, 3))
factory = {
    'vgg16': lambda: VGG16(include_top=False, input_shape=(75, 75, 3), weights='imagenet'),
    'vgg19': lambda: VGG19(include_top=False, input_shape=(75, 75, 3), weights='imagenet'),
    'xception': lambda: Xception(include_top=False, input_shape=(75, 75, 3), weights='imagenet'),
    'InceptionResNetV2': lambda: InceptionResNetV2(include_top=False, input_tensor=input_tensor, weights='imagenet')
}

def get_model(name='simple',train_base=True,use_angle=False,dropout=0.5,layers=(512,256)):
    base = factory[name]()
    inputs = [base.input]
    x = GlobalMaxPooling2D()(base.output)

    if use_angle:
        angle_in = Input(shape=(1,))
        angle_x = Dense(1, activation='relu')(angle_in)
        inputs.append(angle_in)
        x = concatenate([x, angle_x])

    for l_sz in layers:
        x = Dense(l_sz, activation='relu')(x)
        x = Dropout(dropout)(x)

    x = Dense(1, activation='sigmoid')(x)

    for l in base.layers:
        l.trainable = train_base

    return Model(inputs=inputs, outputs=x)

def get_data(data, is_test=False):
    """
    """
    b1 = np.array(data["band_1"].values.tolist()).reshape(-1, 75, 75, 1)
    b2 = np.array(data["band_2"].values.tolist()).reshape(-1, 75, 75, 1)
    b3 = abs(b1) + abs(b2)

    X = np.concatenate([b1, b2, b3], axis=3)
    Y = np.array(data['is_iceberg']) if not is_test else None
    angle = np.array(pd.to_numeric(data['inc_angle'], errors='coerce').fillna(0))

    return X, Y, angle

train = pd.read_json('data/origin/train.json')
test  = pd.read_json("data/origin/test.json")

keys = list(factory.keys())
keys.sort()
for model_name in keys:
    print(model_name)
    # train data
    X, Y, angle = get_data(train)

    # model
    model = get_model(model_name, train_base=False, use_angle=True)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])

    # fit
    history = model.fit([X, angle], Y, shuffle=True, verbose=1, epochs=100)

    # submit
    X, Y, angle = get_data(test, is_test=True)
    predicted_test = model.predict([X, angle])
    
    print(predicted_test)

    submission = pd.DataFrame()
    submission['id'] = test['id']
    submission['is_iceberg'] = predicted_test.reshape((predicted_test.shape[0]))
    submission.to_csv(model_name+'_sub.csv', index=False)
