import keras
import numpy as np
from fer import dataprocessing
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adadelta
from keras import backend

backend.set_image_dim_ordering('th')


def ConvertTo3DVolume(data):
    img_rows, img_cols = 48, 48
    test_set_x = np.asarray(data)
    test_set_x = test_set_x.reshape(test_set_x.shape[0], img_rows, img_cols)
    test_set_x = test_set_x.reshape(test_set_x.shape[0], 1, img_rows, img_cols)
    test_set_x = test_set_x.astype('float32')
    return test_set_x


def predict_prob(number, test_set_x, model):
    toreturn = []
    for data5 in test_set_x:
        if number == 0:
            toreturn.append(dataprocessing.Flip(data5))
        elif number == 1:
            toreturn.append(dataprocessing.Roated15Left(data5))
        elif number == 2:
            toreturn.append(dataprocessing.Roated15Right(data5))
        elif number == 3:
            toreturn.append(dataprocessing.shiftedUp20(data5))
        elif number == 4:
            toreturn.append(dataprocessing.shiftedDown20(data5))
        elif number == 5:
            toreturn.append(dataprocessing.shiftedLeft20(data5))
        elif number == 6:
            toreturn.append(dataprocessing.shiftedRight20(data5))
        elif number == 7:
            toreturn.append(data5)
    toreturn = ConvertTo3DVolume(toreturn)
    proba = model.predict_proba(toreturn)
    return proba


def model_generate():
    img_rows, img_cols = 48, 48

    model = Sequential()
    model.add(Convolution2D(64, 5, 5, border_mode='valid',
                            input_shape=(1, img_rows, img_cols)))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(2, 2), dim_ordering='th'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='th'))
    model.add(Convolution2D(64, 3, 3))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='th'))
    model.add(Convolution2D(64, 3, 3))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
    model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='th'))
    model.add(Convolution2D(128, 3, 3))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='th'))
    model.add(Convolution2D(128, 3, 3))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))

    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), dim_ordering='th'))
    model.add(keras.layers.convolutional.AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
    model.add(Dropout(0.2))
    model.add(Dense(1024))
    model.add(keras.layers.advanced_activations.PReLU(init='zero', weights=None))
    model.add(Dropout(0.2))

    model.add(Dense(7))
    model.add(Activation('softmax'))

    ada = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',
                  optimizer=ada,
                  metrics=['accuracy'])
    model.summary()
    return model


model = model_generate()
filepath = r'E:\Model.547-0.6654.hdf5'
model.load_weights(filepath)

test_set_x, test_set_y = dataprocessing.load_test_data()
# file_names, test_set_x = dataprocessing.read_single_image(r'C:\Users\Scott\Desktop\report\32609443\face2.png')

proba = predict_prob(0, test_set_x, model)
proba1 = predict_prob(1, test_set_x, model)
proba2 = predict_prob(2, test_set_x, model)
proba3 = predict_prob(3, test_set_x, model)
proba4 = predict_prob(4, test_set_x, model)
proba5 = predict_prob(5, test_set_x, model)
proba6 = predict_prob(6, test_set_x, model)
proba7 = predict_prob(7, test_set_x, model)
out = []
for row in zip(proba, proba1, proba2, proba3, proba4, proba5, proba6, proba7):
    a = np.argmax(np.array(row).mean(axis=0))
    out.append(a)

out = np.array(out)
test_set_y = np.array(test_set_y)
c = np.sum(out == test_set_y)
print("Acc:" + str((float(c) / len(out))))
# print({face: emotion for face, emotion in zip(file_names, out)})
