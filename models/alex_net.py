import keras
from keras.models import load_model
from keras.layers import Dense, Flatten, Activation, InputLayer, Dropout, Input
from keras.models import Model
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D

from models.base_model import ClassificationModel
from config import model_config

class AlexnetModel(ClassificationModel):

    def build_network(self):
        input_layer = Input(shape=self._input_shape)

        if self._train_from_scratch:
            print("---------------- build alex network from scratch --------------")

            x = input_layer
            y = Convolution2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')(x)
            y = Convolution2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu')(y)
            y = MaxPooling2D(pool_size=2, strides=2, padding='same')(y)
            y = Dropout(0.25)(y)

            y = Flatten()(y)
            y = Dense(units=self._num_class, activation='softmax')(y)

            model = Model(input_layer, y, name='alex_net')
        else:
            print("---------------- load alex network for train --------------")

            model = load_model(model_config["pre_trained_model"], compile=False)
            model.summary()

            dense_layer = model.layers[-1]
            if dense_layer.get_config()['units'] != self._num_class:

                y = input_layer
                for layer in model.layers[1:4]:
                    y = layer(y)
                y = Dropout(0.25)(y)

                y = Flatten()(y)
                y = Dense(units=self._num_class, activation='softmax')(y)
                
                model = Model(input_layer, y, name='fine tuning alex_net')

        model.summary()

        model.compile(
            loss=self._loss,
            optimizer=self._optimizer,
            metrics=['accuracy']
        )

        return model