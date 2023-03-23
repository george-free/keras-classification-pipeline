import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import keras
from keras import initializers
from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, Activation
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

from models.base_model import ClassificationModel
from config import model_config
from models.utiles import get_initial_weights
from models.BilinearInterpolation import BilinearInterpolation

class LightWeightStnModel(ClassificationModel):
    
    def build_network(self):
        input_layer = Input(shape=self._input_shape)
        sampling_size_from_stn = (self._input_shape[0] // 2, self._input_shape[1] // 2)
        
        if self._train_from_scratch:
            print("---------------- build light weight network with stn model from scratch --------------")
            # y = input_layer
            
            # stn part
            stn_net = MaxPooling2D(pool_size=(2, 2))(input_layer)
            stn_net = Convolution2D(filters=20, kernel_size=5, padding='same', activation='relu', kernel_initializer='he_normal')(stn_net)
            stn_net = MaxPooling2D(pool_size=(2, 2))(stn_net)
            # stn_net = Dropout(0.5)(stn_net)
            stn_net = Convolution2D(filters=20, kernel_size=5, padding='same', activation='relu', kernel_initializer='he_normal')(stn_net)
            
            # optional
            # stn_net = Convolution2D(filters=16, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(stn_net)
            # stn_net = MaxPooling2D(pool_size=(2, 2))(stn_net)
            # stn_net = Dropout(0.5)(stn_net)
            
            stn_net = Flatten()(stn_net)
            stn_net = Dense(50)(stn_net)
            stn_net = Activation('relu')(stn_net)
            transform_weights = get_initial_weights(50)
            stn_net = Dense(6, weights=transform_weights)(stn_net)
            
            stn_output = BilinearInterpolation(sampling_size_from_stn)([input_layer, stn_net])
            
            # Block - 1 (map:64, size:32)
            y = Convolution2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(stn_output)
            y = Dropout(0.2)(y)
            y = Convolution2D(filters=64, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
            y = MaxPooling2D(pool_size=2, strides=2, padding='same')(y)
            y = Dropout(0.2)(y)
            
            # Block - 2 (map:128, size:16)
            y = Convolution2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
            y = Dropout(0.2)(y)
            y = Convolution2D(filters=128, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
            y = MaxPooling2D(pool_size=2, strides=2, padding='same')(y)
            y = Dropout(0.2)(y)

            # Block - 3 (map:256, size:8)
            y = Convolution2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
            y = Dropout(0.2)(y)
            y = Convolution2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
            y = Dropout(0.2)(y)
            y = Convolution2D(filters=256, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
            y = MaxPooling2D(pool_size=2, strides=2, padding='same')(y)
            y = Dropout(0.2)(y)
            
            # Block - 4 (map:512, size:4)
            y = Convolution2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
            y = Dropout(0.1)(y)
            y = Convolution2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
            y = MaxPooling2D(pool_size=2, strides=2, padding='same')(y)
            y = Dropout(0.1)(y)
            
            y = Convolution2D(filters=512, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='he_normal')(y)
            y = Dropout(0.1)(y)

            # feature embedding
            embedding = GlobalAveragePooling2D(name='embedding')(y)
            
            y_out = Dense(units=self._num_class, activation='softmax', kernel_initializer='he_normal', name='y_out')(embedding)
            
            model = Model(inputs=input_layer, outputs=y_out, name='light_weight_model')
        
        # elif model_config["pre_trained_model"] is not None:
        #     print("---------------- load light weight network for train --------------")

        #     model = load_model(model_config["pre_trained_model"])
        #     model.summary()
            
        #     x = input_layer
        #     for layer in model.layers[1:-1]:
        #         x = layer(x)
            
        #     y_out = Dense(units=self._num_class, activation='softmax', kernel_initializer='he_normal',name='y_out')(x)
        #     model = Model(input_layer, y_out, name='fine tuning light_weight_model')
        # else:
        #     raise Exception("{} trained model - {} load error".format(
        #         model_config["mobile_name"],
        #         model_config["pre_trained_model"]
        #     ))

        model.summary()
        
        model.compile(
            loss=self._loss,
            optimizer=self._optimizer,
            metrics=['accuracy']
        )

        return model