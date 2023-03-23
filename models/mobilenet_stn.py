import keras
from keras.models import load_model
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, Activation, Concatenate

from models.base_model import ClassificationModel
from config import model_config
from models.utiles import get_initial_weights
from models.BilinearInterpolation import BilinearInterpolation


class MobilenetStnModel(ClassificationModel):

    def build_network(self):
        input_layer = Input(self._input_shape)
        sampling_size_from_stn = (self._input_shape[0], self._input_shape[1])
        
        if self._train_from_scratch:
            print("---------------- build mobile network with stn from scratch --------------")
            
            stn_net = MaxPooling2D(pool_size=(2, 2))(input_layer)
            stn_net = Convolution2D(filters=16, kernel_size=5, padding='same', activation='relu', kernel_initializer='he_normal')(stn_net)
            stn_net = MaxPooling2D(pool_size=(2, 2))(stn_net)
            stn_net = Dropout(0.5)(stn_net)
            stn_net = Convolution2D(filters=32, kernel_size=5, padding='same', activation='relu', kernel_initializer='he_normal')(stn_net)
            stn_net = MaxPooling2D(pool_size=(2, 2))(stn_net)
            stn_net = Dropout(0.5)(stn_net)
            
            stn_net = Flatten()(stn_net)
            stn_net = Dense(100)(stn_net)
            stn_net = Activation('relu')(stn_net)
            stn_net = Dropout(0.5)(stn_net)
            transform_weights = get_initial_weights(100)
            stn_net = Dense(6, weights=transform_weights)(stn_net)
            
            stn_output = BilinearInterpolation(sampling_size_from_stn)([input_layer, stn_net])

            model_inception = MobileNetV2(weights = 'imagenet', 
                include_top = False, input_shape = self._input_shape
            )
            model = model_inception(stn_output)

            model = Flatten()(model)
            model = Dense(self._num_class)(model)
            model = Activation('softmax', name='soft_max')(model)
            model = Model(input_layer, model, name='mobilenet_v2')
        # elif model_config["pre_trained_model"] is not None:
        #     print("---------------- load mobile network for train --------------")

        #     model = load_model(model_config["pre_trained_model"], compile=False)
        #     model.summary()

        #     num_layers = len(model.layers)
        #     dense_layer_loc = num_layers - 2
        #     dense_layer = model.layers[dense_layer_loc]
        #     if dense_layer.get_config()['units'] != self._num_class:
        #         x = input_layer
        #         for layer in model.layers[1:2]:
        #             x = layer(x)

        #         x = Flatten()(x)
        #         x = Dense(self._num_class)(x)
        #         x = Activation('softmax', name='soft_max')(x)
        #         model = Model(input_layer, x, name='fine tuning mobilenet_v2')
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
        
        
