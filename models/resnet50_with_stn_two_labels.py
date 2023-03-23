import keras
from keras.models import load_model
from keras.applications.resnet50 import ResNet50
from keras.applications import ResNet50V2
from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D, Activation, Concatenate

from models.base_model import ClassificationModel
from config import model_config
from models.utiles import get_initial_weights
from models.BilinearInterpolation import BilinearInterpolation

class Resnet50StnModel2Labels(ClassificationModel):
    def build_network(self):
        input_layer = Input(self._input_shape)
        sampling_size_from_stn = (self._input_shape[0] // 2, self._input_shape[1] // 2)
        
        if self._train_from_scratch:
            print("---------------- build resnet50 with stn network from scratch --------------")

            # stn part
            stn_net = MaxPooling2D(pool_size=(2, 2))(input_layer)
            stn_net = Convolution2D(filters=20, kernel_size=5, padding='same', activation='relu', kernel_initializer='he_normal')(stn_net)
            stn_net = MaxPooling2D(pool_size=(2, 2))(stn_net)
            stn_net = Dropout(0.5)(stn_net)
            stn_net = Convolution2D(filters=20, kernel_size=5, padding='same', activation='relu', kernel_initializer='he_normal')(stn_net)
            stn_net = MaxPooling2D(pool_size=(2, 2))(stn_net)
            stn_net = Dropout(0.5)(stn_net)
            
            stn_net = Flatten()(stn_net)
            stn_net = Dense(100)(stn_net)
            stn_net = Activation('relu')(stn_net)
            stn_net = Dropout(0.5)(stn_net)
            transform_weights = get_initial_weights(100)
            stn_net = Dense(6, weights=transform_weights)(stn_net)
            
            stn_output = BilinearInterpolation(sampling_size_from_stn)([input_layer, stn_net])

            model_resnet = ResNet50(weights='imagenet', include_top=False)
            model = model_resnet(stn_output)
            model = Flatten()(model)
            
            # status classify
            model_status = Dense(self._num_class)(model)
            model_status = Activation('softmax', name='soft_max')(model_status)
            
            # booth existed classify
            booth_exitsed = Dense(self._num_class2)(model)
            booth_exitsed = Activation('softmax', name='soft_max2')(booth_exitsed)
            
            model = Model(
                inputs=input_layer,
                outputs=[model_status, booth_exitsed],
                name='StoreBoothNet'
            )
        elif model_config["pre_trained_model"] is not None:
            print("---------------- load resnet50-stn network for train --------------")
            # TODO: to adapt for different input layer and output layer.
            model = load_model(model_config["pre_trained_model"],
                               custom_objects={"BilinearInterpolation": BilinearInterpolation},
                               compile=False)

        model.summary()

        losses = {
            "soft_max": self._loss,
            "soft_max2": self._loss
        }
        loss_weights = {
            "soft_max": 1.0,
            "soft_max2": 1.0
        }
        
        model.compile(
            loss=losses,
            loss_weights=loss_weights,
            optimizer=self._optimizer,
            metrics=['accuracy']
        )

        return model
