import keras
from keras.models import load_model
from keras.layers import Dense, Flatten, Activation, Input
from keras.applications.resnet import ResNet101
from keras.models import Model

from models.base_model import ClassificationModel
from config import model_config

class Resnet101Model(ClassificationModel):

    def build_network(self):
        input_layer = Input(self._input_shape)
        
        if self._train_from_scratch:
            print("---------------- build resnet101 network from scratch --------------")

            model_resnet = ResNet101(weights='imagenet', include_top=False)
            model = model_resnet(input_layer)
            model = Flatten()(model)
            model = Dense(self._num_class)(model)
            model = Activation('softmax', name='soft_max')(model)
            model = Model(input_layer, model)
        elif model_config["pre_trained_model"] is not None:
            print("---------------- load resnet101 network for train --------------")

            model = load_model(model_config["pre_trained_model"], compile=False)
            model.summary()

            num_layers = len(model.layers)
            dense_layer_loc = num_layers - 2
            dense_layer = model.layers[dense_layer_loc]

            if dense_layer.get_config()['units'] != self._num_class:
                x = input_layer
                for layer in model.layers[1:2]:
                    x = layer(x)            
                x = Flatten()(x)
                x = Dense(self._num_class)(x)
                x = Activation('softmax', name='soft_max')(x)
                model = Model(input_layer, x, name='fine tuning resnet50net_v2')
        else:
            raise Exception("{} trained model - {} load error".format(
                model_config["mobile_name"],
                model_config["pre_trained_model"]
            ))

        model.summary()

        model.compile(
            loss=self._loss,
            optimizer=self._optimizer,
            metrics=['accuracy']
        )

        return model