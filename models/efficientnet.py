from keras.layers import Input, merge, Dropout, Dense, Lambda, Flatten, Activation
import efficientnet.keras as efn
from keras.models import Model

from models.base_model import ClassificationModel
from config import model_config


class Efficientnet(ClassificationModel):
    def build_network(self):
        input_layer = Input(self._input_shape)
        
        if self._train_from_scratch:
            print("---------------- build inceptionresnet v2 with stn network from scratch --------------")
            # model = self.create_inception_resnet_v2(scale=False)
            # model_inception_resnetv2 = InceptionResNetV2(
            #     weights = 'imagenet', 
            #     include_top = False
            # )
            model_efficient_base = efn.EfficientNetB0(input_shape = self._input_shape, include_top = False, weights = 'imagenet')
            # for layer in model_efficient_base.layers:
            #     layer.trainable = False
            model = model_efficient_base(input_layer)
            model = Flatten()(model)
            model = Dense(self._num_class)(model)
            model = Activation('softmax', name='soft_max')(model)
            model = Model(input_layer, model)
        else:
            raise NotImplementedError("resume inceptionresnet v2 not supported yet")

        # elif model_config["pre_trained_model"] is not None:
        #     print("---------------- load inceptionresnet v2 network for train --------------")
        #     # TODO: to adapt for different input layer and output layer.
        #     model = load_model(model_config["pre_trained_model"],
        #                        custom_objects={"BilinearInterpolation": BilinearInterpolation},
        #                        compile=False)

        model.summary()

        model.compile(
            loss=self._loss,
            optimizer=self._optimizer,
            metrics=['accuracy']
        )

        return model