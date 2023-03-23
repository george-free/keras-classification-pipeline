from keras.layers import Input, \
    Convolution2D, MaxPooling2D, \
    add, AveragePooling2D, Flatten, Activation, Dropout, \
    BatchNormalization
from keras.models import Model

from models.base_model import ClassificationModel
from config import model_config


class MalFunctionModel(ClassificationModel):
    def build_network(self):
        input_layer = Input(shape=self._input_shape)
        if self._train_from_scratch is True:
            print("------------------ build multifunction cls network from scratch ----------------")
            
            # # 128 x 128 x 3
            # y = input_layer
            
            # # conv  64 x 64 x 64
            # # y = self.identity_block(y, nb_filter=64, kernel_size=5, strides=1, pooling=False)
            # y = self.identity_block(y, nb_filter=32, kernel_size=5, strides=1, pooling=True)

            # # conv 32 x 32 x 64
            # # y = self.identity_block(y, nb_filter=64, kernel_size=5, strides=1, pooling=False)
            # y = self.identity_block(y, nb_filter=64, kernel_size=5, strides=1, pooling=True)

            # # conv 16 x 16 x 16
            # y = self.identity_block(y, nb_filter=128, kernel_size=3, strides=1, pooling=False)
            # y = self.identity_block(y, nb_filter=128, kernel_size=3, strides=1, pooling=True)

            # # conv 8 x 8 x 2
            # y = self.identity_block(y, nb_filter=2, kernel_size=3, strides=1, pooling=False)
            # y = self.identity_block(y, nb_filter=2, kernel_size=3, strides=1, pooling=True)

            # # avg pooling
            # y = AveragePooling2D((8, 8), name="avg_pooling")(y)

            # y = Flatten()(y)
            # y = Activation("softmax", name="softmax")(y)

            # model = Model(input_layer, y, name="mal_function")

            y = input_layer
            y = Convolution2D(16, kernel_size=5, strides=1, activation="relu", kernel_initializer='he_normal', padding="same")(y)
            y = Convolution2D(16, kernel_size=5, strides=1, activation="relu", kernel_initializer='he_normal', padding="same")(y)
            y = BatchNormalization(axis=3)(y)
            y = MaxPooling2D(2, 2, padding="same")(y)
            y = Dropout(0.4)(y)

            y = Convolution2D(32, kernel_size=3, strides=1, activation="relu", kernel_initializer='he_normal', padding="same")(y)
            y = BatchNormalization(axis=3)(y)
            y = MaxPooling2D(2, 2, padding="same")(y)
            y = Dropout(0.5)(y)

            y = Convolution2D(16, kernel_size=1, strides=1, activation="relu", kernel_initializer='he_normal', padding="same")(y)
            y = BatchNormalization(axis=3)(y)
            # y = MaxPooling2D(2, 2, padding="same")(y)
            y = Dropout(0.4)(y)

            y = Convolution2D(2, kernel_size=1, activation="relu", kernel_initializer='he_normal', padding="same")(y)
            y = BatchNormalization(axis=3)(y)
            y = AveragePooling2D((16, 16), name="avg_pooling")(y)

            y = Flatten()(y)
            y = Activation("softmax", name="softmax")(y)

            model = Model(input_layer, y, name="mal_function")

        else:
            print("---------------- load multifunction cls network for train --------------")
            model.load_weights(model_config["pre_trained_model"], by_name=True)
        
        model.summary()
        model.compile(
            loss=self._loss,
            optimizer=self._optimizer,
            metrics=['accuracy']
        )

        return model

    def identity_block(self,
                       input,
                       nb_filter,
                       kernel_size,
                       strides=2,
                       padding="same",
                       pooling:bool=True,
                       name=None):
        # nf1, nf2, nf3 = nb_filter
        x = Convolution2D(
            nb_filter, 1, strides=1, padding=padding, activation="relu", kernel_initializer='he_normal')(input)
        x = BatchNormalization(axis=3)(x)
        x = Convolution2D(
            nb_filter, kernel_size, strides=strides, padding=padding, activation="relu", kernel_initializer='he_normal')(x)
        x = BatchNormalization(axis=3)(x)
        x = Convolution2D(
            nb_filter, 1, strides=1, padding=padding, activation="relu", kernel_initializer='he_normal')(x)
        x = BatchNormalization(axis=3)(x)
        
        short_cut_layer = Convolution2D(
            nb_filter, 1, strides=strides, padding=padding, activation="relu", kernel_initializer='he_normal')(input)
        short_cut_layer = BatchNormalization(axis=3)(short_cut_layer)

        x = add([x, short_cut_layer])
        x = Dropout(0.2)(x)

        if pooling is True:
            x = MaxPooling2D(pool_size=2, strides=2, padding=padding)(x)

        return x
