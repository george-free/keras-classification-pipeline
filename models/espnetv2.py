from keras.layers import Input, \
    Convolution2D, MaxPooling2D, \
    add, AveragePooling2D, Flatten, Activation, Dropout, \
    BatchNormalization, PReLU
from keras.models import Model
from keras.layers.merge import concatenate

from models.base_model import ClassificationModel
from config import model_config


class Espnetv2(ClassificationModel):
    _map_receptive_ksize = {3: 1, 5: 2, 7: 3, 9: 4, 11: 5, 13: 6, 15: 7, 17: 8}
    _config_feature_map_size = [16, 32, 64, 128, 256, 1024]
    _config_rlim_size = [13, 11, 9, 7, 5]  # receptive field at each spatial level
    def __init__(self, train_data_manager, valid_data_manager, test_data_manager, is_training=True):
        super().__init__(
            train_data_manager, valid_data_manager, test_data_manager, is_training=is_training)
        

    def build_network(self):
        input_layer = Input(shape=self._input_shape)
        if self._train_from_scratch is True:
            print("------------------ build multifunction cls network from scratch --------------－")
            
            # level - 1
            out_l1 = self.layer_cbr(input_layer, self._config_feature_map_size[0], kernal_size=3, stride=2)

            # level - 2
            out_l2 = self.layer_downsample(
                out_l1,
                self._config_feature_map_size[1],
                kernal_size=4,
                r_lim=self._config_rlim_size[0],
                reinf=True,
                x2=input_layer
            )

            # level - 3 - 0
            out_l3_0 = self.layer_downsample(
                out_l2,
                self._config_feature_map_size[2],
                kernal_size=4,
                r_lim=self._config_rlim_size[1],
                reinf=True,
                x2=input_layer
            )
            
            # 3 layers of eesp
            # level - 3 - 1 
            out_l3_1 = self.layer_eesp(
                out_l3_0, self._config_feature_map_size[2], stride=1, kernal_size=4, r_lim=self._config_rlim_size[2])
            # level - 3 - 2
            out_l3_2 = self.layer_eesp(
                out_l3_1, self._config_feature_map_size[2], stride=1, kernal_size=4, r_lim=self._config_rlim_size[2])
            # level - 3 - 3
            out_l3_3 = self.layer_eesp(
                out_l3_2, self._config_feature_map_size[2], stride=1, kernal_size=4, r_lim=self._config_rlim_size[2])

            # level 4 - 0
            out_l4_0 = self.layer_downsample(
                out_l3_3,
                self._config_feature_map_size[3],
                kernal_size=4,
                r_lim=self._config_rlim_size[2],
                reinf=True,
                x2=input_layer
            )
            # level 4 - 1 to 4 - 7
            out_l4_x = self.layer_eesp(out_l4_0, self._config_feature_map_size[3], stride=1, kernal_size=4, r_lim=self._config_rlim_size[3])
            for idx in range(1, 7):
                out_l4_x = self.layer_eesp(
                    out_l4_x, self._config_feature_map_size[3], stride=1, kernal_size=4, r_lim=self._config_rlim_size[3])

            # level 5 - 0
            out_l5_0 =  self.layer_downsample(
                out_l4_x,
                self._config_feature_map_size[4],
                kernal_size=4,
                r_lim=self._config_rlim_size[3],
                reinf=True,
                x2=None
            )
            out_l5_1 = self.layer_eesp(
                out_l5_0, self._config_feature_map_size[4], stride=1, kernal_size=4, r_lim=self._config_rlim_size[4])
            out_l5_2 = self.layer_eesp(
                out_l5_1, self._config_feature_map_size[4], stride=1, kernal_size=4, r_lim=self._config_rlim_size[4])
            out_l5_3 = self.layer_eesp(
                out_l5_2, self._config_feature_map_size[4], stride=1, kernal_size=4, r_lim=self._config_rlim_size[4])

            out_l5_4 = self.layer_cbr(
                out_l5_3, self._config_feature_map_size[4], kernal_size=3, stride=1, groups=self._config_feature_map_size[4])
            out_l5_5 = self.layer_cbr(
                out_l5_4, self._config_feature_map_size[5], kernal_size=1, stride=1, groups=4)

            # avg pool
            avg_pool_l6 = AveragePooling2D((7, 7))(out_l5_5)
            y = Flatten()(avg_pool_l6)
            y = Activation('softmax', name='soft_max')(y)
            model = Model(inputs=input_layer, outputs=y, name='mal_func')
            
        else:
            print("---------------- load multifunction cls network for train --------------")
            model.load_weights(model_config["pre_trained_model"], by_name=True)
        
        # model.summary()
        model.compile(
            loss=self._loss,
            optimizer=self._optimizer,
            metrics=['accuracy']
        )

        return model

    def layer_downsample(self, x, nb_filter, kernal_size=4, r_lim=7, reinf=True, x2=None):
        nb_filter_input = x.shape(0)
        avg_out = AveragePooling2D(pool_size=(2, 2), padding='same', kernal_size=3)(x)
        eesp_out = self.layer_eesp(
            x, nb_filter - nb_filter_input, stride=2, kernal_size=kernal_size, r_lim=r_lim, down_method='avg')
        y = concatenate([avg_out, eesp_out], axis=1)

        w1 = y.shape(2)
        flag = True
        if x2 is not None:
            while True:
                x2 = AveragePooling2D(pool_size=(2, 2), padding='same', kernal_size=3)(x2)
                w2 = x2.shape(2)
                if w2 == w1:
                    break
                if w2.shape(2) <= 1:
                    print("error 111111111111111111111")
                    flag = False
                    break
            inp_reif_layer = self.layer_cbr(x2, 3, 3, 3, 1)
            inp_reif_layer = self.layer_cb(inp_reif_layer, nb_filter, 1, 1)

            y = add([y, inp_reif_layer])
        
        y = PReLU(alpha=0.25)(y)

        return y


    def layer_eesp(self, x, nb_filter, stride=1, kernal_size=4, r_lim=7, down_method='esp'):
        k_sizes = []
        for idx in range(0, kernal_size):
            k = int(3 + 2 * idx)
            k = k if k <= r_lim else 3
            k_sizes.append(k)
        k_sizes.sort()

        y = self.layer_cbr(x, int(nb_filter / kernal_size), 1, stride=1, groups=kernal_size)

        feature_layers = []
        y = Convolution2D(nb_filter, kernel_size=3, 
            strides=stride, groups=nb_filter, dilation_rate=self._map_receptive_ksize[k_sizes[0]])(y)
        feature_layers.append(y)
        for idx in range(1, kernal_size):
            d_rate = self._map_receptive_ksize[k_sizes[idx]]
            y = Convolution2D(nb_filter, kernal_size=3, strides=stride, groups=nb_filter,
                dilation_rate=d_rate)(y) + feature_layers[idx - 1]
            feature_layers.append(y)

        y = concatenate(feature_layers, axis=1)
        y = self.layer_br(y)
        y = self.layer_cb(y, nb_filter, 1, 1, groups=kernal_size)

        if stride == 2 and down_method == "avg":
            return y
        elif y.shape == x.shape:
            y = add([y, input])
        
        y = PReLU(alpha=0.25)(y)

        return y

    def layer_cbr(self, x, nb_filter, kernal_size, stride=1, groups=1):
        y = Convolution2D(
            nb_filter, kernal_size, strides=stride, groups=groups,
            padding="same", kernel_initializer='he_normal')(x)
        y = BatchNormalization(axis=3)(y)
        y = PReLU(alpha=0.25)(y)

        return y

    def layer_br(self, x):
        y = BatchNormalization(axis=3)(x)
        y = PReLU(alpha=0.25)(y)
        
        return y

    def layer_cb(self, x, nb_filter, kernal_size, stride=1, groups=1):
        y = Convolution2D(
            nb_filter, kernal_size, strides=stride, groups=groups,
            padding="same", kernel_initializer='he_normal')(x)
        y = BatchNormalization(axis=3)(y)

        return y

    def layer_c_dilatedB(self, x, nb_filter, kernal_size, stride=1, d=1, groups=1):
        y = Convolution2D(
            nb_filter, kernal_size, strides=stride, groups=groups, dilation_rate=d,
            padding="same", kernel_initializer='he_normal')(x)
        
        y = BatchNormalization(axis=3)(y)
        return y

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
