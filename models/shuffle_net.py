import os
import keras
from keras import backend as K
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.models import Model
from keras.engine.topology import get_source_inputs
from keras.layers import Activation, Add, Concatenate, Conv2D, GlobalMaxPooling2D
from keras.layers import GlobalAveragePooling2D, Input, Dense, InputLayer
from keras.layers import MaxPool2D,AveragePooling2D, BatchNormalization, Lambda, DepthwiseConv2D
from keras.models import Sequential
import numpy as np
from keras.models import load_model

from models.base_model import ClassificationModel
from config import model_config

def channel_split(x, name=''):
    # equipartition
    in_channles = x.shape.as_list()[-1]
    ip = in_channles // 2
    c_hat = Lambda(lambda z: z[:, :, :, 0:ip], name='%s/sp%d_slice' % (name, 0))(x)
    c = Lambda(lambda z: z[:, :, :, ip:], name='%s/sp%d_slice' % (name, 1))(x)
    return c_hat, c

def channel_shuffle(x):
    height, width, channels = x.shape.as_list()[1:]
    channels_per_split = channels // 2
    x = K.reshape(x, [-1, height, width, 2, channels_per_split])
    x = K.permute_dimensions(x, (0,1,2,4,3))
    x = K.reshape(x, [-1, height, width, channels])
    return x

# build unit
def shuffle_unit(inputs, out_channels, bottleneck_ratio,strides=2,stage=1,block=1):
    if K.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        raise ValueError('Only channels last supported')

    prefix = 'stage{}/block{}'.format(stage, block)
    bottleneck_channels = int(out_channels * bottleneck_ratio)
    if strides < 2:
        c_hat, c = channel_split(inputs, '{}/spl'.format(prefix))
        inputs = c

    x = Conv2D(bottleneck_channels, kernel_size=(1,1), strides=1, padding='same', name='{}/1x1conv_1'.format(prefix))(inputs)
    x = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_1'.format(prefix))(x)
    x = Activation('relu', name='{}/relu_1x1conv_1'.format(prefix))(x)
    x = DepthwiseConv2D(kernel_size=3, strides=strides, padding='same', name='{}/3x3dwconv'.format(prefix))(x)
    x = BatchNormalization(axis=bn_axis, name='{}/bn_3x3dwconv'.format(prefix))(x)
    x = Conv2D(bottleneck_channels, kernel_size=1,strides=1,padding='same', name='{}/1x1conv_2'.format(prefix))(x)
    x = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_2'.format(prefix))(x)
    x = Activation('relu', name='{}/relu_1x1conv_2'.format(prefix))(x)

    if strides < 2:
        ret = Concatenate(axis=bn_axis, name='{}/concat_1'.format(prefix))([x, c_hat])
    else:
        s2 = DepthwiseConv2D(kernel_size=3, strides=2, padding='same', name='{}/3x3dwconv_2'.format(prefix))(inputs)
        s2 = BatchNormalization(axis=bn_axis, name='{}/bn_3x3dwconv_2'.format(prefix))(s2)
        s2 = Conv2D(bottleneck_channels, kernel_size=1,strides=1,padding='same', name='{}/1x1_conv_3'.format(prefix))(s2)
        s2 = BatchNormalization(axis=bn_axis, name='{}/bn_1x1conv_3'.format(prefix))(s2)
        s2 = Activation('relu', name='{}/relu_1x1conv_3'.format(prefix))(s2)
        ret = Concatenate(axis=bn_axis, name='{}/concat_2'.format(prefix))([x, s2])

    ret = Lambda(channel_shuffle, name='{}/channel_shuffle'.format(prefix))(ret)

    return ret

def block(x, channel_map, bottleneck_ratio, repeat=1, stage=1):
    x = shuffle_unit(x, out_channels=channel_map[stage-1],
                    strides=2,bottleneck_ratio=bottleneck_ratio,stage=stage,block=1)

    for i in range(1, repeat+1):
        x = shuffle_unit(x, out_channels=channel_map[stage-1],strides=1,
                        bottleneck_ratio=bottleneck_ratio,stage=stage, block=(1+i))

    return x

class ShufflenetModel(ClassificationModel):

    def build_network(self):

        scale_factor = 1.0
        bottleneck_ratio = 1
        pooling = 'max' # 'max' | 'avg' max_pooling or avg_pooling at the end of network
        num_shuffle_units = [3, 7, 3]

        # set input layer
        input_shape = _obtain_input_shape(self._input_shape, default_size=224, min_size=28, require_flatten=True,
                                            data_format=K.image_data_format())
        img_input = Input(shape=input_shape)
        
        if self._train_from_scratch:
            print("---------------- build shuffle network from scratch --------------")

            # check network parameters
            if K.backend() != 'tensorflow':
                raise RuntimeError('Only tensorflow supported for now')

            if pooling not in ['max', 'avg']:
                raise ValueError('Invalid value for pooling')
            if not (float(scale_factor)*4).is_integer():
                raise ValueError('Invalid value for scale_factor, should be x over 4')

            name = 'ShuffleNetV2_{}_{}_{}'.format(scale_factor, bottleneck_ratio, "".join([str(x) for x in num_shuffle_units]))
            
            out_dim_stage_two = {0.5:48, 1:116, 1.5:176, 2:244}
            
            exp = np.insert(np.arange(len(num_shuffle_units), dtype=np.float32), 0, 0)  # [0., 0., 1., 2.]
            out_channels_in_stage = 2**exp     #  [0.,0.,2.,4.]
            out_channels_in_stage *= out_dim_stage_two[bottleneck_ratio]  # calculate output channels for each stage [0,0,232,464]
            out_channels_in_stage[0] = 24  # first stage has always 24 output channels  [24, 0, 232, 464]
            out_channels_in_stage *= scale_factor
            out_channels_in_stage = out_channels_in_stage.astype(int)

            # create shufflenet architecture
            x = Conv2D(filters=out_channels_in_stage[0], kernel_size=(3, 3), padding='same', use_bias=False, strides=(2, 2),
                    activation='relu', name='conv1')(img_input)
            x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='maxpool1')(x)

            # create stages containing shufflenet units beginning at stage 2
            for stage in range(len(num_shuffle_units)):
                repeat = num_shuffle_units[stage]
                x = block(x, out_channels_in_stage,          # [24,0,232,464]
                        repeat=repeat,
                        bottleneck_ratio=bottleneck_ratio,
                        stage=stage + 2)

            if bottleneck_ratio < 2:
                k = 1024
            else:
                k = 2048
            x = Conv2D(k, kernel_size=1, padding='same', strides=1, name='1x1conv5_out', activation='relu')(x)

            if pooling == 'avg':
                x = GlobalAveragePooling2D(name='global_avg_pool')(x)
            elif pooling == 'max':
                x = GlobalMaxPooling2D(name='global_max_pool')(x)

            x = Dense(self._num_class, name='fc')(x)
            x = Activation('softmax', name='softmax')(x)

            model = Model(img_input, x, name=name)

        elif model_config["pre_trained_model"] is not None:
            print("---------------- load shuffle network for train --------------")

            model = load_model(model_config["pre_trained_model"], compile=False)
            num_layers = len(model.layers)
            dense_layer_loc = num_layers - 2
            dense_layer = model.layers[dense_layer_loc]
            input_tensor_shape = model.input_shape[1:]
            if self._input_shape != input_tensor_shape:
                raise ValueError("{} model must have the same input tensor shape for fine-tuning, your input shape : {}, while pretrained input shape : {}".format(
                    model_config['model_name'],
                    self._input_shape,
                    input_tensor_shape          
                ))

            if dense_layer.get_config()['units'] != self._num_class:
                x = model.layers[-3].output
                x = Dense(self._num_class, name='fc', trainable=True)(x)
                x = Activation('softmax', name='softmax')(x)
                
                model = Model(model.inputs, x)
                model.summary()
        else:
            raise Exception("{} trained model - {} load error".format(
                model_config["mobile_name"],
                model_config["pre_trained_model"]
            ))
        
        model.compile(
            loss=self._loss,
            optimizer=self._optimizer,
            metrics=['accuracy']
        )

        return model