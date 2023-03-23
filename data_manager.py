from keras.preprocessing.image import ImageDataGenerator

from config import model_config
from tools.dataset import *


class MultiOutputDataGenerator(ImageDataGenerator):
    def __init__(self, 
                 featurewise_center=False, 
                 samplewise_center=False, 
                 featurewise_std_normalization=False, 
                 samplewise_std_normalization=False, 
                 zca_whitening=False, 
                 zca_epsilon=1e-06, 
                 rotation_range=0, 
                 width_shift_range=0.0, 
                 height_shift_range=0.0, 
                 brightness_range=None, 
                 shear_range=0.0, 
                 zoom_range=0.0, 
                 channel_shift_range=0.0, 
                 fill_mode='nearest', cval=0.0, 
                 horizontal_flip=False, 
                 vertical_flip=False, 
                 rescale=None, 
                 preprocessing_function=None, 
                 data_format=None, 
                 validation_split=0.0, 
                 dtype=None):
        super().__init__(featurewise_center=featurewise_center, 
                         samplewise_center=samplewise_center, 
                         featurewise_std_normalization=featurewise_std_normalization, 
                         samplewise_std_normalization=samplewise_std_normalization, 
                         zca_whitening=zca_whitening, 
                         zca_epsilon=zca_epsilon, 
                         rotation_range=rotation_range, 
                         width_shift_range=width_shift_range, 
                         height_shift_range=height_shift_range, 
                         brightness_range=brightness_range, 
                         shear_range=shear_range, 
                         zoom_range=zoom_range, 
                         channel_shift_range=channel_shift_range, 
                         fill_mode=fill_mode, 
                         cval=cval, 
                         horizontal_flip=horizontal_flip, 
                         vertical_flip=vertical_flip, 
                         rescale=rescale, 
                         preprocessing_function=preprocessing_function, 
                         data_format=data_format, 
                         validation_split=validation_split, 
                         dtype=dtype)
    
    def flow(self,
             x,
             y=None,
             batch_size=32,
             shuffle=True,
             sample_weight=None,
             seed=None,
             save_to_dir=None,
             save_prefix='',
             save_format='png',
             subset=None):

        targets = None
        target_lengths = {}
        ordered_outputs = []
        for output, target in y.items():
            if targets is None:
                targets = target
            else:
                targets = np.concatenate((targets, target), axis=1)
            target_lengths[output] = target.shape[1]
            ordered_outputs.append(output)


        for flowx, flowy in super().flow(x, targets, batch_size=batch_size,
                                         shuffle=shuffle):
            target_dict = {}
            i = 0
            for output in ordered_outputs:
                target_length = target_lengths[output]
                target_dict[output] = flowy[:, i: i + target_length]
                i += target_length

            yield flowx, target_dict


class DataManager:
    def __init__(self):
        self._input_shape = model_config.get("input_shape", None)
        self._batch_size = model_config.get("batch_size", None)
        self._num_class = model_config.get("num_class", None)

    def build_data_generator_for_cifar10_dataset(self, step='train'):

        aug_config = model_config['data_augmentation_config']
        assert(self._num_class == 10)
        assert(self._input_shape[0] == 32 and self._input_shape[1] == 32)

        data_dir = None
        shuffle = True

        if step == 'train':
            data_dir = model_config['train_data_dir']
            images, labels = load_train_data(data_dir , num_class=self._num_class)
        elif step == 'valid':
            data_dir = model_config['valid_data_dir']
            images, labels = load_test_data(data_dir, num_class=self._num_class)
            shuffle = False
        elif step == 'test':
            data_dir = model_config['test_data_dir']
            images, labels = load_test_data(data_dir, num_class=self._num_class)
            shuffle = False
        else:
            raise ValueError("{} step value - {} error not in [train | valid | test]".format(
                model_config["model_name"],
                step
            ))

        if data_dir is None:
            raise ValueError("{} step value - {} dataset is None".format(
                model_config["model_name"],
                step
            ))

        # step 1. generate image generator
        if step == 'train':
            generator = ImageDataGenerator(
                rescale=aug_config['rescale'],
                rotation_range=aug_config['rotation_range'],
                width_shift_range=aug_config['w_shift'],
                height_shift_range=aug_config['h_shift'],
                horizontal_flip=aug_config['horizontal_flip'],
                vertical_flip=aug_config['vertical_flip'],
                shear_range=aug_config['shear_range'],
                zoom_range=aug_config['zoom_range'],
                fill_mode='nearest'
            )
        elif step == 'valid' or step == 'test':
            generator = ImageDataGenerator(
                rescale=aug_config['rescale']
            )
        
        generator.fit(images)
        data_manager = generator.flow(
            images,
            labels,
            batch_size=self._batch_size,
            shuffle=shuffle
        )

        return data_manager
    
    def build_data_generator_from_memory(self, step='train'):
        aug_config = model_config['data_augmentation_config']

        data_dir = None
        shuffle = True

        if step == 'train':
            data_dir = model_config['train_data_dir']
            images, labels = load_dir_datas(data_dir , num_class=self._num_class)
        elif step == 'valid':
            data_dir = model_config['valid_data_dir']
            images, labels = load_dir_datas(data_dir, num_class=self._num_class)
            shuffle = False
        elif step == 'test':
            data_dir = model_config['test_data_dir']
            images, labels = load_dir_datas(data_dir, num_class=self._num_class)
            shuffle = False
        else:
            raise ValueError("{} step value - {} error not in [train | valid | test]".format(
                model_config["model_name"],
                step
            ))

        if data_dir is None:
            raise ValueError("{} step value - {} dataset is None".format(
                model_config["model_name"],
                step
            ))

        featurewise_center = False
        featurewise_std_normalization = False
        if aug_config['normalize']:
            featurewise_center = True
            featurewise_std_normalization = True

        # step 1. generate image generator
        if step == 'train':
            generator = ImageDataGenerator(
                rescale=aug_config['rescale'],
                rotation_range=aug_config['rotation_range'],
                width_shift_range=aug_config['w_shift'],
                height_shift_range=aug_config['h_shift'],
                horizontal_flip=aug_config['horizontal_flip'],
                vertical_flip=aug_config['vertical_flip'],
                shear_range=aug_config['shear_range'],
                zoom_range=aug_config['zoom_range'],
                fill_mode='nearest'
            )
        elif step == 'valid' or step == 'test':
            generator = ImageDataGenerator(
                rescale=aug_config['rescale'],
                rotation_range=aug_config['rotation_range'],
                width_shift_range=aug_config['w_shift'],
                height_shift_range=aug_config['h_shift'],
                horizontal_flip=aug_config['horizontal_flip'],
                vertical_flip=aug_config['vertical_flip'],
                featurewise_center=featurewise_center,
                featurewise_std_normalization=featurewise_std_normalization,
                shear_range=aug_config['shear_range'],
                zoom_range=aug_config['zoom_range'],
                fill_mode='nearest'
            )

        generator.fit(images)
        data_manager = generator.flow(
            images,
            labels,
            batch_size=self._batch_size,
            shuffle=shuffle
        )

        return data_manager
        

    def build_data_generator_from_dirs(self, step='train'):
        ''' acquire train/valid/test image datas from directory,
            step can be 'train' | 'valid' | 'test', because train data generator shall be 
            applied with data augmentation while 'valid' and 'test' not.
        '''

        aug_config = model_config['data_augmentation_config']

        featurewise_center = False
        featurewise_std_normalization = False
        if aug_config['normalize']:
            featurewise_center = True
            featurewise_std_normalization = True

        data_dir = None
        shuffle = True
        if step == 'train':
            data_dir = model_config['train_data_dir']
        elif step == 'valid':
            data_dir = model_config['valid_data_dir']
        elif step == 'test':
            data_dir = model_config['test_data_dir']
            shuffle = False
        else:
            raise ValueError("{} step value - {} error not in [train | valid | test]".format(
                model_config["model_name"],
                step
            ))

        if data_dir is None:
            raise ValueError("{} step value - {} dataset is None".format(
                model_config["model_name"],
                step
            ))

        if step == 'train':
            generator = ImageDataGenerator(
                rescale=aug_config['rescale'],
                rotation_range=aug_config['rotation_range'],
                width_shift_range=aug_config['w_shift'],
                height_shift_range=aug_config['h_shift'],
                horizontal_flip=aug_config['horizontal_flip'],
                vertical_flip=aug_config['vertical_flip'],
                featurewise_center=featurewise_center,
                featurewise_std_normalization=featurewise_std_normalization,
                shear_range=aug_config['shear_range'],
                zoom_range=aug_config['zoom_range'],
                fill_mode='nearest'
            )
        elif step == 'valid' or step == 'test':
            generator = ImageDataGenerator(
                rescale=aug_config['rescale'],
                featurewise_center=featurewise_center,
                featurewise_std_normalization=featurewise_std_normalization,
            )

        data＿manager = generator.flow_from_directory(
                data_dir,
                target_size=(self._input_shape[0], self._input_shape[1]),
                batch_size=self._batch_size,
                shuffle=shuffle,
            )
        
        return data＿manager


class MultiLabelDataManager(DataManager):
    def __init__(self):
        super().__init__()
        self._num_class2 = model_config.get("num_class2", None)
        
    def build_data_generator_from_memory(self, step='train'):
        aug_config = model_config['data_augmentation_config']

        data_dir = None
        shuffle = True

        if step == 'train':
            data_dir = model_config['train_data_dir']
            images, labels1, labels2 = load_dir_datas_with_two_labels(
                data_dir, 
                num_class1=self._num_class,
                num_class2=self._num_class2
            )
        elif step == 'valid':
            data_dir = model_config['valid_data_dir']
            images, labels1, labels2 = load_dir_datas_with_two_labels(
                data_dir, 
                num_class1=self._num_class,
                num_class2=self._num_class2
            )
            shuffle = False
        elif step == 'test':
            data_dir = model_config['test_data_dir']
            images, labels1, labels2 = load_dir_datas_with_two_labels(
                data_dir, 
                num_class1=self._num_class,
                num_class2=self._num_class2
            )
            shuffle = False
        else:
            raise ValueError("{} step value - {} error not in [train | valid | test]".format(
                model_config["model_name"],
                step
            ))

        if data_dir is None:
            raise ValueError("{} step value - {} dataset is None".format(
                model_config["model_name"],
                step
            ))

        # step 1. generate image generator
        if step == 'train':
            generator = MultiOutputDataGenerator(
                rescale=aug_config['rescale'],
                rotation_range=aug_config['rotation_range'],
                width_shift_range=aug_config['w_shift'],
                height_shift_range=aug_config['h_shift'],
                horizontal_flip=aug_config['horizontal_flip'],
                vertical_flip=aug_config['vertical_flip'],
                shear_range=aug_config['shear_range'],
                zoom_range=aug_config['zoom_range'],
                fill_mode='nearest'
            )
        elif step == 'valid' or step == 'test':
            generator = MultiOutputDataGenerator(
                rescale=aug_config['rescale']
            )
        
        generator.fit(images)
        data_manager = generator.flow(
            x=images,
            y={"soft_max": labels1, "soft_max2": labels2},
            batch_size=self._batch_size,
            shuffle=shuffle
        )

        return data_manager
