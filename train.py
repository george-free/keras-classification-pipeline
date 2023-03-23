from model_factory import ModelFactory
from config import model_config
from data_manager import DataManager, MultiLabelDataManager
import keras
import tensorflow as tf
import os


if __name__ == "__main__":

    model_name = model_config['model_name']
    if_multi_label = model_config['multi_label']
    num_gpus = model_config.get("gpus", 1)

    # os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    # num_gpus = len(gpus.split(','))
    # print("use --- {} gpus".format(num_gpus))
    backend_config = tf.ConfigProto(
        device_count={"GPU": num_gpus},
    )
    backend_config.gpu_options.allow_growth = True
    sess = tf.Session(config=backend_config)
    keras.backend.set_session(sess)

    if if_multi_label:
        data_generator = MultiLabelDataManager()
    else:
        data_generator = DataManager()

    # read data from image files
    if model_config['dataset_type'] == 'images':
        train_data_manager = data_generator.build_data_generator_from_dirs(step='train')
        valid_data_manager = data_generator.build_data_generator_from_dirs(step='valid')
        test_data_manager = data_generator.build_data_generator_from_dirs(step='test')
    # read cifar10 dara from binary file
    elif model_config['dataset_type'] == 'cifar10':
        train_data_manager = data_generator.build_data_generator_for_cifar10_dataset(step='train')
        valid_data_manager = data_generator.build_data_generator_for_cifar10_dataset(step='valid')
        test_data_manager = None
    elif model_config['dataset_type'] == 'memory':
        train_data_manager = data_generator.build_data_generator_from_memory(step='train')
        valid_data_manager = data_generator.build_data_generator_from_memory(step='valid')
        test_data_manager = data_generator.build_data_generator_from_memory(step='test')
    else:
        raise ValueError("Only support image file data and cifar image data.")

    # create model
    trainer = ModelFactory.create_model(model_name,
        train_data_manager,
        valid_data_manager,
        test_data_manager
    )

    # train, valid and test
    trainer.train()
