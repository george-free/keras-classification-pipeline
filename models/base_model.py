from abc import abstractmethod
import keras
from keras.utils import multi_gpu_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import matplotlib.pyplot as plt
import numpy as np

from config import model_config

# remove warining logs
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class ClassificationModel:
    def __init__(self,
                 train_data_manager,
                 valid_data_manager,
                 test_data_manager,
                 is_training=True):
        
        self._is_traing = is_training

        self._input_shape = model_config.get("input_shape", None)
        self._num_class = model_config.get("num_class", None)
        self._num_class2 = model_config.get("num_class2", None)
        self._batch_size = model_config.get("batch_size", None)
        self._num_epoches = model_config.get("epoches", None)
        self._class_weights = model_config.get("class_weights", None)

        if self._input_shape is None:
            raise Exception("{} input shape build error".format(model_config["model_name"]))
        if self._num_class is None:
            raise Exception("{} num class build error".format(model_config["model_name"]))
        if self._batch_size is None:
            raise Exception("{} batch size build error".format(model_config["model_name"]))
        if self._num_epoches is None:
            raise Exception("{} num epoches build error".format(model_config["model_name"]))

        self._train_from_scratch = model_config["train_from_scratch"]
        

        self._train_data_manager = train_data_manager
        self._valid_data_manager = valid_data_manager
        self._test_data_manager = test_data_manager


        self._optimizer = self.build_optimizer(None)

        self._loss = self.build_loss()

        self._model_saved_path = os.path.join(model_config['save_model_dir'], model_config["model_name"])
        if self._model_saved_path is None:
            raise ValueError(
                "{} saved model path is set error".format(
                    model_config["model_name"]
                )
            )
        if not os.path.exists(self._model_saved_path):
            os.makedirs(self._model_saved_path)
        
        self._model_saved_path = self._model_saved_path + "/{}_data{}_op{}_lr{}_decay{}_bsize{}_numclass{}_epoch{}_scratch{}_input{}.h5".format(
            model_config['model_name'],
            model_config['dataset_type'],
            model_config['optimizer']['name'],
            model_config['optimizer']['base_lr'],
            model_config['optimizer']['lr_decay'],
            model_config['batch_size'],
            model_config['num_class'],
            model_config['epoches'],
            model_config['train_from_scratch'],
            "{}x{}".format(model_config['input_shape'][0], model_config['input_shape'][1])
        )

        self._log_dir = model_config['log_dir']
        if self._log_dir is None or \
           not os.path.exists(self._log_dir):
            raise ValueError(
                "{} log dir path - {} error".format(
                    model_config["model_name"],
                    self._log_dir
                )
            )

        self._model = self.build_network()
        self._model.summary()
        
        gpus = model_config.get("gpus", 0)
        # print("use --- {} gpus".format(gpus))
        num_gpus = gpus
        if num_gpus > 1:
           self._model = multi_gpu_model(self._model, gpus=num_gpus)

        self._model.compile(
            loss=self._loss,
            optimizer=self._optimizer,
            metrics=['acc']
        )

    def build_loss(self):
        if model_config['loss'] == 'cross_entropy':
            return keras.losses.categorical_crossentropy

        raise NotImplementedError(
            "{} other loss function is not implemented".format(model_config["model_name"])
        )

    def show_training_result(self, h):
        acc, val_acc = h.history['acc'], h.history['val_acc']
        m_acc, m_val_acc = np.argmax(acc), np.argmax(val_acc)
        print("@ Best Training Accuracy: %.2f %% achieved at EP #%d." % (acc[m_acc] * 100, m_acc + 1))
        print("@ Best Validing Accuracy: %.2f %% achieved at EP #%d." % (val_acc[m_val_acc] * 100, m_val_acc + 1))
        
        plt.plot(h.history['acc'])
        plt.plot(h.history['val_acc'])
        plt.title('model acc')
        plt.ylabel('acc')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('{}/{}_train_accuracy.png'.format(self._log_dir, model_config["model_name"]))
    
    def build_optimizer(self, self_optimizer=None):
        if self_optimizer is not None:
            return self_optimizer

        optimizer_config = model_config['optimizer']
        optimizer = None

        lr_rate = optimizer_config['base_lr']
        lr_decay = optimizer_config['lr_decay']
        momentum = optimizer_config['momentum']
        rho = optimizer_config['rho']

        if optimizer_config['name'] == 'sgd':
            optimizer = keras.optimizers.SGD(lr=lr_rate, decay=lr_decay, momentum=momentum, nesterov=True)
        elif optimizer_config['name'] == 'rms':
            optimizer = keras.optimizers.RMSprop(lr=lr_rate, rho=rho, decay=lr_decay)
        elif optimizer_config['name'] == 'adam':
            optimizer = keras.optimizers.Adadelta(lr=lr_rate, rho=rho, decay=lr_decay)
            # optimizer = 'adadelta'
        else:
            raise ValueError('{} optimizer - {} is not support'.format(
                model_config["model_name"],
                optimizer_config['name']
            ))

        return optimizer

    @abstractmethod
    def build_network(self):
        raise NotImplementedError
    
    
    def train(self):
        if self._model is None:
            raise ValueError("{} model is not built before training.".format(model_config["mobile_name"]))
        if self._optimizer is None:
            raise ValueError("{} optimizer is not built before training.".format(model_config["mobile_name"]))

        best_model = ModelCheckpoint(self._model_saved_path,
            verbose=1,
            monitor='val_acc',
            save_best_only=True
        )

        if model_config["multi_label"]:
            history = self._model.fit_generator(
                self._train_data_manager,
                validation_data=self._valid_data_manager,
                steps_per_epoch=100,
                validation_steps=20,
                epochs=self._num_epoches,
                callbacks=[best_model]
            )          
        else:
            history = self._model.fit_generator(
                self._train_data_manager,
                validation_data=self._valid_data_manager,
                steps_per_epoch=self._train_data_manager.n/self._batch_size,
                validation_steps=self._valid_data_manager.n/self._batch_size,
                epochs=self._num_epoches,
                callbacks=[best_model],
                class_weight=self._class_weights            
            )

        self.show_training_result(history)

        if self._test_data_manager is not None:
            test_loss, test_acc = self._model.evaluate_generator(
                self._test_data_manager,
                steps=self._test_data_manager.n/self._batch_size
            )
            print('Test result - loss: {}, acc: {}'.format(
                test_loss,
                test_acc
            ))
