from models.mobile_net import MobilenetModel
from models.alex_net import AlexnetModel
from models.shuffle_net import ShufflenetModel
from models.light_weight_net import LightWeightModel
from models.resnet50 import Resnet50Model
from models.resnet101 import Resnet101Model
from models.light_weight_net_with_stn import LightWeightStnModel
from models.resnet50_with_stn import Resnet50StnModel
from models.mobilenet_stn import MobilenetStnModel
from models.resnet50_with_stn_two_labels import Resnet50StnModel2Labels
from models.deeplabv3 import DeeplabV3
from models.inception_resnetv2 import InceptionResnetv2
from models.efficientnet import Efficientnet
from models.mal_function_cls import MalFunctionModel


class ModelFactory:

    @classmethod
    def create_model(cls, model_name,
                     train_data_manager,
                     valid_data_manager,
                     test_data_manager
                     ):
        if model_name == 'mobile_net':
            return MobilenetModel(
                train_data_manager,
                valid_data_manager,
                test_data_manager,
                is_training=True
            )
        elif model_name == 'alex_net':
            return AlexnetModel(
                train_data_manager,
                valid_data_manager,
                test_data_manager,
                is_training=True
            )
        elif model_name == 'shuffle_net':
            return ShufflenetModel(
                train_data_manager,
                valid_data_manager,
                test_data_manager,
                is_training=True
            )
        elif model_name == 'light_weight_net':
            return LightWeightModel(
                train_data_manager,
                valid_data_manager,
                test_data_manager,
                is_training=True
            )
        elif model_name == 'light_weight_stn_net':
            return LightWeightStnModel(
                train_data_manager,
                valid_data_manager,
                test_data_manager,
                is_training=True
            )
        elif model_name == 'resnet50':
            return Resnet50Model(
                train_data_manager,
                valid_data_manager,
                test_data_manager,
                is_training=True
            )
        elif model_name == 'resnet50_stn':
            return Resnet50StnModel(
                train_data_manager,
                valid_data_manager,
                test_data_manager,
                is_training=True
            )
        elif model_name == 'mobilenet_stn':
            return MobilenetStnModel(
                train_data_manager,
                valid_data_manager,
                test_data_manager,
                is_training=True
            )
        elif model_name == 'resnet101':
            return Resnet101Model(
                train_data_manager,
                valid_data_manager,
                test_data_manager,
                is_training=True
            )
        elif model_name == 'resnet50_stn_with_2labels':
            return Resnet50StnModel2Labels(
                train_data_manager,
                valid_data_manager,
                test_data_manager,
                is_training=True
            )
        elif model_name == 'deeplabv3':
            return DeeplabV3(
                train_data_manager,
                valid_data_manager,
                test_data_manager,
                is_training=True
            )
        elif model_name == "inception_resnet_v2":
            return InceptionResnetv2(
                train_data_manager,
                valid_data_manager,
                test_data_manager,
                is_training=True
            )
        elif model_name == "efficientnet":
            return Efficientnet(
                train_data_manager,
                valid_data_manager,
                test_data_manager,
                is_training=True
            )
        elif model_name == "mal_function_model":
            return MalFunctionModel(
                train_data_manager,
                valid_data_manager,
                test_data_manager,
                is_training=True
            )

        raise ValueError("{} is not implemented at present".format(model_name))
        