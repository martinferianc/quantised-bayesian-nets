from src.models.pointwise.models_p import LinearNetwork, ConvNetwork_LeNet, ConvNetwork_ResNet
from src.models.stochastic.mcdropout.models_mc import LinearNetwork as LinearNetworkMC
from src.models.stochastic.mcdropout.models_mc import ConvNetwork_LeNet as ConvNetwork_LeNetMC
from src.models.stochastic.mcdropout.models_mc import ConvNetwork_ResNet as ConvNetwork_ResNetMC
from src.models.stochastic.bbb.models_bbb import LinearNetwork as LinearNetworkBBB
from src.models.stochastic.bbb.models_bbb import ConvNetwork_LeNet as ConvNetwork_LeNetBBB
from src.models.stochastic.bbb.models_bbb import ConvNetwork_ResNet as ConvNetwork_ResNetBBB
from src.models.stochastic.sgld.models_sgld import Network as NetworkSGLD


class ModelFactory():
    def __init__(self):
        pass 

    @staticmethod
    def get_model(model, input_size, output_size,  q, args, training_mode=True):
        net = None
        if model == "linear":
            net = LinearNetwork(input_size, output_size,  q, args)
        elif model == "conv_lenet":
            net = ConvNetwork_LeNet(input_size, output_size,   q, args)
        elif model == "conv_resnet":
            net = ConvNetwork_ResNet(input_size, output_size,   q, args)
        elif model == "linear_mc":
            net = LinearNetworkMC(input_size, output_size,  q, args)
        elif model == "conv_lenet_mc":
            net = ConvNetwork_LeNetMC(input_size, output_size, q, args)
        elif model == "conv_resnet_mc":
            net = ConvNetwork_ResNetMC(input_size, output_size,   q, args)
        elif model == "linear_bbb":
            net = LinearNetworkBBB(input_size, output_size,  q, args)
        elif model == "conv_lenet_bbb":
            net = ConvNetwork_LeNetBBB(input_size, output_size,   q, args)
        elif model == "conv_resnet_bbb":
            net = ConvNetwork_ResNetBBB(input_size, output_size,   q, args)
        elif "sgld" in model:
            net = NetworkSGLD(input_size, output_size,   q, args, training_mode)
        else:
            raise NotImplementedError("Other models not implemented")

        return net 