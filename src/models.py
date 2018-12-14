from model_vgg16 import vgg16
from model_resnet50 import resnet50
from model_resnet152 import resnet152
from model_inceptionv3 import inceptionv3

def get_model(name, state_dict):
    """
        name can be:
            'vgg16',
            'resnet50',
            'resnet152',
            'inceptionv3'
    """
    model = None
    if name == "vgg16":
        model = vgg16
    elif name == "resnet50":
        model = resnet50
    elif name == "resnet152":
        model = resnet152
    elif name == "inceptionv3":
        model = inceptionv3

    return model(state_dict)