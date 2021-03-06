from __future__ import absolute_import

from .resnet import *

from .resnetAttW2V import ResNet50AttW2VText,ResNet50AttW2VAttribute



__model_factory = {
    'resnet50': ResNet50,
    'resnetAttW2VText': ResNet50AttW2VText,
    'resnetAttW2VAttributes': ResNet50AttW2VAttribute
}


def get_names():
    return list(__model_factory.keys())


def init_model(name, *args, **kwargs):
    if name not in list(__model_factory.keys()):
        raise KeyError("Unknown model: {}".format(name))
    return __model_factory[name](*args, **kwargs)
