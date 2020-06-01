from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .market1501 import Market1501
from .dukemtmcreid import DukeMTMCreID
from .PA100K import PA100K



__imgreid_factory = {
    'market1501': Market1501,
    'dukemtmcreid': DukeMTMCreID,
    'pa100K':PA100K
}



def get_names():
    return list(__imgreid_factory.keys())


def init_imgreid_dataset(name, **kwargs):
    if name not in list(__imgreid_factory.keys()):
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, list(__imgreid_factory.keys())))
    return __imgreid_factory[name](**kwargs)

