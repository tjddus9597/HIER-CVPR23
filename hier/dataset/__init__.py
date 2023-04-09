from .cars import Cars
from .cub import CUBirds
from .SOP import SOP
from .base import BaseDataset


_type = {
    'cars': Cars,
    'cub': CUBirds,
    'SOP': SOP
}

def load(name, root, mode, transform = None):
    return _type[name](root = root, mode = mode, transform = transform)
    
