from .base import *
from . import bo

import os as _os
SICK_ROOT_DIR =  _os.path.dirname(__file__)

if 'SICK_CACHE_DIR' in _os.environ:
    SICK_CACHE_DIR = _os.environ['SICK_CACHE_DIR']
else:
    SICK_CACHE_DIR = _os.path.join(_os.path.dirname(SICK_ROOT_DIR), ".cache")
    if not _os.path.exists(SICK_CACHE_DIR):
        _os.mkdir(SICK_CACHE_DIR)
