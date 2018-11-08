import platform
GPU = False

if platform.system() == 'Linux':
    GPU = True
    import cupy as np
else:
    import numpy as np
