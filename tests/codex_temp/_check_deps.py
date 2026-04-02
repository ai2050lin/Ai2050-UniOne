import sklearn; print('sklearn', sklearn.__version__)
import scipy; print('scipy', scipy.__version__)
import torch; print('torch', torch.__version__)
import numpy; print('numpy', numpy.__version__)
try:
    import geoopt; print('geoopt', geoopt.__version__)
except:
    print('geoopt NOT installed')
