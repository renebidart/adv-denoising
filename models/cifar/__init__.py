"""
lenet, vggnet, resnet, wide_resnet implementations are all based off or exactly identical to: 
https://github.com/meliketoy/wide-resnet.pytorch

Designed for 32x32 images, not exactly the same as the standard imagenet models.
Couple changes to try to make the resnet, vgg, wide-rn work for 64x64 ones using int(self.IM_SIZE/32)**2
"""
                                                                     
from .lenet import *
from .vggnet import *
from .resnet import *
from .wide_resnet import *

