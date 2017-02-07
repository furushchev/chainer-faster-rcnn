import numpy as np
from collections import OrderedDict

from chainer.links import VGG16Layers
import chainer.links as L


class VGG16Feature(VGG16Layers):

    def __init__(self):
        super(VGG16Feature, self).__init__()
        # delete layers that are not necessary
        self.delete_layers('pool5')

    def delete_layers(self, first_delete_layer):
        """
        Exclude layers starting from end
        """
        match_end = False
        for key, funcs in self.functions.items():
            if key == first_delete_layer or match_end:
                match_end = True
                self.functions.pop(key)
                if hasattr(self, key):
                    delattr(self, key)

    def __call__(self, x, layers=['conv5_3'], test=True):
        activations = super(VGG16LayersDepth, self).__call__(x, layers=layers, test=test)
        return activations


if __name__ == '__main__':
    model = VGG16Feature()
    import imageio
    from chainer.links.model.vision.vgg import prepare
    img = imageio.imread('~/projects/depth-est/data/a.png')
    out = model(prepare(img)[None])
