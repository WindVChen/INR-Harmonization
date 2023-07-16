from utils import misc
from albumentations import Resize


class Implicit2DGenerator(object):
    def __init__(self, opt, mode):
        if mode == 'Train':
            sidelength = opt.INR_input_size
        elif mode == 'Val':
            sidelength = opt.input_size
        else:
            raise NotImplementedError

        self.mode = mode

        self.size = sidelength

        if isinstance(sidelength, int):
            sidelength = (sidelength, sidelength)

        self.mgrid = misc.get_mgrid(sidelength)

        self.transform = Resize(self.size, self.size)

    def generator(self, torch_transforms, composite_image, real_image, mask):
        composite_image = torch_transforms(self.transform(image=composite_image)['image'])
        real_image = torch_transforms(self.transform(image=real_image)['image'])

        fg_INR_RGB = composite_image.permute(1, 2, 0).contiguous().view(-1, 3)
        fg_transfer_INR_RGB = real_image.permute(1, 2, 0).contiguous().view(-1, 3)
        bg_INR_RGB = real_image.permute(1, 2, 0).contiguous().view(-1, 3)

        fg_INR_coordinates = self.mgrid
        bg_INR_coordinates = self.mgrid

        return fg_INR_coordinates, bg_INR_coordinates, fg_INR_RGB, fg_transfer_INR_RGB, bg_INR_RGB
