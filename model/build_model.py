import torch.nn as nn
from .backbone import build_backbone


class build_model(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.opt = opt
        self.backbone = build_backbone('baseline', opt)

    def forward(self, composite_image, mask, fg_INR_coordinates):
        if self.opt.INRDecode and self.opt.hr_train and self.training:
            """
                For HR Training, due to the designed RSC strategy in Section 3.4 in the paper, 
                here we need to pass in the coordinates of the cropped regions.
            """
            extracted_features = self.backbone(composite_image, mask, fg_INR_coordinates)
        else:
            extracted_features = self.backbone(composite_image, mask)

        if self.opt.INRDecode:
            return extracted_features
        return None, None, extracted_features