from .network_unet import UNetRes
import torch
from dataclasses import dataclass
from regularizers.denoise_reg import DenoiserStdMapCat, DenoiserTransCovCat, DenoiserFactory, DenoiserIID
from regularizers.denoise_reg import DenoiserIIDFromStdMapCat, DenoiserIIDFromTransCovCat
from regularizers.denoise_reg import DenoiserStdMapCaller, DenoiserTransCovCaller
from typing import Tuple

__all__ = [
    'DrunetFactory', 'DrunetSpecs'
]


@dataclass
class DrunetSpecs:
    name: str
    model_path: str
    out_nc: int
    in_std_nc: int
    in_img_nc: int
    in_img_trans_nc: int = 0
    inf_t1_t2_p: Tuple[float, float, float] = (float('inf'), float('inf'), 1.0)


drunet_versions = {
    'cst':
        DrunetSpecs('drunet-cst', model_path='./model_zoo/drunet_color_fixed_LR.pth',
                    in_img_nc=3, in_std_nc=1, out_nc=3),
    'cst-original':
        DrunetSpecs('drunet-cst', model_path='./model_zoo/drunet_color.pth',
                    in_img_nc=3, in_std_nc=1, out_nc=3),
    'cst-gray-original':
        DrunetSpecs('drunet-cst', model_path='./model_zoo/drunet_gray.pth',
                    in_img_nc=1, in_std_nc=1, out_nc=1),
    'var':
        DrunetSpecs('drunet-var', model_path='./model_zoo/drunet_color_var-aff_LR.pth',
                    in_img_nc=3, in_std_nc=1, out_nc=3),
    'var-rgb':
        DrunetSpecs('drunet-var-rgb', model_path='./model_zoo/drunet_color_var-col-aff_LR.pth',
                    in_img_nc=3, in_std_nc=3, out_nc=3),
    'dem':
        DrunetSpecs('drunet-dem', model_path='./model_zoo/drunet_color_var-col-dem_LR.pth',
                    in_img_nc=3, in_std_nc=3, out_nc=3),
}


class DrunetFactory(DenoiserFactory):

    def __init__(self, version_flag: str = 'cst', pretrained: bool = True):
        super(DrunetFactory, self).__init__()

        if version_flag not in drunet_versions:
            raise ValueError(f'Unkown drunet version flag \'{version_flag}\'. '
                             f'Possible version flags are: {list(drunet_versions.keys())}.')
        self.specs = drunet_versions[version_flag]
        self._pretrained = pretrained

    def denoiser_non_iid(self):
        if self.specs.in_img_trans_nc > 0:
            return DenoiserTransCovCat(
                self.raw_module(),
                self.specs.in_img_nc,
                self.specs.in_img_trans_nc,
                self.specs.in_std_nc,
                self.specs.name)
        else:
            return DenoiserStdMapCat(
                self.raw_module(),
                self.specs.in_img_nc,
                self.specs.in_std_nc,
                self.specs.name)

    def denoiser_iid(self) -> DenoiserIID:
        if self.specs.in_img_trans_nc > 0:
            return DenoiserIIDFromTransCovCat(
                self.raw_module(),
                self.specs.in_img_nc,
                self.specs.in_img_trans_nc,
                self.specs.in_std_nc,
                self.specs.name)
        else:
            return DenoiserIIDFromStdMapCat(
                self.raw_module(),
                self.specs.in_img_nc,
                self.specs.in_std_nc,
                self.specs.name)

    def denoiser_caller(self):
        if self.specs.in_img_trans_nc > 0:
            return DenoiserTransCovCaller(*self.specs.inf_t1_t2_p)
        else:
            return DenoiserStdMapCaller()

    def _create_raw_module(self):
        in_nc = self.specs.in_img_nc + self.specs.in_img_trans_nc + self.specs.in_std_nc
        model = UNetRes(in_nc=in_nc, out_nc=self.specs.out_nc, nc=[64, 128, 256, 512], nb=4, act_mode='R',
                        downsample_mode="strideconv", upsample_mode="convtranspose")
        if self._pretrained:
            # model.load_state_dict(torch.load(self.specs.model_path), strict=True)
            data = torch.load(self.specs.model_path, map_location=torch.device('cpu'))
            if 'model' in data:
                data = data['model']
            model.load_state_dict(data, strict=True)

        model.eval()
        for _, v in model.named_parameters():
            v.requires_grad = False

        return model
