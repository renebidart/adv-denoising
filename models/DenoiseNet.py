import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class DenoiseNet(nn.Module):
    """Join a classifier and a denoiser together

    If only one image is given, will return outputs from full denoiser, like a normal model
    If an adversarial image is specified, will output both original outputs and adversarial
    """
    def __init__(self, classifer, denoiser, loss):
        super(DenoiseNet, self).__init__()
        self.classifer = classifer
        self.denoiser = denoiser
        self.loss = loss

    def forward(self, orig_x, adv_x=None, requires_control=True, eval_mode=False):
        # get full denoiser model outputs on original image
        orig_out = self.classifer(orig_x)
        denoised_orig_x = self.denoiser(orig_x)
        denoised_orig_out = self.classifer(denoised_orig_x)

        # if adversarial image is given, output everything
        if (adv_x is not None):
            adv_out = self.classifer(adv_x)
            denoised_adv_x = self.denoiser(adv_x)
            denoised_adv_out = self.classifer(denoised_adv_x)

            # control loss is difference between adv_x and orig_x with no denoising
            if requires_control:
                control_loss = self.loss(adv_out, orig_out)
            loss = self.loss(denoised_adv_out, orig_out)
            
            # Must be a nicer way:
            if eval_mode:
                return orig_out, denoised_orig_out, adv_out, denoised_adv_out
            elif not requires_control:
                return orig_out, denoised_adv_out, loss
            else:
                return orig_out, denoised_adv_out, loss, control_loss

        return denoised_orig_out

