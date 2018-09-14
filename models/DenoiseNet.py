import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class DenoiseNet(nn.Module):
    """Join a classifier and a denoiser together"""
    def __init__(self, classifer, denoiser, loss):
        super(DenoiseNet, self).__init__()
        self.classifer = classifer
        self.denoiser = denoiser
        self.loss = loss

    def forward(self, orig_x, adv_x, requires_control=True, eval_mode=False):
        # get model outputs without any denoising
        orig_out = self.classifer(orig_x)
        adv_out = self.classifer(adv_x)

        # model output using the denoiser (no check on orig_x after denoise)
        denoised_adv_x = self.denoiser(adv_x)
        denoised_adv_out = self.classifer(denoised_adv_x)

        # how does denoiser perfom on non-adversarial samples
        denoised_orig_x = self.denoiser(orig_x)
        denoised_orig_out = self.classifer(denoised_orig_x)

        # control loss is difference between adv_x and orig_x with no denoising
        if requires_control:
            control_loss = self.loss(adv_out, orig_out)
        loss = self.loss(denoised_adv_out, orig_out)

        # if train:
            # need to add something to make sure gradients are kept on orig_out, adv_x ???

        if eval_mode:
            return orig_out, denoised_orig_out, adv_out, denoised_adv_out

        if not requires_control:
            return orig_out, denoised_adv_out, loss
        else:
            return orig_out, denoised_adv_out, loss, control_loss

