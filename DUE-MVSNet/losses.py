import torch
from torch import nn

class SL1Loss(nn.Module):
    def __init__(self, levels=3):
        super(SL1Loss, self).__init__()
        self.levels = levels
        self.loss = nn.SmoothL1Loss(reduction='mean')

    def forward(self, inputs, targets, masks):
        loss = 0
        for l in range(self.levels):
            depth_pred_l = inputs[f'depth_{l}']
            depth_gt_l = targets[f'level_{l}']
            mask_l = masks[f'level_{l}']
            if l == 1:
                uncert_est = inputs[f'uncert_{l}']
                disp_loss = torch.abs(depth_pred_l[mask_l] - depth_gt_l[mask_l])
                uncert_loss = torch.exp(uncert_est[mask_l])
                mdist_loss = disp_loss / uncert_loss
                log_s = uncert_est[mask_l]
                loss += mdist_loss.mean()
                loss += log_s.mean()
                uc_loss = log_s.mean() + mdist_loss.mean()
            if l == 2:
                uncert_est = inputs[f'uncert_{l}']
                disp_loss = torch.abs(depth_pred_l[mask_l] - depth_gt_l[mask_l])
                uncert_loss = torch.exp(uncert_est[mask_l])
                mdist_loss = disp_loss / uncert_loss
                log_s = uncert_est[mask_l]
                loss += mdist_loss.mean()*0.5
                loss += log_s.mean()*0.5
                uc_loss = log_s.mean() + mdist_loss.mean() * 0.5

            loss += self.loss(depth_pred_l[mask_l], depth_gt_l[mask_l]) * 2**(1-l)
        return loss, uc_loss

loss_dict = {'sl1': SL1Loss}