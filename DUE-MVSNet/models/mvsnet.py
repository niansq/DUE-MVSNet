from einops import reduce, rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import *

class FeatureNet(nn.Module):
    """
    output 3 levels of features using a FPN structure
    """
    def __init__(self, norm_act=InPlaceABN):
        super(FeatureNet, self).__init__()

        self.conv0 = nn.Sequential(
                        ConvBnReLU(3, 8, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(8, 8, 3, 1, 1, norm_act=norm_act))

        self.conv1 = nn.Sequential(
                        ConvBnReLU(8, 16, 5, 2, 2, norm_act=norm_act),
                        ConvBnReLU(16, 16, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(16, 16, 3, 1, 1, norm_act=norm_act))

        self.conv2 = nn.Sequential( 
                        ConvBnReLU(16, 32, 5, 2, 2, norm_act=norm_act),
                        ConvBnReLU(32, 32, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(32, 32, 3, 1, 1, norm_act=norm_act))

        self.toplayer = nn.Conv2d(32, 32, 1)
        self.lat1 = nn.Conv2d(16, 32, 1)
        self.lat0 = nn.Conv2d(8, 32, 1)

        # to reduce channel size of the outputs from FPN
        self.smooth1 = nn.Conv2d(32, 16, 3, padding=1)
        self.smooth0 = nn.Conv2d(32, 8, 3, padding=1)

    def _upsample_add(self, x, y):
        return F.interpolate(x, scale_factor=2, 
                             mode="bilinear", align_corners=True) + y

    def forward(self, x):
        # x: (B, 3, H, W)
        conv0 = self.conv0(x) # (B, 8, H, W)
        conv1 = self.conv1(conv0) # (B, 16, H//2, W//2)
        conv2 = self.conv2(conv1) # (B, 32, H//4, W//4)
        feat2 = self.toplayer(conv2) # (B, 32, H//4, W//4)
        feat1 = self._upsample_add(feat2, self.lat1(conv1)) # (B, 32, H//2, W//2)
        feat0 = self._upsample_add(feat1, self.lat0(conv0)) # (B, 32, H, W)

        # reduce output channels
        feat1 = self.smooth1(feat1) # (B, 16, H//2, W//2)
        feat0 = self.smooth0(feat0) # (B, 8, H, W)

        feats = {"level_0": feat0,
                 "level_1": feat1,
                 "level_2": feat2}

        return feats


class CostRegNet(nn.Module):
    def __init__(self, in_channels, norm_act=InPlaceABN):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(in_channels, 8, norm_act=norm_act)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2, norm_act=norm_act)
        self.conv2 = ConvBnReLU3D(16, 16, norm_act=norm_act)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2, norm_act=norm_act)
        self.conv4 = ConvBnReLU3D(32, 32, norm_act=norm_act)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2, norm_act=norm_act)
        self.conv6 = ConvBnReLU3D(64, 64, norm_act=norm_act)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(32))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(16))

        self.conv11 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, 3, padding=1, output_padding=1,
                               stride=2, bias=False),
            norm_act(8))

        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))

        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        del conv4
        x = conv2 + self.conv9(x)
        del conv2
        x = conv0 + self.conv11(x)
        del conv0
        x = self.prob(x)
        return x


class CascadeMVSNet(nn.Module):
    def __init__(self, n_depths=[8, 32, 48],
                       interval_ratios=[1, 2, 4],
                       num_groups=1,
                       norm_act=InPlaceABN,
                       est_uncert=True):
        super(CascadeMVSNet, self).__init__()
        self.levels = 3 # 3 depth levels
        self.n_depths = n_depths
        self.interval_ratios = interval_ratios
        self.G = num_groups # number of groups in groupwise correlation
        self.feature = FeatureNet(norm_act)
        for l in range(self.levels):
            if self.G > 1:
                cost_reg_l = CostRegNet(self.G, norm_act)
            else:
                cost_reg_l = CostRegNet(8*2**l, norm_act)
            setattr(self, f'cost_reg_{l}', cost_reg_l)

        self.est_uncert = est_uncert
        self.max_depth = 953.0
        if self.est_uncert:
            self.uncertdec = UncertaintyDecoder(maxdisp=self.max_depth)

        if self.G == 1:
            self.pixelwise_net = PixelwiseNet(in_channels=32)
        if self.G > 1:
            self.pixelwise_net = PixelwiseNet(in_channels=self.G)

        self.uncertainty_net_2 = REM(in_channels=self.n_depths[-1]) 
        self.uncertainty_net_1 = REM(in_channels=self.n_depths[1]) 



    def predict_depth(self, feats, proj_mats, depth_values, cost_reg, view_weights):
        # feats: (B, V, C, H, W)
        # proj_mats: (B, V-1, 3, 4)
        # depth_values: (B, D, H, W)
        # cost_reg: nn.Module of input (B, C, D, h, w) and output (B, 1, D, h, w)
        B, V, C, H, W = feats.shape
        D = depth_values.shape[1]

        ref_feats, src_feats = feats[:, 0], feats[:, 1:]
        src_feats = rearrange(src_feats, 'b vm1 c h w -> vm1 b c h w') # (V-1, B, C, h, w)
        proj_mats = rearrange(proj_mats, 'b vm1 x y -> vm1 b x y') # (V-1, B, 3, 4)

        ref_volume = rearrange(ref_feats, 'b c h w -> b c 1 h w')
        # ref_volume = repeat(ref_volume, 'b c 1 h w -> b c d h w', d=D) # (B, C, D, h, w)
        dtype, device = ref_volume.dtype, ref_volume.device
        if self.G == 1:
            # volume_sum = ref_volume
            volume_sum = torch.zeros((B, C, D, H, W), dtype=dtype, device=device)  # (B, C, D, H, W)
            # volume_sq_sum = ref_volume ** 2
            pixelwise_weight_sum = 1e-5 * torch.ones((B, 1, 1, H, W), dtype=dtype, device=device)
        else:
            ref_volume = ref_volume.view(B, self.G, C // self.G, 1, *ref_volume.shape[-2:])  # (B, G, C//G, 1, H, W)
            # volume_sum = 0
            volume_sum = torch.zeros((B, self.G, D, H, W), dtype=dtype, device=device)  # (B, G, D, H, W)
            pixelwise_weight_sum = 1e-5 * torch.ones((B, 1, 1, H, W), dtype=dtype, device=device)
        del ref_feats

        if view_weights == None:  # coarsest level

            pixelwise_net = self.pixelwise_net
            view_weights_list = []
            for src_feat, proj_mat in zip(src_feats, proj_mats):
                warped_volume = homo_warp(src_feat, proj_mat, depth_values)
                warped_volume = warped_volume.to(ref_volume.dtype)
                if self.G == 1:
                    warped_volume = warped_volume * ref_volume  # (B, C, D, H, W)*(B, C, 1, H, W)

                    view_weight = pixelwise_net(warped_volume)  # (B, 1, H, W)
                    view_weights_list.append(view_weight)
                else:
                    warped_volume = warped_volume.view(B, self.G, C // self.G, D, H, W)
                    similarity = (warped_volume * ref_volume).mean(2)  # (B, G, D, H, W)
                    view_weight = pixelwise_net(similarity)  # (B, 1, H, W)
                    view_weights_list.append(view_weight)
                if self.G == 1:
                    if self.training:
                        # volume_sum = volume_sum + warped_volume
                        volume_sum = volume_sum + warped_volume * view_weight.unsqueeze(1)
                        # volume_sq_sum = volume_sq_sum + warped_volume ** 2
                        pixelwise_weight_sum = pixelwise_weight_sum + view_weight.unsqueeze(1)
                    else:
                        # volume_sum += warped_volume
                        # volume_sq_sum += warped_volume.pow_(2)
                        volume_sum += warped_volume * view_weight.unsqueeze(1)
                        pixelwise_weight_sum += view_weight.unsqueeze(1)
                else:
                    # warped_volume = warped_volume.view_as(ref_volume)
                    if self.training:
                        # volume_sum = volume_sum + warped_volume # (B, G, C//G, D, h, w)
                        volume_sum = volume_sum + similarity * view_weight.unsqueeze(1)
                        pixelwise_weight_sum = pixelwise_weight_sum + view_weight.unsqueeze(1)
                    else:
                        # volume_sum += warped_volume
                        volume_sum += similarity * view_weight.unsqueeze(1)
                        pixelwise_weight_sum += view_weight.unsqueeze(1)
                del warped_volume, src_feat, proj_mat
            view_weights = torch.cat(view_weights_list, dim=1)  # [B, nview, H, W]
            volume_aggregated = volume_sum.div_(pixelwise_weight_sum)
            del src_feats, proj_mats

        else:
            idx = 0
            for src_feat, proj_mat in zip(src_feats, proj_mats):
                warped_volume = homo_warp(src_feat, proj_mat, depth_values)
                warped_volume = warped_volume.to(ref_volume.dtype)

                # warped_volume = warped_volume * ref_volume  # (B, C, D, H, W)*(B, C, 1, H, W)
                # view_weight = view_weights[:, idx].unsqueeze(1)
                if self.G == 1:
                    warped_volume = warped_volume * ref_volume  # (B, C, D, H, W)*(B, C, 1, H, W)
                    view_weight = view_weights[:, idx].unsqueeze(1)
                else:
                    warped_volume = warped_volume.view(B, self.G, C // self.G, D, H, W)
                    similarity = (warped_volume * ref_volume).mean(2)
                    view_weight = view_weights[:, idx].unsqueeze(1)
                idx += 1

                if self.G == 1:
                    if self.training:
                        # volume_sum = volume_sum + warped_volume
                        volume_sum = volume_sum + warped_volume * view_weight.unsqueeze(1)
                        # volume_sq_sum = volume_sq_sum + warped_volume ** 2
                        pixelwise_weight_sum = pixelwise_weight_sum + view_weight.unsqueeze(1)
                    else:
                        # volume_sum += warped_volume
                        # volume_sq_sum += warped_volume.pow_(2)
                        volume_sum += warped_volume * view_weight.unsqueeze(1)
                        pixelwise_weight_sum += view_weight.unsqueeze(1)
                else:
                    # warped_volume = warped_volume.view_as(ref_volume)
                    if self.training:
                        # volume_sum = volume_sum + warped_volume # (B, G, C//G, D, h, w)
                        volume_sum = volume_sum + similarity * view_weight.unsqueeze(1)
                        pixelwise_weight_sum = pixelwise_weight_sum + view_weight.unsqueeze(1)
                    else:
                        volume_sum = volume_sum + similarity * view_weight.unsqueeze(1)
                        pixelwise_weight_sum = pixelwise_weight_sum + view_weight.unsqueeze(1)

            volume_aggregated = volume_sum.div_(pixelwise_weight_sum)

        # aggregate multiple feature volumes by variance
        if self.G == 1:
            # volume_variance = volume_sq_sum.div_(V).sub_(volume_sum.div_(V).pow_(2))
            # del volume_sq_sum, volume_sum
            pass
        else:
            # volume_variance = reduce(volume_sum*ref_volume,
            #                          'b g c d h w -> b g d h w', 'mean').div_(V-1) # (B, G, D, h, w)
            del volume_sum, ref_volume

        # cost_reg = rearrange(cost_reg(volume_variance), 'b 1 d h w -> b d h w')
        cost_reg = rearrange(cost_reg(volume_aggregated), 'b 1 d h w -> b d h w')
        prob_volume = F.softmax(cost_reg, 1)  # (B, D, h, w)
        del cost_reg
        depth = depth_regression(prob_volume, depth_values)

        with torch.no_grad():
            # sum probability of 4 consecutive depth indices
            prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1),
                                                      pad=(0, 0, 0, 0, 1, 2)),
                                                (4, 1, 1), stride=1).squeeze(1)  # (B, D, h, w)
            # find the (rounded) index that is the final prediction
            depth_index = depth_regression(prob_volume,
                                           torch.arange(D,
                                                        device=prob_volume.device,
                                                        dtype=prob_volume.dtype)
                                           ).long()  # (B, h, w)
            depth_index = torch.clamp(depth_index, 0, D - 1)
            # the confidence is the 4-sum probability at this index
            confidence = torch.gather(prob_volume_sum4, 1,
                                      depth_index.unsqueeze(1)).squeeze(1)  # (B, h, w)

        return depth, confidence, view_weights.detach(), prob_volume

    def forward(self, imgs, proj_mats, init_depth_min, depth_interval):
        # imgs: (B, V, 3, H, W)
        # proj_mats: (B, V-1, self.levels, 3, 4) from fine to coarse
        # init_depth_min, depth_interval: (B) or float
        B, V, _, H, W = imgs.shape
        results = {}

        imgs = imgs.reshape(B*V, 3, H, W)
        feats = self.feature(imgs) # (B*V, 8, H, W), (B*V, 16, H//2, W//2), (B*V, 32, H//4, W//4)

        view_weights = None
        
        for l in reversed(range(self.levels)): # (2, 1, 0)
            feats_l = feats[f"level_{l}"] # (B*V, C, h, w)
            feats_l = feats_l.view(B, V, *feats_l.shape[1:]) # (B, V, C, h, w)
            proj_mats_l = proj_mats[:, :, l] # (B, V-1, 3, 4)
            depth_interval_l = depth_interval * self.interval_ratios[l]
            D = self.n_depths[l]
            if l == self.levels-1: # coarsest level
                h, w = feats_l.shape[-2:]
                if isinstance(init_depth_min, float):
                    depth_values = init_depth_min + depth_interval_l * \
                                   torch.arange(0, D,
                                                device=imgs.device,
                                                dtype=imgs.dtype) # (D)
                    depth_values = rearrange(depth_values, 'd -> 1 d 1 1')
                    depth_values = repeat(depth_values, '1 d 1 1 -> b d h w', b=B, h=h, w=w)
                else:
                    depth_values = init_depth_min + depth_interval_l * \
                                   rearrange(torch.arange(0, D,
                                                          device=imgs.device,
                                                          dtype=imgs.dtype),
                                             'd -> 1 d') # (B, D)
                    depth_values = rearrange(depth_values, 'b d -> b d 1 1')
                    depth_values = repeat(depth_values, 'b d 1 1 -> b d h w', h=h, w=w)
            else:
                depth_lm1 = depth_l.detach() # the depth of previous level
                depth_lm1 = F.interpolate(rearrange(depth_lm1, 'b h w -> b 1 h w'),
                                          scale_factor=2, mode='bilinear',
                                          align_corners=True) # (B, 1, h, w)

                uncertainty_map = F.interpolate(rearrange(uncertainty_map, 'b h w -> b 1 h w'), scale_factor=2,
                                                mode='bilinear', align_corners=True)

                depth_values = get_depth_values(depth_lm1, D, depth_interval_l, uncertainty_map)

                view_weights = F.interpolate(
                    view_weights,
                    scale_factor=2,
                    mode="nearest"
                )

                del depth_lm1
            depth_l, confidence_l, view_weights, prob_volume = self.predict_depth(feats_l, proj_mats_l, depth_values,
                                                       getattr(self, f'cost_reg_{l}'), view_weights=view_weights)

            if l == self.levels-1:
                uncertainty_map = self.uncertainty_net_2(prob_volume.detach()).squeeze(1)         
            if l == self.levels-2:
                uncertainty_map = self.uncertainty_net_1(prob_volume.detach()).squeeze(1)

            del feats_l, proj_mats_l, depth_values
            results[f"depth_{l}"] = depth_l
            results[f"confidence_{l}"] = confidence_l
            results[f'uncert_{l}'] = uncertainty_map  # 1 128 160    1 256  320    1 512 640


        pred2 = F.interpolate(rearrange(results[f'depth_2'], 'b h w -> b 1 h w'), scale_factor=4, mode='bilinear', align_corners=True).squeeze(1)
        pred1 = F.interpolate(rearrange(results[f'depth_1'], 'b h w -> b 1 h w'), scale_factor=2, mode='bilinear', align_corners=True).squeeze(1)
        pred0 = results[f'depth_0']

        if self.est_uncert:
            uncert = self.uncertdec([pred2, pred1, pred0])

            results['uncert'] = [uncert[0],uncert[1],uncert[2]]
            # print(uncert[0].shape,uncert[0].shape,uncert[2].shape)  # 1,512,640
            results['disp'] = [pred2, pred1, pred0]


            

        return results



'''
uncertainty estimate
'''
class UncertaintyDecoder(nn.Module):
    def __init__(self, maxdisp, num_scale=3):
        super(UncertaintyDecoder, self).__init__()
        self.num_scale = num_scale
        self.maxdisp = maxdisp
        self.idx_list = self.index_combinations(self.num_scale)
        self.input_len = len(self.idx_list)
        self.fc1 = nn.Linear(self.input_len, self.input_len*2)
        self.fc2 = nn.Linear(self.input_len*2, self.input_len)
        # self.fc3 = nn.Linear(self.input_len, 4)
        self.fc3 = nn.Linear(self.input_len, 3)
        self.act = nn.ELU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)

    def index_combinations(self,num_scales):
        L = []
        for i in range(num_scales):
            for j in range(i + 1, num_scales):
                L.append((i, j))
        return L

    def forward(self, disp_list):
        assert len(disp_list) == self.num_scale, \
            "Expected disp predictions from each scales"
        feature_list = []
        for i,j in self.idx_list:
            disp1,disp2 = disp_list[i]/self.maxdisp,disp_list[j]/self.maxdisp  # (b,w,h)
            feature_list.append((disp1-disp2)**2)

        disp_var = torch.stack(feature_list,dim=0)  #(6,b,w,h)  num_scale=3,(3, b, h ,w)
        #print(disp_var.shape)
        disp_var = disp_var.permute(1,2,3,0)
        #out = self.fc1(disp_var.cuda())
        out = self.fc1(disp_var)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = out.permute(3,0,1,2)
        #print(out.shape)  #3 4 512 640
        return out


class PixelwiseNet(nn.Module):
    """Pixelwise Net: A simple pixel-wise view weight network, composed of 1x1x1 convolution layers
    and sigmoid nonlinearities, takes the initial set of similarities to output a number between 0 and 1 per
    pixel as estimated pixel-wise view weight.
    1. The similarity is calculated by ref_feature and other source_features warped by differentiable_warping
    2. The learned pixel-wise view weight is estimated in the first iteration of Patchmatch and kept fixed in the
    matching cost computation.
    """

    def __init__(self, in_channels: int) -> None:
        """Initialize method
        Args:
            in_channels: the feature channels of input
        """
        super(PixelwiseNet, self).__init__()
        self.conv0 = ConvBnReLU3D(in_channels=in_channels, out_channels=16, kernel_size=1, stride=1, pad=0)
        self.conv1 = ConvBnReLU3D(in_channels=16, out_channels=8, kernel_size=1, stride=1, pad=0)
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.output = nn.Sigmoid()

    def forward(self, x1: torch.Tensor) -> torch.Tensor:
        """Forward method for PixelwiseNet
        Args:
            x1: pixel-wise view weight, [B, in_channels, Ndepth, H, W]
        """
        # [B, Ndepth, H, W]
        x1 = self.conv2(self.conv1(self.conv0(x1))).squeeze(1)

        output = self.output(x1)
        del x1
        # [B,H,W]
        output = torch.max(output, dim=1)[0]

        return output.unsqueeze(1)










class Conv2d(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)

def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return


def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == "kaiming":
            nn.init.kaiming_uniform_(module.weight)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(module.weight)
    return