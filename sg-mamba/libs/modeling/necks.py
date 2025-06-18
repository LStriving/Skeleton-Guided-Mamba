import torch
from torch import nn
from torch.nn import functional as F

from .models import register_neck
from .blocks import MaskedConv1D, LayerNorm



@register_neck("fpn")
class FPN1D(nn.Module):
    """
        Feature pyramid network
    """
    def __init__(
        self,
        in_channels,      # input feature channels, len(in_channels) = # levels
        out_channel,      # output feature channel
        scale_factor=2.0, # downsampling rate between two fpn levels
        start_level=0,    # start fpn level
        end_level=-1,     # end fpn level
        with_ln=True      # if to apply layer norm at the end
    ):
        super().__init__()
        assert isinstance(in_channels, list) or isinstance(in_channels, tuple)

        self.in_channels = in_channels
        self.out_channel = out_channel
        self.scale_factor = scale_factor

        self.start_level = start_level
        if end_level == -1:
            self.end_level = len(in_channels)
        else:
            self.end_level = end_level
        assert self.end_level <= len(in_channels)
        assert (self.start_level >= 0) and (self.start_level < self.end_level)

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.fpn_norms = nn.ModuleList()
        for i in range(self.start_level, self.end_level):
            # disable bias if using layer norm
            l_conv = MaskedConv1D(
                in_channels[i], out_channel, 1, bias=(not with_ln))
            # use depthwise conv here for efficiency
            fpn_conv = MaskedConv1D(
                out_channel, out_channel, 3,
                padding=1, bias=(not with_ln), groups=out_channel
            )
            # layer norm for order (B C T)
            if with_ln:
                fpn_norm = LayerNorm(out_channel)
            else:
                fpn_norm = nn.Identity()

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
            self.fpn_norms.append(fpn_norm)

    def forward(self, inputs, fpn_masks):
        # inputs must be a list / tuple
        assert len(inputs) == len(self.in_channels)
        assert len(fpn_masks) ==  len(self.in_channels)

        # build laterals, fpn_masks will remain the same with 1x1 convs
        laterals = []
        for i in range(len(self.lateral_convs)):
            x, _ = self.lateral_convs[i](
                inputs[i + self.start_level], fpn_masks[i + self.start_level]
            )
            laterals.append(x)

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i-1] += F.interpolate(
                laterals[i],
                scale_factor=self.scale_factor,
                mode='nearest'
            )

        # fpn conv / norm -> outputs
        # mask will remain the same
        fpn_feats = tuple()
        for i in range(used_backbone_levels):
            x, _ = self.fpn_convs[i](
                laterals[i], fpn_masks[i + self.start_level])
            x = self.fpn_norms[i](x)
            fpn_feats += (x, )

        return fpn_feats, fpn_masks

@register_neck('identity')
class FPNIdentity(nn.Module):
    def __init__(
        self,
        in_channels,      # input feature channels, len(in_channels) = # levels
        out_channel,      # output feature channel
        scale_factor=2.0, # downsampling rate between two fpn levels
        start_level=0,    # start fpn level
        end_level=-1,     # end fpn level
        with_ln=True      # if to apply layer norm at the end
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channel = out_channel
        self.scale_factor = scale_factor

        self.start_level = start_level
        if end_level == -1:
            self.end_level = len(in_channels)
        else:
            self.end_level = end_level
        assert self.end_level <= len(in_channels)
        assert (self.start_level >= 0) and (self.start_level < self.end_level)

        self.fpn_norms = nn.ModuleList()
        for i in range(self.start_level, self.end_level):
            # check feat dims
            assert self.in_channels[i + self.start_level] == self.out_channel
            # layer norm for order (B C T)
            if with_ln:
                fpn_norm = LayerNorm(out_channel)
            else:
                fpn_norm = nn.Identity()
            self.fpn_norms.append(fpn_norm)

    def forward(self, inputs, fpn_masks):
        # inputs must be a list / tuple
        assert len(inputs) == len(self.in_channels)
        assert len(fpn_masks) ==  len(self.in_channels)

        # apply norms, fpn_masks will remain the same with 1x1 convs
        fpn_feats = tuple()
        for i in range(len(self.fpn_norms)):
            x = self.fpn_norms[i](inputs[i + self.start_level])
            fpn_feats += (x, )

        return fpn_feats, fpn_masks


@register_neck('fusion_fpn')
class FusionFPN(nn.Module):
    '''
    used for fusion of two towers to reduce the feature dimension
    '''
    def __init__(
        self,
        in_channels,      # input feature channels, len(in_channels) = # levels
        out_channel,      # output feature channel
        start_level=0,    # start fpn level
        end_level=-1,     # end fpn level
        activation='none' # activation function
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channel = out_channel
       
        # Using 1x1 conv to reduce the dimension
        self.fusion_convs = nn.ModuleList()
        for i in range(start_level, end_level):
            fusion_conv = MaskedConv1D(
                in_channels[i], out_channel, 1
            )
            self.fusion_convs.append(fusion_conv)

        # activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none' or activation is None:
            self.activation = nn.Identity()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(inplace=True)
        else:
            raise ValueError(f"Unknown activation function: {activation}")
        
    def forward(self, inputs, fpn_masks):
        # inputs must be a list / tuple
        assert len(inputs) == len(self.in_channels)
        assert len(fpn_masks) == len(self.in_channels), f"{len(fpn_masks)} != {len(self.in_channels)}"

        # apply 1x1 convs
        fused_feats = tuple()
        out_masks = tuple()
        for i in range(len(self.fusion_convs)):
            x, out_mask = self.fusion_convs[i](inputs[i], fpn_masks[i])
            x = self.activation(x)
            fused_feats += (x, )
            out_masks += (out_mask, )

        return fused_feats, out_masks