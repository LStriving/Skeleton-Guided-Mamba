import math
import os
from collections import OrderedDict
from einops import rearrange
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet50_Weights

from fairscale.nn.checkpoint import checkpoint_wrapper

from libs.modeling.videomamba import VisionMamba

from .models import register_video_stem, register_image_stem
from .blocks import MaskMambaBlock, MaskedPoolingMHCA, MaskedPoolingMHCAv2, MaskMultiScaleMambaBlock
from .videomamba import PatchEmbed
from pytorch_i3d import InceptionI3d

@register_image_stem('resnet50')
class ResNet50(nn.Module):
    '''
        ResNet50 image stem
    '''
    def __init__(
        self, 
        **kwargs
    ):
        super().__init__()
        self.module = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.module.fc = nn.Identity()
    
    def forward(self, x):
        channel_dim = x.shape[1]
        if channel_dim == 1:
            x = x.repeat(1,3,1,1)
        return self.module(x) # output dim: 2048
    
@register_image_stem('resnet50_projector')
class ResNet50Projector(nn.Module):
    '''
        ResNet50 image stem with projection head
    '''
    def __init__(
        self, 
        out_dim,
        **kwargs
    ):
        super().__init__()
        self.module = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.module.fc = nn.Linear(2048, out_dim)

    def forward(self, x):
        channel_dim = x.shape[1]
        if channel_dim == 1:
            x = x.repeat(1,3,1,1)
        return self.module(x)

@register_image_stem('projector')
class Projector(nn.Module):
    '''
        Projector image stem
    '''
    def __init__(
        self, 
        in_dim,
        out_dim,
        **kwargs
    ):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        return self.proj(x)


class SpatialNet(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        num_layers: int,
        pool_size: int,
        pool_mode: str,
        abs_pos_embed: bool = False,
        pos_drop: float = 0.0,
        act_checkpoint: bool = False,
        *args, 
        **kwargs
    ):
        super().__init__()
        self.wrapper = checkpoint_wrapper if act_checkpoint else lambda x: x
        self.patch_size = patch_size
        self.image_size = image_size
        self.embed_dim = embed_dim
        self.pool_size = pool_size
        self.pool_mode = pool_mode
        self.conv_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        

        # init
        fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
        nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
        if self.conv_proj.bias is not None:
            nn.init.zeros_(self.conv_proj.bias)

        self.conv_proj = self.wrapper(self.conv_proj)

        self.n_layers = num_layers
        self.blocks = nn.ModuleList()
        
        num_patches = (image_size // patch_size) ** 2
        
        self.abs_pos_embed = abs_pos_embed
        if self.abs_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=.02)
            self.pos_drop = nn.Dropout(p=pos_drop)


    def _process_image(self, x):
        n, c, h, w = x.shape
        torch._assert(h == w == self.image_size, f"Input image size must be {self.image_size}x{self.image_size}")
        torch._assert(h % self.patch_size == 0, f"Image size must be divisible by patch size {self.patch_size}")
        n_w = w // self.patch_size
        n_h = h // self.patch_size

        
        x = self.conv_proj(x)
        x = x.reshape(n, self.embed_dim, n_h * n_w)
        x = x.permute(0, 2, 1)
        
        if self.abs_pos_embed:
            x = x + self.pos_embed
            x = self.pos_drop(x)
        return x


    def forward(self, x):
        # x: (B, C, H, W)
        x = self._process_image(x)
        # x: (B, N, C) where N is the number of patches

        x = x.transpose(1, 2)
        mask = torch.ones((x.shape[0], 1, x.shape[-1]), device=x.device).bool()
        for i in range(self.n_layers):
            x, mask = self.blocks[i](x, mask)

        # desired output shape: (B, C)
        x = x.squeeze(-1)
        return x

@register_image_stem('s-mvit')
class SpatialMViT(SpatialNet):
    def __init__(
            self,
            image_size: int,
            patch_size: int,
            in_channels: int,
            embed_dim: int,
            num_heads: int,
            attn_dropout: float,
            num_layers: int,
            pool_size: int,
            pool_mode: str,
            abs_pos_embed: bool = False,
            pos_drop: float = 0.0,
            act_checkpoint: bool = False,
            *args, 
            **kwargs) -> None:
        super().__init__(
            image_size, 
            patch_size, 
            in_channels, 
            embed_dim, 
            num_layers, 
            pool_size, 
            pool_mode, 
            abs_pos_embed,
            pos_drop, 
            act_checkpoint, 
            *args, 
            **kwargs
        )
        
        for _ in range(num_layers):
            self.blocks.append(
                self.wrapper(MaskedPoolingMHCA(
                    n_embd=embed_dim,
                    n_head=num_heads,
                    attn_pdrop=attn_dropout,
                    pool_qx_size=pool_size,
                    pool_kv_size=pool_size,
                    pool_method=pool_mode
                ))
            )

@register_image_stem('s-mvitV2')
class SpatialMViT_v2(SpatialNet):
    def __init__(self, 
                 image_size: int, 
                 patch_size: int, 
                 in_channels: int, 
                 embed_dim: int, 
                 num_heads: int, 
                 attn_dropout: float, 
                 num_layers: int, 
                 pool_size: int, 
                 pool_mode: str, 
                 abs_pos_embed: bool = False, 
                 pos_drop: float = 0.0, 
                 act_checkpoint: bool = False, 
                 *args, 
                 **kwargs) -> None:
        super().__init__(image_size, 
                         patch_size, 
                         in_channels, 
                         embed_dim, 
                         num_layers, 
                         pool_size, 
                         pool_mode, 
                         abs_pos_embed,
                         pos_drop,
                         act_checkpoint, 
                         *args, 
                         **kwargs)
        
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(
                self.wrapper(MaskedPoolingMHCAv2(
                    n_embd=embed_dim,
                    n_head=num_heads,
                    attn_pdrop=attn_dropout,
                    pool_qx_size=pool_size,
                    pool_kv_size=pool_size,
                    pool_method=pool_mode
                ))
            )
    

@register_image_stem('MVMamba')
class SpatialVMamba(SpatialNet):
    def __init__(
        self,
        drop_path_rate: float = 0.3,
        gamma: float = 0.0, 
        mamba_type: str = 'mdbm',
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.blocks = nn.ModuleList()
        for _ in range(self.n_layers):
            self.blocks.append(
                self.wrapper(MaskMultiScaleMambaBlock(
                    n_embd=self.embed_dim,
                    drop_path_rate=drop_path_rate,
                    n_ds_stride=self.pool_size,
                    pool_method=self.pool_mode,
                    gamma=gamma,
                    use_mamba_type=mamba_type.lower()
                ))
            )

@register_image_stem('postpoolVMamba')
class PostPoolVMamba(SpatialNet):
    def __init__(
        self,
        drop_path_rate: float = 0.3,
        mamab_type: str = 'dbm',
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.blocks = nn.ModuleList()
        for _ in range(self.n_layers):
            self.blocks.append(
                self.wrapper(MaskMambaBlock(
                    n_embd=self.embed_dim,
                    n_ds_stride=self.pool_size,
                    pool_method=self.pool_mode,
                    drop_path_rate=drop_path_rate,
                    use_mamba_type=mamab_type.lower()
                ))
            )

@register_video_stem('i3d')
class Inception3D(nn.Module):
    '''
        I3D video stem
    '''
    def __init__(
        self,
        in_channels: int,
        dropout_prob: float,
        pretrained: str,
        act_checkpoint: bool,
        **kwargs
    ):
        super().__init__()
        self.i3d = InceptionI3d(num_classes=7, in_channels=in_channels, dropout_keep_prob=dropout_prob, **kwargs)
        self.wrapper = checkpoint_wrapper if act_checkpoint else lambda x: x
        self.i3d = self.wrapper(self.i3d)
        if pretrained:
            self._init_from_pretrained(pretrained)

    def _init_from_pretrained(self, pretrained):
        os.path.exists(pretrained), f"Pretrained model {pretrained} does not exist."
        ckpt = torch.load(pretrained)['state_dict']
        new_ckpt = OrderedDict()
        for k, v in ckpt.items():
            new_ckpt[k.replace('module.', '')] = v
        self.i3d.load_state_dict(new_ckpt)
        print(f"I3D: Load pretrained model from {pretrained}")

    def forward(self, x):
        return self.i3d.extract_features(x)[:,:,0,0,0]

@register_image_stem('posec3d')
class PoseC3D(nn.Module):
    '''
        PoseC3D image stem
    '''
    def __init__(
        self, 
        **kwargs
    ):
        super().__init__()
        pass

@register_video_stem('tcn')
class TCN(nn.Module):
    '''
        TCN video stem
    '''
    def __init__(
        self, 
        **kwargs
    ):
        super().__init__()
        pass


class TemporalNet(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        pool_size: int,
        pool_mode: str,
        act_checkpoint: bool,
        *args, 
        **kwargs
    ):
        super().__init__()
        self.wrapper = checkpoint_wrapper if act_checkpoint else lambda x: x
        self.embed_dim = embed_dim
        self.n_layers = num_layers
        self.pool_size = pool_size
        self.pool_mode = pool_mode
        self.blocks = nn.ModuleList()
    
    def forward(self, x, mask=None):
        # input shape: (B, C, T) desired output shape: (B, C) 
        ## T is the frame length within a segment/window
        if mask is None:
            b, c, t = x.shape
            mask = torch.ones((b,1,t), device=x.device)

        for i in range(self.n_layers):
            x, mask = self.blocks[i](x, mask)
            
        return x, mask

@register_video_stem('mvit')
class MViT(TemporalNet):
    '''
        MViT video stem
        Refer to  https://arxiv.org/pdf/2104.11227
    '''
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: float,
        num_layers: int,
        pool_size: int,
        pool_mode: str,
        act_checkpoint: bool,
        **kwargs
    ):
        super().__init__(
            embed_dim, 
            num_layers, 
            pool_size, 
            pool_mode, 
            act_checkpoint, 
            **kwargs
        )
        for _ in range(num_layers):
            self.blocks.append(
                self.wrapper(MaskedPoolingMHCA(
                    n_embd=embed_dim,
                    n_head=num_heads,
                    attn_pdrop=attn_dropout,
                    pool_qx_size=pool_size,
                    pool_kv_size=pool_size,
                    pool_method=pool_mode
                ))
            )

@register_video_stem('MMamba')
class MMamba(TemporalNet):
    def __init__(
        self,
        drop_path_rate: float = 0.3,
        gamma: float = 0.1,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.blocks = nn.ModuleList()
        for _ in range(self.n_layers):
            self.blocks.append(
                self.wrapper(MaskMultiScaleMambaBlock(
                    n_embd=self.embed_dim,
                    drop_path_rate=drop_path_rate,
                    n_ds_stride=self.pool_size,
                    pool_method=self.pool_mode,
                    gamma=gamma
                ))
            )

@register_video_stem('postpoolMamba')
class PostPoolMamba(TemporalNet):
    def __init__(
        self,
        drop_path_rate: float = 0.3,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.blocks = nn.ModuleList()
        for _ in range(self.n_layers):
            self.blocks.append(
                self.wrapper(MaskMambaBlock(
                    n_embd=self.embed_dim,
                    n_ds_stride=self.pool_size,
                    drop_path_rate=drop_path_rate,
                    pool_method=self.pool_mode
                ))
            )

@register_video_stem('mvitV2')
class MViT_v2(TemporalNet):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: float,
        num_layers: int,
        pool_size: int,
        pool_mode: str,
        act_checkpoint: bool,
        **kwargs
    ):
        super().__init__(
            embed_dim, 
            num_layers, 
            pool_size, 
            pool_mode, 
            act_checkpoint, 
            **kwargs
        )
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(
                self.wrapper(MaskedPoolingMHCAv2(
                    n_embd=embed_dim,
                    n_head=num_heads,
                    attn_pdrop=attn_dropout,
                    pool_qx_size=pool_size,
                    pool_kv_size=pool_size,
                    pool_method=pool_mode
                ))
            )

class SpatialTemporalNet(nn.Module):
    """
    SpatialTemporalNet is a base class for all spatial temporal networks.
    Using the learnable positional embeddings
    """
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        num_frames: int,
        num_layers: int,
        pos_drop: float,
        act_checkpoint: bool,
        *args, 
        **kwargs
    ):
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size=image_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
        )
        self.embed_dim = embed_dim
        self.num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.embed_dim))
        self.pos_drop = nn.Dropout(p=pos_drop)
        self.wrapper = checkpoint_wrapper if act_checkpoint else lambda x: x
        self.n_layers = num_layers
        self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
        self.blocks = nn.ModuleList()
        self.use_mask = False

    def forward_features(self, x):
        x = self.patch_embed(x)
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed

        # temporal pos
        cls_tokens = x[:B, :1, :]
        x = x[:, 1:]
        x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
        x = x + self.temporal_pos_embedding
        x = rearrange(x, '(b n) t m -> b (t n) m', b=B, t=T)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.pos_drop(x)
        # x: B, N, D
        x = x.transpose(1, 2)

        if self.use_mask:
            mask = torch.ones((x.shape[0], 1, x.shape[-1]), device=x.device).bool()
            

        for i in range(self.n_layers):
            if self.use_mask:
                x, mask = self.blocks[i](x, mask)
            else:
                x = self.blocks[i](x)
        
        # desired output shape: (B, C)
        x = x.squeeze(-1)

        return x

@register_video_stem("video_mvit")
class VideoMViT(SpatialTemporalNet):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        embed_dim: int,
        in_channels: int,
        num_frames: int,
        num_heads: int,
        attn_dropout: float,
        pool_size: int,
        pool_mode: str,
        pos_drop: float,
        act_checkpoint: bool,
        **kwargs
    ):
        super().__init__(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_frames=num_frames,
            pos_drop=pos_drop,
            act_checkpoint=act_checkpoint,
        )
        self.num_frames = num_frames
        for _ in range(num_layers):
            self.blocks.append(
                self.wrapper(MaskedPoolingMHCA(
                    n_embd=embed_dim,
                    n_head=num_heads,
                    attn_pdrop=attn_dropout,
                    pool_qx_size=pool_size,
                    pool_kv_size=pool_size,
                    pool_method=pool_mode
                ))
            )
            self.use_mask = True
    
    def forward(self, x):
        
        return self.forward_features(x)


@register_video_stem('video_mamba')
class VideoMamba(VisionMamba):
    '''
        VideoMamba video stem
    '''
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_layers: int,
        embed_dim: int,
        in_channels: int,
        num_frames: int,
        pos_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        rms_norm: bool = True,
        fused_add_norm: bool = True,
        feature_output_method: str = 'mean',
        act_checkpoint: bool = True,
        device: str = 'cuda',
        **kwargs
    ):
        super().__init__(
            image_size, 
            patch_size, 
            num_layers, 
            embed_dim, 
            in_channels, 
            num_classes=1,
            drop_rate=pos_drop_rate,
            drop_path_rate=drop_path_rate,
            num_frames=num_frames, 
            rms_norm=rms_norm,
            fused_add_norm=fused_add_norm,
            feature_output_method = feature_output_method, 
            use_checkpoint=act_checkpoint, 
            checkpoint_num=num_layers if act_checkpoint else 0,
            device=device,
            **kwargs
        )
        self.head = nn.Identity()
        