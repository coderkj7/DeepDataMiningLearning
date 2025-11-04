from typing import Callable, Dict, List, Optional, Union
import torch
from torch import nn, Tensor
import torchvision
import torch.nn.functional as F
from torchvision.ops import misc as misc_nn_ops
from torchvision.ops.feature_pyramid_network import ExtraFPNBlock, FeaturePyramidNetwork, LastLevelMaxPool
from torchvision.models import resnet #, resnet50, ResNet50_Weights
from torchvision.models import get_model, get_model_weights, get_weight, list_models
from collections import OrderedDict


def get_backbone(model_name: str,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 ):
    weights_enum = get_model_weights(model_name)
    weights = weights_enum.DEFAULT #IMAGENET1K_V1
    #weights = ResNet50_Weights.DEFAULT
    if model_name.startswith('resnet'):
        backbone = resnet.__dict__[model_name](weights=weights, norm_layer=norm_layer)
    elif model_name.startswith('swin'):
        backbone = get_model(model_name)
    else:
        backbone = get_model(model_name)

    return backbone
    # weights_backbone = ResNet50_Weights.verify(weights)
    # backbone = resnet50(weights=weights_backbone, progress=True)


class MyBackboneWithFPN(nn.Module):
    def __init__(
        self,
        model_name: str, #= 'resnet50'
        trainable_layers: int,
        #return_layers: Dict[str, str],
        #in_channels_list: List[int],
        out_channels: int = 256, #the number of channels in the FPN
        extra_blocks: Optional[ExtraFPNBlock] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.out_channels = out_channels
        
        if extra_blocks is None:
            extra_blocks = LastLevelMaxPool()

        if model_name.startswith('dino'):
            # --- DINO ViT-B/16 PATH ---
            
            # For DINO, 'trainable_layers' is interpreted as 'freeze_blocks'
            # We receive 8, meaning we freeze blocks 0-7 and train 8,9,10,11 (last 4)
            freeze_blocks = trainable_layers 
            
            self.vit_model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16') 
            self.vit_blocks = self.vit_model.blocks # 12 Transformer Blocks
            self.adapter = ViTFeatureAdapter(in_dim=768, out_channels_list=[512, 1024, 2048])
            
            # --- DINO Freezing Logic ---
            for param in self.vit_model.parameters():
                param.requires_grad = False
            # Unfreeze the last (12 - freeze_blocks) blocks
            for i, block in enumerate(self.vit_blocks):
                if i >= freeze_blocks:
                    for param in block.parameters():
                        param.requires_grad = True
            # Unfreeze adapter (it's new and must be trained)
            for param in self.adapter.parameters(): 
                param.requires_grad = True

            # Define the FPN
            in_channels_list = [512, 1024, 2048] # From adapter
            self.fpn = FeaturePyramidNetwork(
                in_channels_list=in_channels_list,
                out_channels=self.out_channels,
                extra_blocks=extra_blocks,
                norm_layer=norm_layer,
            )
            # Set the body to None as we have a custom forward
            self.body = None
        else:
            weights_enum = get_model_weights(model_name) #ResNet152_Weights
            weights = weights_enum.DEFAULT #ResNet152_Weights.IMAGENET1K_V2
            #weights = ResNet50_Weights.DEFAULT
            backbone = resnet.__dict__[model_name](weights=weights, norm_layer=norm_layer)
            # weights_backbone = ResNet50_Weights.verify(weights)
            # backbone = resnet50(weights=weights_backbone, progress=True)

            #trainable_layers =2
            layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][:trainable_layers] #trainable_layers=0=>layers_to_train=[]
            for name, parameter in backbone.named_parameters():
                if all([not name.startswith(layer) for layer in layers_to_train]):
                    parameter.requires_grad_(False)

            if extra_blocks is None:
                extra_blocks = LastLevelMaxPool()
            
            returned_layers = [1, 2, 3, 4]
            #return_layers (Dict[name, new_name]): a dict containing the names of the modules for which the activations will be returned as the key of the dict
            return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)} #{'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
            in_channels_stage2 = backbone.inplanes // 8 #2048//8=256
            #in_channels_list:List[int] number of channels for each feature map that is returned, in the order they are present in the OrderedDict
            in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
            #[256, 512, 1024, 2048]
            # BackboneWithFPN(
            #     backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks, norm_layer=norm_layer
            # )
            #return_layers={'layer1': 'feat1', 'layer3': 'feat2'} #[name, new_name]
            #https://github.com/pytorch/vision/blob/main/torchvision/models/_utils.py
            self.body = torchvision.models._utils.IntermediateLayerGetter(backbone, return_layers=return_layers)
            # >>> out = new_m(torch.rand(1, 3, 224, 224))
            #     >>> print([(k, v.shape) for k, v in out.items()])
            #     >>>     [('feat1', torch.Size([1, 64, 56, 56])),
            #     >>>      ('feat2', torch.Size([1, 256, 14, 14]))]

            self.fpn = FeaturePyramidNetwork(
                in_channels_list=in_channels_list,
                out_channels=out_channels,
                extra_blocks=extra_blocks,
                norm_layer=norm_layer,
            )
            self.out_channels = out_channels

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        if self.model_name.startswith('dino'):
            # --- DINO Forward Path ---
            H, W = x.shape[-2], x.shape[-1]
            # 1. Pass through ViT
            tokens = self.vit_model.prepare_tokens(x)
            for blk in self.vit_blocks:
                tokens = blk(tokens)
            # 2. Adapt tokens to 2D feature maps (C3, C4, C5)
            features = self.adapter(tokens, H, W)
            # 3. Pass through FPN
            x = self.fpn(features)
            
        else:
            x = self.body(x) #[16, 3, 800, 1344]
            x = self.fpn(x)
        return x
    
    #not used
    def create_fpnbackbone(self, backbone, trainable_layers):
        #backbone = get_model(backbone_modulename, weights="DEFAULT")
        trainable_layers =2
        layers_to_train = ["layer4", "layer3", "layer2", "layer1", "conv1"][:trainable_layers]
        for name, parameter in backbone.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)
        
        extra_blocks = LastLevelMaxPool()
        returned_layers = [1, 2, 3, 4]
        #return_layers (Dict[name, new_name]): a dict containing the names of the modules for which the activations will be returned as the key of the dict
        return_layers = {f"layer{k}": str(v) for v, k in enumerate(returned_layers)}
        in_channels_stage2 = backbone.inplanes // 8
        #in_channels_list:List[int] number of channels for each feature map that is returned, in the order they are present in the OrderedDict
        in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
        #the number of channels in the FPN
        out_channels = 256
        # BackboneWithFPN(
        #     backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks, norm_layer=norm_layer
        # )
        #return_layers={'layer1': 'feat1', 'layer3': 'feat2'} #[name, new_name]
        #https://github.com/pytorch/vision/blob/main/torchvision/models/_utils.py
        body = torchvision.models._utils.IntermediateLayerGetter(backbone, return_layers=return_layers)
        # >>> out = new_m(torch.rand(1, 3, 224, 224))
        #     >>> print([(k, v.shape) for k, v in out.items()])
        #     >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        #     >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
        
        fpn = FeaturePyramidNetwork(
            in_channels_list=in_channels_list,
            out_channels=out_channels,
            extra_blocks=extra_blocks,
            norm_layer=None,
        )
        return body, fpn


class ViTFeatureAdapter(nn.Module):
    """
    Takes ViT-B/16 (768-dim) tokens and projects them into 2D feature maps
    at different scales (C3, C4, C5) to simulate a CNN's hierarchy for the FPN.
    The output channels [512, 1024, 2048] are chosen to match the
    in_channels_list that the original ResNet-based FPN expects.
    """
    def __init__(self, in_dim=768, out_channels_list=[512, 1024, 2048]):
        super().__init__()
        self.patch_size = 16 
        self.in_dim = in_dim
        
        # 1x1 Convs to project 768-dim tokens to the channel sizes FPN expects
        self.conv_c3 = nn.Conv2d(in_dim, out_channels_list[0], kernel_size=1)
        self.conv_c4 = nn.Conv2d(in_dim, out_channels_list[1], kernel_size=1)
        self.conv_c5 = nn.Conv2d(in_dim, out_channels_list[2], kernel_size=1)

    def forward(self, tokens: Tensor, H: int, W: int) -> Dict[str, Tensor]:
        # Discard the [CLS] token (tokens shape: [B, N+1, C])
        tokens = tokens[:, 1:] 
        
        # Reshape tokens to 2D feature map: [B, C, H/P, W/P]
        H_feat, W_feat = H // self.patch_size, W // self.patch_size
        x = tokens.transpose(1, 2).reshape(tokens.shape[0], self.in_dim, H_feat, W_feat)
        
        # --- Create multi-scale feature maps ---
        c5_in = x
        c5_out = self.conv_c5(c5_in)

        c4_in = F.interpolate(c5_in, size=(H_feat * 2, W_feat * 2), mode='bilinear', align_corners=False)
        c4_out = self.conv_c4(c4_in)
        
        c3_in = F.interpolate(c4_in, size=(H_feat * 4, W_feat * 4), mode='bilinear', align_corners=False)
        c3_out = self.conv_c3(c3_in)

        # Return OrderedDict with the same keys as the ResNet path
        return OrderedDict([('0', c3_out), ('1', c4_out), ('2', c5_out)])



import os
try:
    from torchinfo import summary
except:
    print("[INFO] Couldn't find torchinfo... installing it.") #pip install -q torchinfo

def remove_classificationheader(model, num_removeblock):
    modulelist=model.children() #resnet50(pretrained=True).children()
    num_removeblock = 0-num_removeblock #-2
    newbackbone = nn.Sequential(*list(modulelist)[:num_removeblock])
    return newbackbone


if __name__ == "__main__":
    os.environ['TORCH_HOME'] = '/data/cmpe249-fa23/torchhome/'
    DATAPATH='/data/cmpe249-fa23/torchvisiondata/'

    #model_name = 'resnet50' #["layer4", "layer3", "layer2", "layer1", "conv1"]
    #model_name = 'resnet152' #["layer4", "layer3", "layer2", "layer1", "conv1"]
    #https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py
    model_name = 'swin_s' # 'avgpool','flatten','head'
    backbone = get_model(model_name, weights="DEFAULT")
    backbone=remove_classificationheader(backbone, 3)
    summary(model=backbone, 
        input_size=(1, 3, 64, 64), #(32, 3, 224, 224), # make sure this is "input_size", not "input_shape"
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
    ) 

    trainable_layers = 2
    out_channels = 256
    model = MyBackboneWithFPN(model_name,trainable_layers, out_channels)
    x=torch.rand(1,3,64,64) #image.tensors #[2, 3, 800, 1312] list of tensors x= torch.rand(1,3,64,64)
    output = model(x) 
    print([(k, v.shape) for k, v in output.items()])
    #[('0', torch.Size([1, 256, 16, 16])), ('1', torch.Size([1, 256, 8, 8])), ('2', torch.Size([1, 256, 4, 4])), ('3', torch.Size([1, 256, 2, 2])), ('pool', torch.Size([1, 256, 1, 1]))]
