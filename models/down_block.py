import torch
from torch import nn

class DownBlock(torch.nn.Module):
    """
    input: x and t_emb 
    iterate num_layers times:
        ├─ ResNet block:
        │   ├─ resnet_conv1[i] 
        │   ├─ + t_emb_layers[i](t_emb) (combine time embedding)
        │   ├─ resnet_conv2[i] 
        │   └─ + residual_input_conv[i](x) (residual connection)
        │
        └─ Attention block:
            ├─ reshape ==> attention_norms[i]
            ├─ transpose ==> attentions[i] 
            └─ transpose + reshape 

    After iterating through num_layers:
        └─ down_sample_conv (if down_sample is True)
    """  
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, down_sample: bool = True, num_heads:int = 4, num_layers: int = 1):
        super().__init__()
        self.num_layers = num_layers
        self.down_sample = down_sample
        
        self.resnet_conv1 = nn.ModuleList()
        self.t_emb_layers = nn.ModuleList()
        self.resnet_conv2 = nn.ModuleList()
        self.attention_norms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.residual_input_conv = nn.ModuleList() 
        
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            
            # ResNet first conv: GroupNorm ==> SiLU ==> Conv2d
            self.resnet_conv1.append(
                nn.Sequential(
                    nn.GroupNorm(8, in_ch),
                    nn.SiLU(),
                    nn.Conv2d(in_ch, out_channels, kernel_size=3, stride=1, padding=1)
                )
            )

            # time embedding layers: siLU ==> Linear
            self.t_emb_layers.append(
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(time_emb_dim, out_channels)
                )
            )

            # ResNet second conv: GroupNorm ==> SiLU ==> Conv2d
            self.resnet_conv2.append(
                nn.Sequential(
                    nn.GroupNorm(8, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
                )
            )

            # attention normalization (This is the normalization layer before the attention layer): 
            self.attention_norms.append(nn.GroupNorm(8, out_channels))

            # Multihead attention layer:
            self.attention_layers.append(nn.MultiheadAttention(out_channels, num_heads, batch_first=True))

            # ResNet 1x1 conv: This is the residual connection to add the input x to the output of the ResNet block. 
            # This is used to match the number of channels of the input x and the output of the ResNet block.
            self.residual_input_conv.append(nn.Conv2d(in_ch, out_channels, kernel_size=1))
            
            self.down_sample_conv = nn.Conv2d(out_channels, out_channels,
                                          4, 2, 1) if self.down_sample else nn.Identity()
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        out = x  # original: (B, in_channels, H, W)

        for i in range(self.num_layers):
            # =================================================================
            # ResNet Block                                                    #
            # =================================================================
            resnet_input = out  # keep the original input for residual connection
            
            # resnet_conv1: GroupNorm ==> SiLU ==> Conv2d
            # input shape: (B, in_channels, H, W) (if i == 1) or (B, out_channels, H, W) ==> output shape: (B, out_channels, H, W)
            out = self.resnet_conv1[i](out)  
            
            # time-embedding combination: SiLU ==> Linear ==> broadcast to feature map
            # t_emb shape: (B, t_emb_dim) ==> output shape: (B, out_channels) ==> broadcast to (B, out_channels, 1, 1)
            t_emb_proj = self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = out + t_emb_proj  # feature map + time embedding info
            
            # resnet_conv2: GroupNorm → SiLU → Conv2d
            # input/output shape: (B, out_channels, H, W)
            out = self.resnet_conv2[i](out)  
            
            # We use residual_input_conv to adjust the number of channels
            # input/output shape: (B, out_channels, H, W)
            out = out + self.residual_input_conv[i](resnet_input)

            # =================================================================
            # Attention Block
            # =================================================================
            batch_size, channels, h, w = out.shape
            
            
            # input shape: (B, out_channels, H*W) ==> output shape: (B, out_channels, H*W)
            in_attn = out.reshape(batch_size, channels, h * w)
            in_attn = self.attention_norms[i](in_attn)
            
            # multihead attention needs shape of (Batch_size, seq_len, embed_dim) ==> we transpose it into (B, H*W, out_channels)
            in_attn = in_attn.transpose(1, 2)
            
            # MultiheadAttention calculation
            # input shape: (B, H*W, out_channels) ==> output shape: (B, H*W, out_channels)
            out_attn, _ = self.attention_layers[i](in_attn, in_attn, in_attn)
            
            # convert it back into shape: (B, out_channels, h, w)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
            
            # resnet connection
            out = out + out_attn  
            
        # input shape: (B, out_channels, H, W)
        # if downsample:
        #   Conv2d(kernel=4, stride=2, padding=1)
        #   output shape: (B, out_channels, H//2, W//2)
        out = self.down_sample_conv(out)
        
        return out