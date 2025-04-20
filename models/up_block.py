import torch
from torch import nn

class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, t_emb_dim: int, up_sample: bool = True, num_heads: int = 4, num_layers: int = 1):
        super().__init__()
        
        self.num_layers = num_layers
        self.up_sample = up_sample
        
        self.resnet_conv1 = nn.ModuleList()
        self.t_emb_layers = nn.ModuleList()
        self.resnet_conv2 = nn.ModuleList()
        self.residual_input_conv = nn.ModuleList()
        self.attention_norms = nn.ModuleList()
        self.attentions = nn.ModuleList()
        
        for i in range(num_layers):
            in_ch = in_channels if i == 0 else out_channels
            
            # first ResNet conv: GroupNorm ==> SiLU ==> 3×3 Conv
            self.resnet_conv1.append(
                nn.Sequential(
                    nn.GroupNorm(8, in_ch),
                    nn.SiLU(),
                    nn.Conv2d(in_ch, out_channels, kernel_size=3, stride=1, padding=1)
                )
            )

            # time‑embedding projection: SiLU ==> Linear
            self.t_emb_layers.append(
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(t_emb_dim, out_channels)
                )
            )

            # second ResNet conv: GroupNorm ==> SiLU ==> 3×3 Conv
            self.resnet_conv2.append(
                nn.Sequential(
                    nn.GroupNorm(8, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
                )
            )

            # 1×1 conv to align channel count for residual path
            self.residual_input_conv.append(
                nn.Conv2d(in_ch, out_channels, kernel_size=1)
            )

            # attention: GroupNorm + Multi‑head self‑attention
            self.attention_norms.append(nn.GroupNorm(8, out_channels))
            self.attentions.append(
                nn.MultiheadAttention(out_channels, num_heads, batch_first=True)
            )        

        if self.up_sample:
            self.up_sample_conv = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, 4, 2, 1)
        else:
            self.up_sample_conv = nn.Identity()

    def forward(self, x, out_down, t_emb):
        x = self.up_sample_conv(x)
        x = torch.cat([x, out_down], dim=1)
        
        out = x
        for i in range(self.num_layers):
            resnet_input = out
            out = self.resnet_conv1[i](out)
            out = out + self.t_emb_layers[i](t_emb)[:, :, None, None]
            out = self.resnet_conv2[i](out)
            out = out + self.residual_input_conv[i](resnet_input)
            
            batch_size, channels, h, w = out.shape
            in_attn = out.reshape(batch_size, channels, h * w)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2)
            out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
            out = out + out_attn

        return out