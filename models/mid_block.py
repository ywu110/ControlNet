import torch
from torch import nn

class MidBlock(nn.Module):
    """
    Data Flow:
    Initial Input x
    │
    ├─ Initial ResNet Block (index 0):
    │   ├─ resnet_conv1[0] (Conv path)
    │   ├─ + t_emb_layers[0](t_emb) (Time embedding)
    │   ├─ resnet_conv2[0] (Conv path)
    │   └─ + residual_input_conv[0] (Skip connection)
    │
    └─ Loop num_layers times:
        │
        ├─ Attention Block (index i):
        │   ├─ reshape ==> attention_norms[i] ==> transpose
        │   ├─ MultiheadAttention
        │   └─ transpose + reshape ==> residual add
        │
        └─ ResNet Block (index i+1):
            ├─ resnet_conv1[i+1]
            ├─ + t_emb_layers[i+1](t_emb)
            ├─ resnet_conv2[i+1]
            └─ + residual_input_conv[i+1]
    """
    def __init__(self, in_channels:int, out_channels:int, t_emb_dim:int, num_heads:int=4, num_layers:int =1):
        super().__init__()
        
        self.num_layers = num_layers
        
        self.resnet_conv1   = nn.ModuleList()
        self.t_emb_layers        = nn.ModuleList()
        self.resnet_conv2  = nn.ModuleList()
        self.residual_input_conv = nn.ModuleList()

        self.attention_norms = nn.ModuleList()
        self.attentions      = nn.ModuleList()

        #######################################################
        # 1. ResNet Block  2. Attention Block 3. ResNet Block
        #                  |---- iterative ResNet Block ----|
        #######################################################
        total_resnets = num_layers + 1

        for i in range(total_resnets):
            in_ch = in_channels if i == 0 else out_channels

            # ---- ResNet: resnet_conv1 ----
            self.resnet_conv1.append(
                nn.Sequential(
                    nn.GroupNorm(8, in_ch),
                    nn.SiLU(),
                    nn.Conv2d(in_ch, out_channels, kernel_size=3, stride=1, padding=1)
                )
            )

            # ---- time embeddding (Linear) ----
            self.t_emb_layers.append(
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(t_emb_dim, out_channels)
                )
            )

            # ---- ResNet: resnet_conv2 ----
            self.resnet_conv2.append(
                nn.Sequential(
                    nn.GroupNorm(8, out_channels),
                    nn.SiLU(),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
                )
            )

            # ---- residual projection ==> make sure the number of channels matches ----
            self.residual_input_conv.append(
                nn.Conv2d(in_ch, out_channels, kernel_size=1)
            )

            # ---- Attention Block ----
            # NOTE: The attention block is one layer less than the resnet block. See the diagram above.
            if i < num_layers:
                self.attention_norms.append(nn.GroupNorm(8, out_channels))
                self.attentions.append(nn.MultiheadAttention(out_channels, num_heads, batch_first=True))
    
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        out = x
        
        # First resnet block
        resnet_input = out
        out = self.resnet_conv1[0](out)
        out = out + self.t_emb_layers[0](t_emb)[:, :, None, None]
        out = self.resnet_conv2[0](out)
        out = out + self.residual_input_conv[0](resnet_input)
        
        for i in range(self.num_layers):
            
            # Attention Block
            batch_size, channels, h, w = out.shape
            in_attn = out.reshape(batch_size, channels, h * w)
            in_attn = self.attention_norms[i](in_attn)
            in_attn = in_attn.transpose(1, 2)
            out_attn, _ = self.attentions[i](in_attn, in_attn, in_attn)
            out_attn = out_attn.transpose(1, 2).reshape(batch_size, channels, h, w)
            out = out + out_attn
            
            # Resnet Block
            resnet_input = out
            out = self.resnet_conv1[i+1](out)
            out = out + self.t_emb_layers[i+1](t_emb)[:, :, None, None]
            out = self.resnet_conv2[i+1](out)
            out = out + self.residual_input_conv[i+1](resnet_input)
        
        return out
