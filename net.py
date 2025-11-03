import torch
import torch.nn as nn
import torch.nn.functional as F


class PartialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, activation=nn.ReLU):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, bias=False)
        self.activation = activation()
        
        nn.init.kaiming_normal_(self.input_conv.weight, a=0.0, mode='fan_in', nonlinearity='relu') 
        nn.init.zeros_(self.input_conv.bias)
        
        torch.nn.init.constant_(self.mask_conv.weight, 1.0)
        for param in self.mask_conv.parameters():
            param.requires_grad = False            


    def forward(self, input, mask):
        output = self.input_conv(input * mask)

        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask <= 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)

        output = self.activation(output)

        return output, new_mask


class PConvUNet(nn.Module):
    def __init__(self, layer_size=7, input_channels=3, upsampling_mode='nearest'):
        super().__init__()
        self.upsampling_mode = upsampling_mode
        self.layer_size = layer_size
        
        # --- ENCODER ---
        self.encoders = nn.ModuleList([
            PartialConv(input_channels, 64, 7, 2, 3),
            PartialConv(64, 128, 5, 2, 2),
            PartialConv(128, 256, 5, 2, 2),
            PartialConv(256, 512, 3, 2, 1)
        ])

        for _ in range(4, layer_size):
            self.encoders.append(PartialConv(512, 512, 3, 2, 1))

        # --- DECODER ---
        self.decoders = nn.ModuleList([])
        for _ in range(4, layer_size):
            self.decoders.append(PartialConv(512 + 512, 512, 3, 1, 1))
        self.decoders.extend([
            PartialConv(512 + 256, 256, 3, 1, 1),
            PartialConv(256 + 128, 128, 3, 1, 1),
            PartialConv(128 + 64, 64, 3, 1, 1),
            PartialConv(64 + input_channels, input_channels, 3, 1, 1, activation=nn.Identity)
        ])
        

    def forward(self, x, m):
        # ---- ENCODER ----
        h_list, m_list = [x], [m]
        h, mask = x, m
        for enc in self.encoders:
            h, mask = enc(h, mask)
            h_list.append(h)
            m_list.append(mask)

        # ---- DECODER ----
        for i, dec in enumerate(self.decoders):
            h = F.interpolate(h,    scale_factor=2, mode=self.upsampling_mode)
            mask = F.interpolate(mask, scale_factor=2, mode='nearest')

            skip_h = h_list[-(i + 2)]
            skip_m = m_list[-(i + 2)]

            h = torch.cat([h, skip_h], dim=1)
            mask = torch.cat([mask, skip_m], dim=1)

            h, mask = dec(h, mask)
            
        h = torch.sigmoid(h)
        return h, mask
