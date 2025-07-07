class MultiAspectGCAttention(nn.Module):
    """
    Simplified re-implementation of the GLASS Global-to-Local Attention fusion module. Focuses on core logic: channel interleaving, spatial GC attention, and configurable fusion.
    """
    def __init__(self, in_channels: int, ratio: float = 0.25, headers: int = 8, pooling_type: str = 'att', out_channels: int = None, fusion_type: str = 'channel_add'):
        super().__init__()
        
        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.headers = headers
        self.pooling_type = pooling_type
        self.fusion_type = fusion_type
        
        # Intermediate channels for MLPs in fusion paths
        self.planes = int(in_channels * ratio) 
        self.single_header_in_channels = in_channels // headers

        # "channel-wise interleaved" aspect of GLASS. This pre-computes an index mapping to interleave the first half of input channels with the second half.
        self.order = torch.zeros(in_channels, dtype=torch.long)
        self.order[0::2] = torch.arange(in_channels // 2, dtype=torch.long)
        self.order[1::2] = torch.arange(in_channels // 2, in_channels, dtype=torch.long)

        # Spatial Pooling for Global Context Generation 
        if self.pooling_type == 'att':
            # This 1x1 Conv acts like a query projection to derive spatial attention scores
            self.conv_mask = nn.Conv2d(self.single_header_in_channels, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2) # Softmax over flattened spatial dimensions (H*W)
        else: # pooling_type == 'avg'
            self.avg_pool = nn.AdaptiveAvgPool2d(1) # Global Average Pooling

        # Fusion Paths (Channel Modulation)
        # These are small MLPs that process the global context vector and prepare it for addition, multiplication, or concatenation with the original features.
        if self.fusion_type == 'channel_add':
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.in_channels, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]), # LayerNorm for (C, 1, 1) tensor
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.in_channels, kernel_size=1)
            )
        elif self.fusion_type == 'channel_concat':
            self.channel_concat_conv = nn.Sequential(
                nn.Conv2d(self.in_channels, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.in_channels, kernel_size=1)
            )
            # This conv is used after concatenation to reduce channels back
            self.cat_conv = nn.Conv2d(2 * self.in_channels, self.in_channels, kernel_size=1)
        elif self.fusion_type == 'channel_mul':
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.in_channels, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.in_channels, kernel_size=1)
            )
        
        # A 3x3 convolution to project the fused feature map to the desired output channels.
        self.out = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, padding=1)

    def spatial_pool(self, x: torch.Tensor) -> torch.Tensor:
        """
        Derives a global context vector using either attention-based spatial pooling or global average pooling.
        """
        batch, channel, height, width = x.size()

        if self.pooling_type == 'att':
            # Reshape for multi-head processing: [N * headers, C_per_head, H, W]
            x_per_head = x.view(batch * self.headers, self.single_header_in_channels, height, width)
            input_x_flattened = x_per_head.view(
                batch * self.headers, self.single_header_in_channels, height * width
            )

            # Compute spatial attention mask: [N * headers, 1, H, W] -> [N * headers, 1, H * W]
            context_mask = self.conv_mask(x_per_head)
            context_mask = self.softmax(context_mask.view(batch * self.headers, 1, height * width))

            # Weighted sum over spatial dimensions: [N*headers, C_per_head, 1]
            # (B*H, C_ph, S) @ (B*H, S, 1) -> (B*H, C_ph, 1)
            context = torch.matmul(input_x_flattened, context_mask.transpose(-2, -1))
            
            # Reshape back to original batch and channel structure: [N, C, 1, 1]
            context = context.view(batch, self.headers * self.single_header_in_channels, 1, 1)
        else: # pooling_type == 'avg'
            context = self.avg_pool(x)
        return context

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: x (torch.Tensor): Input feature map, expected as (B, C_global + C_local, H, W).
                               Assumes first C_global channels are global, rest are local.
        Returns: torch.Tensor: Fused feature map (B, out_channels, H, W).
        """
        # Explicitly reorder channels to interleave global and local features.
        x_interleaved = x[:, self.order, ...]
        
        # Store original for potential residual connections in some fusion types
        out = x_interleaved 

        # 2. Global Context (GC) Attention / Spatial Pooling 
        context = self.spatial_pool(x_interleaved)

        # 3. Fusion
        if self.fusion_type == 'channel_mul':
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        elif self.fusion_type == 'channel_add':
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term
        elif self.fusion_type == 'channel_concat':
            channel_concat_term = self.channel_concat_conv(context)
            
            # Expand context to match spatial dimensions of out for concatenation
            _, _, H, W = out.shape
            channel_concat_term_expanded = channel_concat_term.expand(-1, -1, H, W)
            
            out = torch.cat([out, channel_concat_term_expanded], dim=1)
            out = self.cat_conv(out)
            
            # Layer norm and ReLU applied specifically for concat fusion
            out = F.layer_norm(out, [self.in_channels, H, W]) 
            out = F.relu(out)
        
        # 4. Final Output Convolution
        out = self.out(out)
        return out
