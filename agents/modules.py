
import torch.nn as nn
import torch
import math
        
class Parallel(nn.Module):
    def __init__(self, modules: dict[str, nn.Module]):
        super().__init__()
        if not isinstance(modules, dict):
            raise ValueError("Modules are not dicts!")
        
        self.model = nn.ModuleDict(modules) # Otherwise modeules are not registered correctly

    def forward(self, inputs: dict[str, torch.Tensor]):
        return [module(inputs[name]) for name, module in self.model.items()]
    

class Downsample(nn.Module):
    def __init__(self, space: int, dim: int, downsample: int = 2, phi: nn.Module = nn.SiLU()):
        super().__init__()
        self.phi = phi

        if space == 2:
            Conv = nn.Conv2d
        else:
            Conv = nn.Conv3d
            downsample = (1, downsample, downsample)

        self.conv = Conv(dim, dim, downsample, downsample, bias=False)

    def forward(self, x: torch.Tensor):
        return self.phi(self.conv(x))

class MultiHeadAttention(nn.Module):
    
    def __init__(self, dim:int, num_heads: int=8, dropout:float=0.1, phi: nn.Module = nn.SiLU()):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        
        self.phi = phi
        
        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(dim, dim)
        
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Linear transformations
        q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        q = q.transpose(1, 2)  # (batch, heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = self.softmax(scores)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        
        return self.out(attn_output)

class ResidualBlock(nn.Module):
    def __init__(self, space: int, dim: int, act_fn: nn.Module = nn.SiLU(), depth: int = 2, kernel_size: int = 3, padding: int = 1, stride: int = 1):
        super().__init__()
        self.phi = act_fn

        Conv = nn.Conv2d if space == 2 else nn.Conv3d
        convs = []
        for d in range(depth):
            conv = nn.Sequential(
                Conv(dim, dim, kernel_size=kernel_size, padding=padding, stride=stride),
                self.phi,
                Conv(dim, dim, kernel_size=kernel_size, padding=padding, stride=stride),
            )
            convs.append(conv)
            if d < depth - 1:
                convs.append(self.phi)
        self.convs = nn.ModuleList(convs)

    def forward(self, x):
        residual = x
        for conv in self.convs:
            x = conv(x) + residual
            residual = x
        return x
    
    
class OwnModule(nn.Module):
    def __init__(self):
        super().__init__()
        
    def init_weights(self, debug: bool=False):  
        """ Initialize weights for Linear features. Kaiming He for (Leaky-)Relu otherwise Xavier """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if isinstance(self.phi, nn.ReLU):
                    nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                    print("Weights initialized with Kaiming He") if debug else None
                elif isinstance(self.phi, nn.LeakyReLU):
                    nn.init.kaiming_uniform_(module.weight, nonlinearity="leaky_relu")
                    print("Weights initialized with Kaiming He") if debug else None
                else:
                    nn.init.xavier_uniform_(module.weight) 
                    print("Weights initialized with Glorot") if debug else None
                    
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def get_padding(self, kernel_size: int) -> int:
        """ Preserve the image size if stride is one, otherwise reduce spacial dimension """
        return (kernel_size - 1) // 2
        
    def get_size_estimate(self) -> float:
        param_size = 0
        buffer_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2 # Conversion from bytes to megabytes
    
        return size_all_mb
    

class PreResBlock(OwnModule):
    """ Residual block using skip-connections on pre-activation level. """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, phi: nn.Module = nn.ReLU(), apply_squeeze:bool = False):
        """
        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : int
            Size of the kernel in all dimensions.
        stride : int
            Factor by which to reduce the spatial dimensions.
        phi : nn.Module
            The activation function to use in the residual branch.
        apply_squeeze: bool
            Whether to apply SqueezeExcitementBlock after the feature features
        """
        super().__init__()
        # YOUR CODE HERE
        
        self._stride = stride
        self._kernel_size = kernel_size
        
        self.downsample_fn = None
        if in_channels != out_channels or stride != 1:
            self.downsample_fn = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        
        # Sequential layer block so PyTorch automatically keeps track of the comp. graph (no backprop implementation needed)
        self.features = nn.Sequential(
            nn.BatchNorm2d(in_channels), 
            phi,
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=self.get_padding(kernel_size), bias=False), 
            nn.BatchNorm2d(out_channels),
            phi,
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=self.get_padding(kernel_size), bias=False) 
            )
        
        if apply_squeeze:
            self.se_block = SqueezeExcitationBlock(out_channels, phi, r=4, bias=True, init_weights=False)
        else:
            self.se_block = None
    
    
    def forward(self, x):
        """ Make a forward pass and add the (scaled) identity. """
        # Transformation for identity in case stride != 1
        identity = x if self.downsample_fn is None else self.downsample_fn(x)
        
        # Iterate over all features in the residual block and make a forward pass
        out = self.features(x)
        
        if self.se_block:
            out = self.se_block(out)

        # Return the sum of the identity and the output from the residual block
        return  identity + out
    
    
class DepthwiseConvBlock(OwnModule):
    """ Depthwise convolution with a separate kernel for every channel. """

    def __init__(self, in_channels: int, phi: nn.Module = nn.ReLU(inplace=True), kernel_size: int=2, bias: bool=True):
        super().__init__()
        
        self.bias = bias
        self.phi = phi
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=in_channels, kernel_size=kernel_size, groups=in_channels, bias=self.bias),
            nn.BatchNorm2d(in_channels),
            self.phi,
        )
    
    def forward(self, x):
        return self.features(x)
        
    
class PointwiseConvBlock(OwnModule):
    """ Pointwise convolution with 1x1 kernel size applied to extract features from the channels. """
    def __init__(self, in_channels: int, out_channels: int, phi: nn.Module = nn.ReLU(inplace=True), bias: bool=True):
        super().__init__()

        self.bias = bias
        self.phi = phi
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, out_channels=out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),  
            self.phi,
        )
    
    def forward(self, x):
        return self.features(x)
    
    
class SqueezeExcitationBlock(OwnModule):
    """ 
        Squeeze and Excitation Block that performs dynamic channel-wise feature recalibration. 
        https://arxiv.org/abs/1709.01507
    
    """
    def __init__(self, c: int, phi: nn.Module = nn.ReLU(inplace=True), r: int = 4, bias: bool=True, init_weights: bool=True):
        super().__init__()
        
        self.bias = bias
        self.phi = phi
        
        # Squeeze
        self.squeeze = nn.AdaptiveAvgPool2d(output_size=1)
        
        # Excitation
        self.fc1 = nn.Linear(c, c//r, bias=self.bias)
        self.fc2 = nn.Linear(c//r, c, bias=self.bias)
        self.sigmoid = nn.Sigmoid()
        
        if init_weights:
            self.init_weights()
        
    def forward(self, x):
        # Inspiration from: https://medium.com/@tahasamavati/squeeze-and-excitation-explained-387b5981f249
        
        bs, c, _, _ = x.size()
        
        # Squeeze
        y = self.squeeze(x).reshape(bs, c)
        
        # Excitation
        y = self.fc1(y)
        y = self.phi(y)
        y = self.fc2(y)
        y = self.sigmoid(y).reshape(bs, c, 1, 1)
        
        return x * y