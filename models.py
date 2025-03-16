import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNormFunction(torch.autograd.Function):

	@staticmethod
	def forward(ctx, x, weight, bias, eps):
		ctx.eps = eps
		N, C, H, W = x.size()
		mu = x.mean(1, keepdim=True)
		var = (x - mu).pow(2).mean(1, keepdim=True)
		y = (x - mu) / (var + eps).sqrt()
		ctx.save_for_backward(y, var, weight)
		y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
		return y

	@staticmethod
	def backward(ctx, grad_output):
		eps = ctx.eps

		N, C, H, W = grad_output.size()
		y, var, weight = ctx.saved_variables
		g = grad_output * weight.view(1, C, 1, 1)
		mean_g = g.mean(dim=1, keepdim=True)

		mean_gy = (g * y).mean(dim=1, keepdim=True)
		gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
		return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
			dim=0), None

class LayerNorm2d(nn.Module):

	def __init__(self, channels, eps=1e-6):
		super(LayerNorm2d, self).__init__()
		self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
		self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
		self.eps = eps

	def forward(self, x):
		return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
	def forward(self, x):
		x1, x2 = x.chunk(2, dim=1)
		return x1 * x2


class depthwise_separable_conv(nn.Module):
	def __init__(self, nin, nout, kernel_size=3, padding=0, stide=1, bias=False):
		super(depthwise_separable_conv, self).__init__()
		self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)
		self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, stride=stide, padding=padding, groups=nin,
								   bias=bias)

	def forward(self, x):
		x = self.depthwise(x)
		x = self.pointwise(x)
		return x


class UpsampleWithFlops(nn.Upsample):
	def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
		super(UpsampleWithFlops, self).__init__(size, scale_factor, mode, align_corners)
		self.__flops__ = 0

	def forward(self, input):
		self.__flops__ += input.numel()
		return super(UpsampleWithFlops, self).forward(input)


class GlobalContextExtractor(nn.Module):
	def __init__(self, c, kernel_sizes=[3, 3, 5], strides=[3, 3, 5], padding=0, bias=False):
		super(GlobalContextExtractor, self).__init__()

		self.depthwise_separable_convs = nn.ModuleList([
			depthwise_separable_conv(c, c, kernel_size, padding, stride, bias)
			for kernel_size, stride in zip(kernel_sizes, strides)
		])

	def forward(self, x):
		outputs = []
		for conv in self.depthwise_separable_convs:
			x = F.gelu(conv(x))
			outputs.append(x)
		return outputs


class CascadedGazeBlock(nn.Module):
	def __init__(self, c, GCE_Conv=3, DW_Expand=2, FFN_Expand=2, drop_out_rate=0):
		super().__init__()
		self.dw_channel = c * DW_Expand
		self.GCE_Conv = GCE_Conv
		self.conv1 = nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=1,
							   padding=0, stride=1, groups=1, bias=True)
		self.conv2 = nn.Conv2d(in_channels=self.dw_channel, out_channels=self.dw_channel,
							   kernel_size=3, padding=1, stride=1, groups=self.dw_channel,
							   bias=True)

		if self.GCE_Conv == 4:
			self.GCE = GlobalContextExtractor(c=c, kernel_sizes=[3, 3, 5], strides=[2, 3, 4])

			self.project_out = nn.Conv2d(int(self.dw_channel * 2.5), c, kernel_size=1)

			self.sca = nn.Sequential(
				nn.AdaptiveAvgPool2d(1),
				nn.Conv2d(in_channels=int(self.dw_channel * 2.5), out_channels=int(self.dw_channel * 2.5),
						  kernel_size=1, padding=0, stride=1,
						  groups=1, bias=True))
		else:
			self.GCE = GlobalContextExtractor(c=c, kernel_sizes=[3, 3], strides=[2, 1])

			self.project_out = nn.Conv2d(self.dw_channel * 2, c, kernel_size=1)

			self.sca = nn.Sequential(
				nn.AdaptiveAvgPool2d(1),
				nn.Conv2d(in_channels=self.dw_channel * 2, out_channels=self.dw_channel * 2, kernel_size=1, padding=0,
						  stride=1,
						  groups=1, bias=True))

		# SimpleGate
		self.sg = SimpleGate()

		ffn_channel = FFN_Expand * c
		self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
							   bias=True)
		self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
							   groups=1, bias=True)

		self.norm1 = LayerNorm2d(c)
		self.norm2 = LayerNorm2d(c)

		self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
		self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

		self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
		self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

	def forward(self, inp):
		x = inp
		b, c, h, w = x.shape
		self.upsample = UpsampleWithFlops(size=(h, w), mode='nearest')

		x = self.norm1(x)
		x = self.conv1(x)
		x = self.conv2(x)
		x = F.gelu(x)

		# Global Context Extractor + Range fusion
		x_1, x_2 = x.chunk(2, dim=1)
		if self.GCE_Conv == 4:
			x1, x2, x3 = self.GCE(x_1 + x_2)
			x = torch.cat([x, self.upsample(x1), self.upsample(x2), self.upsample(x3)], dim=1)
		else:
			x1, x2 = self.GCE(x_1 + x_2)
			x = torch.cat([x, self.upsample(x1), self.upsample(x2)], dim=1)

		x = self.sca(x) * x
		x = self.project_out(x)

		# channel-mixing
		x = self.dropout1(x)
		y = inp + x * self.beta
		x = self.conv4(self.norm2(y))
		x = self.sg(x)
		x = self.conv5(x)

		x = self.dropout2(x)

		return y + x * self.gamma


class NAFBlock0(nn.Module):
	def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.0):
		super().__init__()
		dw_channel = c * DW_Expand
		self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
							   bias=True)
		self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
							   groups=dw_channel,
							   bias=True)
		self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
							   groups=1, bias=True)

		# Simplified Channel Attention
		self.sca = nn.Sequential(
			nn.AdaptiveAvgPool2d(1),
			nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
					  groups=1, bias=True),
		)

		# SimpleGate
		self.sg = SimpleGate()

		ffn_channel = FFN_Expand * c
		self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
							   bias=True)
		self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
							   groups=1, bias=True)

		self.norm1 = LayerNorm2d(c)
		self.norm2 = LayerNorm2d(c)

		self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
		self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

		self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
		self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

	def forward(self, inp):
		x = inp

		x = self.norm1(x)

		x = self.conv1(x)
		x = self.conv2(x)
		x = self.sg(x)
		x = x * self.sca(x)
		x = self.conv3(x)

		x = self.dropout1(x)

		y = inp + x * self.beta

		x = self.conv4(self.norm2(y))
		x = self.sg(x)
		x = self.conv5(x)

		x = self.dropout2(x)

		return y + x * self.gamma


class CascadedGazeNetBigger(nn.Module):

	def __init__(self, img_channel=4, width=32, middle_blk_num=6, enc_blk_nums=[2, 2, 3, 4], dec_blk_nums=[1, 1, 1, 2], GCE_CONVS_nums=[4,4,3,3]):
		super().__init__()

		self.intro = nn.Conv2d(in_channels=3 * (img_channel + 2), out_channels=3 * 16, kernel_size=3, padding=1,
							   stride=1,
							   groups=3,
							   bias=True)
		self.intro2 = nn.Conv2d(in_channels=3 * 16, out_channels=width, kernel_size=1, padding=0, stride=1,
								groups=1,
								bias=True)

		self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
								groups=1,
								bias=True)

		self.encoders = nn.ModuleList()
		self.decoders = nn.ModuleList()
		self.middle_blks = nn.ModuleList()
		self.ups = nn.ModuleList()
		self.downs = nn.ModuleList()

		chan = width
		# for num in enc_blk_nums:
		for i in range(len(enc_blk_nums)):
			num = enc_blk_nums[i]
			GCE_Convs = GCE_CONVS_nums[i]
			self.encoders.append(
				nn.Sequential(
					*[CascadedGazeBlock(chan, GCE_Conv=GCE_Convs) for _ in range(num)]
				)
			)
			self.downs.append(
				nn.Conv2d(chan, 2 * chan, 2, 2)
			)
			chan = chan * 2

		self.middle_blks = \
			nn.Sequential(
				*[NAFBlock0(chan) for _ in range(middle_blk_num)]
			)

		for i in range(len(dec_blk_nums)):
			num = dec_blk_nums[i]
			self.ups.append(
				nn.Sequential(
					nn.Conv2d(chan, chan * 2, 1, bias=False),
					nn.PixelShuffle(2)
				)
			)
			chan = chan // 2
			self.decoders.append(
				nn.Sequential(
					*[NAFBlock0(chan) for _ in range(num)]
				)
			)

		self.padder_size = 2 ** len(self.encoders)

	def forward(self, in0, in1, in2, noise_map, factor=None):

		inp = torch.cat((in0, noise_map, in1, noise_map, in2, noise_map), dim=1)

		x = self.intro2(self.intro(inp))

		encs = []

		for encoder, down in zip(self.encoders, self.downs):
			x = encoder(x)
			encs.append(x)
			x = down(x)

		x = self.middle_blks(x)

		for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
			x = up(x)
			x = x + enc_skip
			x = decoder(x)

		x = self.ending(x)
		if factor is not None:
			x = x + in1 * factor
		else:
			x = x + in1

		return x

class SpaceToDepth(nn.Module):
	def __init__(self, block_size=2):
		super().__init__()
		assert block_size in {2, 4}, "Space2Depth only supports blocks size = 4 or 2"
		self.block_size = block_size

	def forward(self, x):
		N, C, H, W = x.size()
		S = self.block_size
		x = x.view(N, C, H // S, S, W // S, S)  # (N, C, H//bs, bs, W//bs, bs)
		x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
		x = x.view(N, C * S * S, H // S, W // S)  # (N, C*bs^2, H//bs, W//bs)
		return x

class CGNet_D3(nn.Module):

	def __init__(self, num_input_frames=7):
		super(CGNet_D3, self).__init__()
		self.num_input_frames = num_input_frames
		self.S2D = SpaceToDepth(2)
		self.PS = nn.PixelShuffle(2)
		# Define models of each denoising stage
		self.temp1 = CascadedGazeNetBigger()
		self.temp2 = CascadedGazeNetBigger()
		self.temp3 = CascadedGazeNetBigger()
		# Init weights
		self.reset_params()

	@staticmethod
	def weight_init(m, scale=1, bias_fill=0):
		if isinstance(m, nn.Conv2d):
			nn.init.kaiming_normal_(m.weight)
			m.weight.data *= scale
			if m.bias is not None:
				m.bias.data.fill_(bias_fill)
		elif isinstance(m, nn.Linear):
			nn.init.kaiming_normal_(m.weight)
			m.weight.data *= scale
			if m.bias is not None:
				m.bias.data.fill_(bias_fill)

	def reset_params(self):
		for _, m in enumerate(self.modules()):
			self.weight_init(m)

	def forward(self, x, noise_map, fip=None, factor=None):
		'''Args:
			x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		# Unpack inputs
		(x0, x1, x2, x3, x4, x5, x6) = tuple(self.S2D(x[:, m:m+1, :, :]) for m in range(self.num_input_frames))
		noise_map = noise_map[:, :, 0:noise_map.shape[2] // 2, 0:noise_map.shape[3] // 2]

		if fip is not None:
			zeros = torch.zeros_like(x2)
			x20 = self.temp1(x0, zeros, x1, noise_map)
			x21 = self.temp1(x1, zeros, x2, noise_map)
			x22 = self.temp1(x2, zeros, x4, noise_map)
			x23 = self.temp1(x4, zeros, x5, noise_map)
			x24 = self.temp1(x5, zeros, x6, noise_map)

			# Second stage
			x31 = self.temp2(x20, x21, x22, noise_map)
			x32 = self.temp2(x21, x22, x23, noise_map)
			x33 = self.temp2(x22, x23, x24, noise_map)

			x = self.temp3(x31, x32, x33, noise_map)
			return self.PS(x), self.PS(x32), self.PS(x22)

		# First stage
		x20 = self.temp1(x0, x1, x2, noise_map, factor)
		x21 = self.temp1(x1, x2, x3, noise_map, factor)
		x22 = self.temp1(x2, x3, x4, noise_map, factor)
		x23 = self.temp1(x3, x4, x5, noise_map, factor)
		x24 = self.temp1(x4, x5, x6, noise_map, factor)

		# Second stage
		x31 = self.temp2(x20, x21, x22, noise_map)
		x32 = self.temp2(x21, x22, x23, noise_map)
		x33 = self.temp2(x22, x23, x24, noise_map)

		x = self.temp3(x31, x32, x33, noise_map)
		x = self.PS(x)
		return x

	def forward_single_image(self, x, noise_map):
		'''Args:
			x: Tensor, [N, num_frames*C, H, W] in the [0., 1.] range
			noise_map: Tensor [N, 1, H, W] in the [0., 1.] range
		'''
		# Unpack inputs
		noise_map = noise_map[:, :, 0:noise_map.shape[2] // 2, 0:noise_map.shape[3] // 2]
		# First stage
		x = self.S2D(x)
		x = self.temp1(x, x, x, noise_map)
		x = self.temp2(x, x, x, noise_map)
		x = self.temp2(x, x, x, noise_map)

		return self.PS(x)