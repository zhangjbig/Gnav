from copy import deepcopy
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
import torch
import numpy as np
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
import torch.nn.functional as F
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
import math


class Attention(nn.Module):
    "Implementation of self-attention"

    def __init__(self, dim, qkv_bias=False,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = int(dim/2)
        num_heads = int(dim/2)
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, D, H, W= x.shape
        x = x.permute(0, 2, 3, 4, 1)


        qkv = self.qkv(x).reshape(B, D * H * W, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        # 这里是先输入线性层，然后重新塑性
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, D, H, W, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Mlp(nn.Module):
    "Implementation of MLP"

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = x.permute(0,2,3,4,1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.permute(0,4,1,2,3)
        return x


class Transformer(nn.Module):
    """
    Implementation of Transformer,
    Transformer is the second stage in our VOLO
    """
    def __init__(self, dim,output_channels, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm=nn.InstanceNorm3d):
        # dim表示图像特征的通道数
        # mlp_ratio表示MLP的隐藏层维度和输入维度的比例；
        # act_layer表示激活函数
        # norm_layer表示归一化方法
        super().__init__()
        self.output_channels = output_channels
        self.norm1 = norm(dim) # 定义了一个LayerNorm对象self.norm1，对输入进行归一化。
        self.attn = Attention(dim, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, attn_drop=attn_drop)
        # 定义了一个Attention对象self.attn，实现了自注意力机制。
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        # 定义了一个DropPath对象self.drop_path，实现了随机深度正则化

        self.norm2 = norm(dim) #定义了一个LayerNorm对象self.norm2，对输入进行归一化。
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer)

    def forward(self, x):
        # 对输入进行LayerNorm归一化，然后经过Attention和DropPath处理，再通过残差连接到原始输入上；
        m = self.drop_path(self.attn(self.norm1(x)))
        m = m.permute(0,4,1,2,3)
        x = x + m
        # 对上一步的结果再进行LayerNorm归一化，然后经过Mlp和DropPath处理，再通过残差连接到上一步的
        # m = self.drop_path(self.mlp(self.norm2(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class ConvDropoutNormNonlin(nn.Module):
    """
    fixes a bug in ConvDropoutNormNonlin where lrelu was used regardless of nonlin. Bad.
    """

    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(ConvDropoutNormNonlin, self).__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        self.conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
            'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)


    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))


class ConvDropoutNonlinNorm(ConvDropoutNormNonlin):
    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.instnorm(self.lrelu(x))


class StackedConvLayers(nn.Module):
    def __init__(self, input_feature_channels, output_feature_channels, num_convs,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, first_stride=None, basic_block=ConvDropoutNormNonlin):
        '''
        stacks ConvDropoutNormLReLU layers. initial_stride will only be applied to first layer in the stack. The other parameters affect all layers
        :param input_feature_channels:
        :param output_feature_channels:
        :param num_convs:堆叠的模块数
        :param dilation:
        :param kernel_size:
        :param padding:
        :param dropout:
        :param initial_stride:
        :param conv_op:
        :param norm_op:
        :param dropout_op:
        :param inplace:
        :param neg_slope:
        :param norm_affine:
        :param conv_bias:
        '''
        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels

        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        if first_stride is not None:
            self.conv_kwargs_first_conv = deepcopy(conv_kwargs)
            self.conv_kwargs_first_conv['stride'] = first_stride
        else:
            self.conv_kwargs_first_conv = conv_kwargs

        super(StackedConvLayers, self).__init__()
        self.blocks = nn.Sequential(
            *([basic_block(input_feature_channels, output_feature_channels, self.conv_op,
                           self.conv_kwargs_first_conv,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs)] +
              [basic_block(output_feature_channels, output_feature_channels, self.conv_op,
                           self.conv_kwargs,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs) for _ in range(num_convs - 1)]))

    def forward(self, x):
        return self.blocks(x)


def print_module_training_status(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d) or isinstance(module, nn.Dropout3d) or \
            isinstance(module, nn.Dropout2d) or isinstance(module, nn.Dropout) or isinstance(module, nn.InstanceNorm3d) \
            or isinstance(module, nn.InstanceNorm2d) or isinstance(module, nn.InstanceNorm1d) \
            or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d) or isinstance(module,
                                                                                                      nn.BatchNorm1d):
        print(str(module), module.training)


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return F.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                         align_corners=self.align_corners)



class Generic_UNet(SegmentationNetwork):
    DEFAULT_BATCH_SIZE_3D = 2
    DEFAULT_PATCH_SIZE_3D = (64, 192, 160)
    SPACING_FACTOR_BETWEEN_STAGES = 2
    BASE_NUM_FEATURES_3D = 30
    MAX_NUMPOOL_3D = 999
    MAX_NUM_FILTERS_3D = 320 # 修改这里可以增加网络的深度，在魔改nnunet论文里有巨大提升

    DEFAULT_PATCH_SIZE_2D = (256, 256)
    BASE_NUM_FEATURES_2D = 30
    DEFAULT_BATCH_SIZE_2D = 50
    MAX_NUMPOOL_2D = 999
    MAX_FILTERS_2D = 480

    use_this_for_batch_size_computation_2D = 19739648
    use_this_for_batch_size_computation_3D = 520000000  # 505789440

    def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin,
                 seg_output_use_bias=False):
        '''

        :param input_channels: 输入通道数
        :param base_num_features: 第一层卷积的通道数
        :param num_classes: 输出通道数？
        :param num_pool: 网络中的池化层数
        :param num_conv_per_stage: 每个阶段的卷积层数
        :param feat_map_mul_on_downscale: 下采样时特征图的通道数乘数
        :param conv_op:卷积操作函数
        :param norm_op:归一化操作函数
        :param norm_op_kwargs:归一化操作函数的参数
        :param dropout_op:dropout 操作函数，如 nn.Dropout2d；
        :param dropout_op_kwargs:dropout 操作函数的参数；
        :param nonlin:非线性激活函数，如 nn.LeakyReLU；
        :param nonlin_kwargs:非线性激活函数的参数；
        :param deep_supervision:是否使用深度监督；
        :param dropout_in_localization:是否在定位部分使用 dropout；
        :param final_nonlin:输出层的激活函数；
        :param weightInitializer:权重初始化函数；
        :param pool_op_kernel_sizes:池化操作函数的核大小；是一个列表，每个元素是一个元组。
        :param conv_kernel_sizes:卷积操作函数的核大小；
        :param upscale_logits:是否在进行上采样时放大输出的 logits；？
        :param convolutional_pooling:是否使用卷积池化层；
        :param convolutional_upsampling:是否使用卷积上 采样层；
        :param max_num_features:最大特征数
        :param basic_block:基本块类型，如 ConvDropoutNormNonlin；
        :param seg_output_use_bias:输出层是否使用 bias。
        '''
        """
        basically more flexible than v1, architecture is the same

        Does this look complicated? Nah bro. Functionality > usability

        This does everything you need, including world peace.

        Questions? -> f.isensee@dkfz.de
        """
        super(Generic_UNet, self).__init__() #初始化该类
        self.convolutional_upsampling = convolutional_upsampling
        self.convolutional_pooling = convolutional_pooling
        self.upscale_logits = upscale_logits
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

        self.nonlin = nonlin # 默认是relu函数
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer #默认为xavier_uniform
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.num_classes = num_classes
        self.final_nonlin = final_nonlin
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        '''
        [(2, 2)] * num_pool表示将元组(2, 2)重复num_pool次，生成一个包含num_pool个元素的列表，每个元素都是(2, 2)。
        例如，当num_pool为3时，[(2, 2)] * num_pool的结果为[(2, 2), (2, 2), (2, 2)]。
        '''
        if conv_op == nn.Conv2d:
            upsample_mode = 'bilinear'
            pool_op = nn.MaxPool2d
            transpconv = nn.ConvTranspose2d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2)] * num_pool

            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3)] * (num_pool + 1)
        elif conv_op == nn.Conv3d:
            upsample_mode = 'trilinear'
            pool_op = nn.MaxPool3d
            transpconv = nn.ConvTranspose3d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3, 3)] * (num_pool + 1)
        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(conv_op))

        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64) #它是 pool_op_kernel_sizes 列表中所有元素的乘积，用来检查输入张量的每个维度是否可以被kernel size整除
        self.pool_op_kernel_sizes = pool_op_kernel_sizes # self.pool_op_kernel_sizes 是一个包含 (w,h) 或 (w,h,d) 元组的列表，表示每个池化层在空间维度上的大小。如果 conv_op 是 3D 卷积，元组中的第三个元素 d 表示深度。
        self.conv_kernel_sizes = conv_kernel_sizes # 是一个包含 (w,h) 或 (w,h,d) 元组的列表，表示每个卷积层在空间维度上的大小。如果 conv_op 是 3D 卷积，元组中的第三个元素 d 表示深度。

        self.conv_pad_sizes = []
        '''
        对于2D卷积操作，self.conv_pad_sizes 中的值为 [0, 1, 1, 0]。这是因为二维卷积核的形状为 (height=3, width=3)，需要在高和宽的两侧各填充1个单元格。

对于3D卷积操作，self.conv_pad_sizes 中的值为 [0, 1, 1, 1, 0]。这是因为三维卷积核的形状为 (depth=3, height=3, width=3)，需要在深度、高和宽的两侧各填充1个单元格。
        '''
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

        if max_num_features is None:
            if self.conv_op == nn.Conv3d:
                self.max_num_features = self.MAX_NUM_FILTERS_3D
            else:
                self.max_num_features = self.MAX_FILTERS_2D
        else:
            self.max_num_features = max_num_features
            '''
            self.max_num_features 是一个整数，表示网络中任何时候的最大特征映射数。如果未指定 max_num_features，
            则分别为 2D 和 3D 卷积使用默认值 MAX_FILTERS_2D 或 MAX_NUM_FILTERS_3D。
            如果指定了 max_num_features，则使用该值。
            '''

        self.conv_blocks_context = [] # 对应unet的编码器，用于储存nn.Module对象，每个卷积快由一系列的卷积层组成
        self.conv_blocks_localization = [] # 对应unet的解码器，同上
        self.td = [] # 编码器部分的下采样组件。通过最大池化操作实现下采样组件
        self.tu = [] # 通过转置卷积操作实现上采样组件
        self.seg_outputs = [] # 用于储存模型的分割分支

        output_features = base_num_features # 第一层卷积层的输出特征数
        input_features = input_channels # 输入图像的通道，代码中所有的特征数都是等于通道数的


        for d in range(num_pool): # 循环 num_pool 次，即下采样的层数。但是注意range(num_pool)生成的是[0,...,num_pool]的序列，所以第一个卷积块是两个一样的卷积层，剩下num_pool-1个
            # 是包含了下采样卷积层的卷积块
            # determine the first stride
            if d != 0 and self.convolutional_pooling: # 当不是第一次做池化操作并且convolutional_pooling是True
                first_stride = pool_op_kernel_sizes[d - 1] # first_stride 将等于上一次池化层的大小
            else:
                first_stride = None

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]

            # 每次卷积之前添加一个transformer层
            #class Transformer(nn.Module):
            #Implementation of Transformer,
            #Transformer is the second stage in our VOLO
            #   def __init__(self, dim, num_heads = 8, mlp_ratio=4., qkv_bias=False,
            #     qk_scale=None, attn_drop=0., drop_path=0.,
            #     act_layer=nn.GELU, norm_layer=nn.LayerNorm):

            # add convolutions
            self.conv_blocks_context.append(StackedConvLayers(input_features, output_features, num_conv_per_stage,
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              first_stride, basic_block=basic_block))
            # 向编码器中添加卷积层。‘StackedConvLayers’接受一些参数，然后将其转换成一些列的卷积层堆叠在一起
            # num_conv_per_stage是每个卷积快中的卷积层数量，默认等于2
            # first_stride 是第一个卷积层的步长，是（2,2,2）（相当合理）
            # basic_block 是基本卷积块类型，等于ConvDropoutNormNonlin，所以其实我只需要把这个部分添加一个transformer就可以了吧？

            if not self.convolutional_pooling:
                self.td.append(pool_op(pool_op_kernel_sizes[d]))
            # 如果 self.convolutional_pooling 为 False，则将池化层加入 self.td 列表中，具体参数为 pool_op_kernel_sizes[d]，即当前下采样层的池化核大小。
            # 池化的方法下采样，一般用不到哦

            # 更新下一层的输出特征数
            input_features = output_features # 将这一层的输出特征数赋给下一层的输入特征数
            output_features = int(np.round(output_features * feat_map_mul_on_downscale)) # ‘feat_map_mul_on_downscale’每次下采样时通道数翻倍的倍数，这里默认的数量是2，表示下一次输出的特征数是这次的两倍
            output_features = min(output_features, self.max_num_features) # 输出通道数最大不能超过max_num_features
            print(output_features)

        #下面生成最后一个下采样卷积块
        # now the bottleneck.
        # determine the first stride
        if self.convolutional_pooling:
            first_stride = pool_op_kernel_sizes[-1] # 步长等于pool_op_kernel_sizes最后一个元组（2，2，2）也是（w,h,d）
        else:
            first_stride = None

        # the output of the last conv must match the number of features from the skip connection if we are not using
        # convolutional upsampling. If we use convolutional upsampling then the reduction in feature maps will be
        # done by the transposed conv
        # 这段不是很清楚意义是什么
        if self.convolutional_upsampling:
            final_num_features = output_features # 最后一个下采样之后的特征值
        else: # 采用双线性插值来进行上采样
            final_num_features = self.conv_blocks_context[-1].output_channels

        # 下采样进行完后
        self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[num_pool] # 循环到最后的卷积核大小
        self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool] # conv_pad_sizes里第num_pool个

        # self.conv_blocks_context.append(Transformer(input_features, output_features))
        self.conv_blocks_context.append(nn.Sequential(
            StackedConvLayers(input_features, output_features, num_conv_per_stage - 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, first_stride, basic_block=basic_block), # 这里num_conv_per_stage - 1，所以只堆了一个卷积层，是下采样卷积层
            Transformer(input_features, output_features),
            Transformer(input_features, output_features),
            Transformer(input_features, output_features),
            StackedConvLayers(output_features, final_num_features, 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, basic_block=basic_block))) # 这里堆了一个正常卷积层


#######################################################下面是解码器部分#################################################################

        # if we don't want to do dropout in the localization pathway then we set the dropout prob to zero here
        if not dropout_in_localization:
            old_dropout_p = self.dropout_op_kwargs['p']
            self.dropout_op_kwargs['p'] = 0.0

        # now lets build the localization pathway(上采样路径)
        for u in range(num_pool):
            nfeatures_from_down = final_num_features # 下采样路径特征
            nfeatures_from_skip = self.conv_blocks_context[
                -(2 + u)].output_channels  # self.conv_blocks_context[-1] is bottleneck, so start with -2
            n_features_after_tu_and_concat = nfeatures_from_skip * 2

            # the first conv reduces the number of features to match those of skip
            # the following convs work on that number of features
            # if not convolutional upsampling then the final conv reduces the num of features again
            if u != num_pool - 1 and not self.convolutional_upsampling:
                final_num_features = self.conv_blocks_context[-(3 + u)].output_channels
            else:
                final_num_features = nfeatures_from_skip

            # 如果当前层数不是最后一层，且不使用卷积上采样，则从上一层的输出通道数中获取最终特征映射的通道数；否则，将最终通道数设置为来自skip-connection的特征映射的通道数。


            if not self.convolutional_upsampling:
                self.tu.append(Upsample(scale_factor=pool_op_kernel_sizes[-(u + 1)], mode=upsample_mode))
            else:
                self.tu.append(transpconv(nfeatures_from_down, nfeatures_from_skip, pool_op_kernel_sizes[-(u + 1)],
                                          pool_op_kernel_sizes[-(u + 1)], bias=False))

            # 如果不使用卷积上采样，则将一个上采样层添加到UNet中；否则，将一个转置卷积层添加到UNet中。

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[- (u + 1)]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[- (u + 1)]
            self.conv_blocks_localization.append(nn.Sequential(
                StackedConvLayers(n_features_after_tu_and_concat, nfeatures_from_skip, num_conv_per_stage - 1,
                                  self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                                  self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block),
                StackedConvLayers(nfeatures_from_skip, final_num_features, 1, self.conv_op, self.conv_kwargs,
                                  self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                  self.nonlin, self.nonlin_kwargs, basic_block=basic_block)
            ))

        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(conv_op(self.conv_blocks_localization[ds][-1].output_channels, num_classes,
                                            1, 1, 0, 1, 1, seg_output_use_bias))
        # 对于每个 downsampling 操作，将最后一个卷积层的输出通道数（output_channels）作为输入，
        # 使用conv_op函数（通常是一个卷积操作）进行1x1卷积（1个kernel，stride=1，padding=0）得到一个num_classes通道的输出，这个输出会被添加到seg_outputs列表中。

        self.upscale_logits_ops = []
        cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]

        #upscale_logits为True，则将Upsample函数添加到列表中，该函数用于通过插值（interpolation）进行上采样，
        #每个元素都是一个scale_factor为cum_upsample[usl + 1]的Upsample函数。
        #cum_upsample是一个数组，其中包含每个上池化步骤的比例因子的乘积。
        #如果upscale_logits为False，则将lambda函数添加到列表中，这个函数只是返回其输入x本身。

        for usl in range(num_pool - 1):
            if self.upscale_logits:
                self.upscale_logits_ops.append(Upsample(scale_factor=tuple([int(i) for i in cum_upsample[usl + 1]]),
                                                        mode=upsample_mode))
            else:
                self.upscale_logits_ops.append(lambda x: x)

        if not dropout_in_localization:
            self.dropout_op_kwargs['p'] = old_dropout_p

        # register all modules properly
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.td = nn.ModuleList(self.td)
        self.tu = nn.ModuleList(self.tu)
        self.seg_outputs = nn.ModuleList(self.seg_outputs)

        #在这里，将所有的可训练模块（即conv_blocks_localization、conv_blocks_context、td、tu、
        #seg_outputs和upscale_logits_ops）都转换成一个PyTorch的nn.ModuleList对象，
        #这样才能在训练和测试的时候进行迭代和使用。

        if self.upscale_logits:
            self.upscale_logits_ops = nn.ModuleList(
                self.upscale_logits_ops)  # lambda x:x is not a Module so we need to distinguish here

        #如果self.upscale_logits为True，则upscale_logits_ops中的每个元素都转换成
        #nn.ModuleList，否则就不需要转换。

        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)
            # self.apply(print_module_training_status)

        #如果给定了weightInitializer，则将它应用到整个模型中


    def forward(self, x):
        skips = []
        seg_outputs = []
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[d](x)

        x = self.conv_blocks_context[-1](x)

        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]

    @staticmethod
    def compute_approx_vram_consumption(patch_size, num_pool_per_axis, base_num_features, max_num_features,
                                        num_modalities, num_classes, pool_op_kernel_sizes, deep_supervision=False,
                                        conv_per_stage=2):
        """
        This only applies for num_conv_per_stage and convolutional_upsampling=True
        not real vram consumption. just a constant term to which the vram consumption will be approx proportional
        (+ offset for parameter storage)
        :param deep_supervision:
        :param patch_size:
        :param num_pool_per_axis:
        :param base_num_features:
        :param max_num_features:
        :param num_modalities:
        :param num_classes:
        :param pool_op_kernel_sizes:
        :return:
        """
        if not isinstance(num_pool_per_axis, np.ndarray):
            num_pool_per_axis = np.array(num_pool_per_axis)

        npool = len(pool_op_kernel_sizes)

        map_size = np.array(patch_size)
        tmp = np.int64((conv_per_stage * 2 + 1) * np.prod(map_size, dtype=np.int64) * base_num_features +
                       num_modalities * np.prod(map_size, dtype=np.int64) +
                       num_classes * np.prod(map_size, dtype=np.int64))

        num_feat = base_num_features

        for p in range(npool):
            for pi in range(len(num_pool_per_axis)):
                map_size[pi] /= pool_op_kernel_sizes[p][pi]
            num_feat = min(num_feat * 2, max_num_features)
            num_blocks = (conv_per_stage * 2 + 1) if p < (npool - 1) else conv_per_stage  # conv_per_stage + conv_per_stage for the convs of encode/decode and 1 for transposed conv
            tmp += num_blocks * np.prod(map_size, dtype=np.int64) * num_feat
            if deep_supervision and p < (npool - 2):
                tmp += np.prod(map_size, dtype=np.int64) * num_classes
            # print(p, map_size, num_feat, tmp)
        return tmp

