import torch
import timm
import torch.nn as nn
from torch import Tensor
from einops import rearrange
from OIQAND_config import config


def creat_model(config, pretrained=False):
    model_oiqa = OIQAND(embed_dim=config.embed_dim)
    if pretrained:
        model_oiqa.load_state_dict(torch.load(config.model_weight_path), strict=False)
        print("weights had been load!\n")
    return model_oiqa


def global_avg_pool2d(x):
    mean = nn.functional.adaptive_avg_pool2d(x, 1)
    return mean


class MultiHeadAttention(nn.Module):
    """
    This layer applies a multi-head self- or cross-attention as described in
    `Attention is all you need <https://arxiv.org/abs/1706.03762>`_ paper

    Args:
        embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(N, P, C_{in})`
        num_heads (int): Number of heads in multi-head attention
        attn_dropout (float): Attention dropout. Default: 0.0
        bias (bool): Use bias or not. Default: ``True``

    Shape:
        - Input: :math:`(N, P, C_{in})` where :math:`N` is batch size, :math:`P` is number of patches,
        and :math:`C_{in}` is input embedding dim
        - Output: same shape as the input

    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        bias: bool = True,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                "Embedding dim must be divisible by number of heads in {}. Got: embed_dim={} and num_heads={}".format(
                    self.__class__.__name__, embed_dim, num_heads
                )
            )

        self.qkv_proj = nn.Linear(in_features=embed_dim, out_features=3 * embed_dim, bias=bias)

        self.attn_dropout = nn.Dropout(p=attn_dropout)
        self.out_proj = nn.Linear(in_features=embed_dim, out_features=embed_dim, bias=bias)

        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.num_heads = num_heads
        self.embed_dim = embed_dim

    def forward(self, x_q: Tensor) -> Tensor:
        # [N, P, C]
        b_sz, n_patches, in_channels = x_q.shape

        # self-attention
        # [N, P, C] -> [N, P, 3C] -> [N, P, 3, h, c] where C = hc
        qkv = self.qkv_proj(x_q).reshape(b_sz, n_patches, 3, self.num_heads, -1)

        # [N, P, 3, h, c] -> [N, h, 3, P, C]
        qkv = qkv.transpose(1, 3).contiguous()

        # [N, h, 3, P, C] -> [N, h, P, C] x 3
        query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        query = query * self.scaling

        # [N h, P, c] -> [N, h, c, P]
        key = key.transpose(-1, -2)

        # QK^T
        # [N, h, P, c] x [N, h, c, P] -> [N, h, P, P]
        attn = torch.matmul(query, key)
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        # weighted sum
        # [N, h, P, P] x [N, h, P, c] -> [N, h, P, c]
        out = torch.matmul(attn, value)

        # [N, h, P, c] -> [N, P, h, c] -> [N, P, C]
        out = out.transpose(1, 2).reshape(b_sz, n_patches, -1)
        out = self.out_proj(out)

        return out




class Distortion_Adaptive_Perception(nn.Module):
    def __init__(self, in_dim=128):
        super().__init__()
        self.chanel_in = in_dim
        self.conv2_1 = nn.Conv2d(512, in_dim, 1, 1, 0)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # 可学习的参数
        self.gamma = nn.Parameter(torch.zeros(1))
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.weight = nn.Parameter(torch.ones(in_dim))  # 每个通道有一个自适应权重
        self.softmax = nn.Softmax(dim=-1)
        self.liner = nn.Linear(128, 1, bias=False)

    def forward(self, x1234):

        x1 = x1234[0]
        x2 = x1234[1]
        x3 = x1234[2]
        x4 = x1234[3]

        x2_ = self.upsample(x2) #torch.Size([8, 128, 28, 28])
        x3_ = self.upsample(self.upsample(x3))#torch.Size([8, 128, 28, 28])
        x4_ = self.upsample(self.upsample(x4))#torch.Size([8, 128, 28, 28])
        x_sm = torch.cat([x1, x2_, x3_, x4_], dim=1)#torch.Size([8, 512, 28, 28])
        x_sm = self.conv2_1(x_sm)#torch.Size([8, 128, 28, 28])

        Vps, C, H, W = x_sm.size()

        #得到视口注意力矩阵 VAC
        v_q = self.liner(global_avg_pool2d(x_sm).reshape(Vps, C)) ##torch.Size([8, 1])
        v_k = v_q.permute(1, 0)
        att_Vps_mat = torch.matmul(v_q, v_k)
        att_Vps_mat_new = torch.max(att_Vps_mat, -1, keepdim=True)[0].expand_as(att_Vps_mat)
        vps_mat = self.softmax(att_Vps_mat_new)
        v_v = x_sm.reshape(Vps, C*H*W)
        vps_att = torch.matmul(vps_mat, v_v)
        vps_att = vps_att.reshape(Vps, C, H, W)
        x_sm = vps_att * self.gamma + x_sm

        # ACAC
        c_q = global_avg_pool2d(x_sm).reshape(Vps, C)
        c_k = c_q.permute(1, 0)
        att_C_mat = torch.matmul(c_k, c_q)
        C_mat = self.softmax(att_C_mat) * self.weight
        c_v = x_sm.reshape(C, Vps*H*W)
        C_att = torch.matmul(C_mat, c_v)
        C_att = C_att.reshape(Vps, C, H, W)
        out = C_att + x_sm

        return out.flatten(2)
    

class Multi_scale_Feature_Fusion(nn.Module):
    def __init__(self):
        super().__init__()

        self.liner1 = nn.Linear(1078, 784, bias=False)

    def forward(self, x1234):
        x1 = x1234[0]
        x2 = x1234[1]
        x3 = x1234[2]
        x4 = x1234[3]

        multi_x = torch.cat((x1.flatten(2), x2.flatten(2), x3.flatten(2), x4.flatten(2)), dim=-1)#torch.Size([8, 128, 1078])
        multi_x = self.liner1(multi_x) #torch.Size([8, 128, 784])

        return multi_x


class Viewport_wise_Perceptual_Quality_Prediction(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.ma = MultiHeadAttention(128, 4)
        self.quality = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.GELU(),
            nn.Linear(embed_dim, 1, bias=False)
        )
        self.liner2 = nn.Sequential(
            nn.Linear(28*28*config().viewport_nums, 1, bias=False)
        )
        

    def forward(self, x):

        B, V_H_W, C = x.shape
       
        vplist = torch.tensor([]).cpu()
        for i in range(B):
            seq = x[i].unsqueeze(0)
            seq = self.ma(seq)
            seq = x[i].squeeze(0)
            vplist = torch.cat((vplist, seq.unsqueeze(0).cpu()), dim=0)
        vplist = vplist.cuda() # (B, V, C)
       
        score = torch.tensor([]).cpu()
        for i in range(B):
            vp_ = self.quality(vplist[i]) # 8*128, 1
            vp_ = self.liner2(vp_.permute(1, 0)).permute(1, 0)
            score = torch.cat((score, vp_.unsqueeze(0).cpu()), dim=0)
        score = score.flatten(1).cuda()

        return score

class OIQAND(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.act_layer = nn.GELU()
        self.backbone = timm.create_model('swin_base_patch4_window7_224', checkpoint_path='/home/d310/10t/rjl/swin_base_patch4_window7_224_22kto1k.pth')

        self.conv1_1 = nn.Conv2d(256, embed_dim, 1, 1, 0)
        self.conv2_1 = nn.Conv2d(512, embed_dim, 1, 1, 0)
        self.conv3_1 = nn.Conv2d(1024, embed_dim, 1, 1, 0)
        self.conv4_1 = nn.Conv2d(1024, embed_dim, 1, 1, 0)

        self.MFF = Multi_scale_Feature_Fusion()
        self.DAP = Distortion_Adaptive_Perception()
        self.VPQP = Viewport_wise_Perceptual_Quality_Prediction()

    def forward(self, x):
        B, V, C, H, W = x.shape # V: 视口的数量
        feats = torch.tensor([]).cuda()
        for i in range(B):
            vp_f = self.vp_forward(x[i])
            feats = torch.cat((feats, vp_f.unsqueeze(0)), dim=0)
        # print(feats.shape)#torch.Size([3, 8, 128])
        x = feats
        x = self.VPQP(x)

        return x

    def vp_forward(self, x):
        x3, x0, x1, x2 = self.backbone(x)

        x0 = rearrange(x0, 'V_n (h w) c -> V_n c h w', h=28, w=28)
        x1 = rearrange(x1, 'V_n (h w) c -> V_n c h w', h=14, w=14)
        x2 = rearrange(x2, 'V_n (h w) c -> V_n c h w', h=7, w=7)
        x3 = rearrange(x3, 'V_n (h w) c -> V_n c h w', h=7, w=7)


        x0 = self.conv1_1(x0) # torch.Size([8, 128, 28, 28])
        x1 = self.conv2_1(x1) # torch.Size([8, 128, 14, 14])
        x2 = self.conv3_1(x2) # torch.Size([8, 128, 7, 7])
        x3 = self.conv4_1(x3) # torch.Size([8, 128, 7, 7])

        xs = [x0, x1, x2, x3]

        x_mff = self.MFF(xs)
        x_dap = self.DAP(xs)
        
        multi_x = torch.cat([x_mff, x_dap], dim=1) #torch.Size([8, 256, 784])
        
        multi_x = multi_x.view(config().viewport_nums, 256, 28, 28)

        multi_x = self.conv1_1(multi_x) #torch.Size([8, 128, 28, 28])

        multi_x = multi_x.flatten(2) # torch.Size([8, 128, 784])

        multi_x = multi_x.permute(0, 2, 1)

        multi_x = torch.flatten(multi_x, start_dim=0, end_dim=1)#torch.Size([6272, 128])
        

        return multi_x
