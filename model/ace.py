import torch
import torch.nn as nn
import torch.nn.functional as F
from model.att_modules import *
from model.encoder import *
    

class ACE(nn.Module):
    def __init__(self, dim, cross_att_layers, self_att_layers, mlp_layers, mlp_dim, data_num=None, agg_model=None):
        super().__init__()
        # self.analyzer1 = nn.ModuleList([])
        # print('pre-norm')
        # for _ in range(cross_att_layers):
        #     self.analyzer1.append(nn.ModuleList([PreNormAttention(dim, dim, hidden_dim=dim), PreNormFFN(dim)]))
        # self.norm1 = nn.LayerNorm(dim)

        # self.analyzer2 = nn.ModuleList([])
        # for _ in range(self_att_layers):
        #     self.analyzer2.append(nn.ModuleList([PreNormAttention(dim, dim, hidden_dim=dim), PreNormFFN(dim)]))
        # self.norm2 = nn.LayerNorm(dim)

        # MLP - used for CA ablation study
        # self.analyzer1 = BasicMLP([dim * (data_num + 1)] + [mlp_dim] * (cross_att_layers - 1) + [dim])

        self.analyzer1 = nn.ModuleList([])
        for _ in range(cross_att_layers):
            self.analyzer1.append(nn.ModuleList([PostNormAttention(dim, dim, hidden_dim=dim), PostNormFFN(dim)]))

        # MLP - used for SA ablation study
        # self.analyzer2 = BasicMLP([dim] + [mlp_dim] * (self_att_layers - 1) + [dim])

        self.analyzer2 = nn.ModuleList()
        for _ in range(self_att_layers):
            self.analyzer2.append(nn.ModuleList([PostNormAttention(dim, dim, hidden_dim=dim), PostNormFFN(dim)]))

        # self.mean_pool = nn.AdaptiveAvgPool2d((1, dim))
        # self.pool = AttentionPooling(dim, 1)
        # self.agg = agg_model
        # self.mlp = BasicMLP([dim] + [mlp_dim] * (mlp_layers - 1) + [1])

        self.mean_pool = nn.AdaptiveAvgPool2d((1, dim + 1))
        self.pool = Pooling(dim + 1, 1)
        self.mlp = BasicMLP([dim + 1] + [mlp_dim] * (mlp_layers - 1) + [1])
    

    def forward(self, q, x, sel):
    # def forward(self, q, q_mask, sel):
        x = x.repeat(q.size(0), 1, 1)
        q_mask = torch.sign(torch.sum(torch.abs(q), -1))

        # for _, (cross_attn, ffn1) in enumerate(self.analyzer1):
        #     q = q + cross_attn(q, x, mask=mask)
        #     q = q + ffn1(q)
        # q = self.norm1(q)

        # for _, (self_attn, ffn2) in enumerate(self.analyzer2):
        #     q = q + self_attn(q, mask=mask)
        #     q = q + ffn2(q)
        # q = self.norm2(q)

        for _, (cross_attn, ffn1) in enumerate(self.analyzer1):
            q = cross_attn(q, x)#, mask=q_mask
            q = ffn1(q)
        
        # MLP - used for CA ablation study
        # q = self.analyzer1(q)
        
        for _, (self_attn, ffn2) in enumerate(self.analyzer2):
            q = self_attn(q)#, mask=q_mask
            q = ffn2(q)
        
        # MLP - used for SA ablation study
        # q = self.analyzer2(q)

        mask = q_mask.unsqueeze(-1)
        mask = mask.expand(-1, -1, q.size(-1))
        q = q * mask
        q = torch.cat([q, sel], -1)
        q = self.pool(q, q_mask)

        res = self.mlp(q)

        return res


class Pooling(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.mab = Attention(dim, dim, dim, heads)
        self.pooling_emb = nn.Parameter(torch.zeros(1, dim))
        self.pool = nn.AdaptiveAvgPool2d((1, dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = BasicMLP([dim, 512, dim])
        nn.init.trunc_normal_(self.pooling_emb, std=0.02)

    def forward(self, x, mask=None):
        # out = self.pooling_emb.repeat(x.size(0), 1, 1)
        # out = self.norm1(self.pool(x) + self.mab(out, x, k_mask=mask))
        out = self.pool(x)

        out = self.norm1(out + self.mab(out, x, k_mask=mask))
        out = self.norm2(out + self.ffn(out))

        # return out
        return out.squeeze(1)
