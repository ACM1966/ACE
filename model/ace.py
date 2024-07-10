import torch
import torch.nn as nn
import torch.nn.functional as F
from model.att_modules import *
from model.model import *


# class ACE(nn.Module):
#     def __init__(self, dim, att_layers, mlp_layers, mlp_dim):
#         super().__init__()
#         self.analyzer = nn.ModuleList([])

#         # print('pre-norm')
#         # for _ in range(att_layers):
#         #     self.analyzer.append(nn.ModuleList([PreNormAttention(dim, dim, hidden_dim=dim), PreNormFFN(dim)]))
        
#         print('post-norm')
#         for _ in range(att_layers):
#             self.analyzer.append(nn.ModuleList([PostNormAttention(dim, dim, hidden_dim=dim), PostNormFFN(dim)]))
        
#         self.mlp = BasicMLP([dim] + [mlp_dim] * (mlp_layers - 1) + [1])
    

#     def forward(self, q, x):
#         x = x.repeat(q.size(0), 1, 1)

#         # for _, (attn, ffn) in enumerate(self.analyzer):
#         #     q = q + attn(q, x)
#         #     q = q + ffn(q)
        
#         for _, (attn, ffn) in enumerate(self.analyzer):
#             q = attn(q, x)
#             q = ffn(q)
#         # print(q)
#         res = self.mlp(q)
#         # print(res)
        
#         return res.squeeze(-1)
    

# class ACE(nn.Module):
#     def __init__(self, dim, att_layers, mlp_layers, mlp_dim):
#         super().__init__()
#         self.analyzer = nn.ModuleList([])
#         for _ in range(att_layers):
#             self.analyzer.append(nn.ModuleList([PreNormAttention(dim, dim, hidden_dim=dim), PreNormFFN(dim), PreNormAttention(dim, dim, hidden_dim=dim), PreNormFFN(dim)]))
#         # for _ in range(att_layers):
#         #     self.analyzer.append(nn.ModuleList([PostNormAttention(dim, dim, hidden_dim=dim), PostNormFFN(dim), PostNormAttention(dim, dim, hidden_dim=dim), PostNormFFN(dim)]))
#         self.norm = nn.LayerNorm(dim)
#         self.mean_pool = nn.AdaptiveAvgPool2d((1, dim))
#         self.pool = AttentionPooling(dim)
#         self.mlp = BasicMLP([dim] + [mlp_dim] * (mlp_layers - 1) + [1])
    

#     def forward(self, q, x):
#         x = x.repeat(q.size(0), 1, 1)
#         mask = torch.sign(torch.sum(torch.abs(q), -1))

#         for _, (cross_attn, ffn1, self_attn, ffn2) in enumerate(self.analyzer):
#             q = q + cross_attn(q, x, mask=mask)
#             q = q + ffn1(q)
#             q = q + self_attn(q, mask=mask)
#             q = q + ffn2(q)
#         q = self.norm(q)

#         # for _, (cross_attn, ffn1, self_attn, ffn2) in enumerate(self.analyzer):
#         #     q = cross_attn(q, x, mask=mask)
#         #     q = ffn1(q)
#         #     q = self_attn(q, mask=mask)
#         #     q = ffn2(q)

#         mask = mask.unsqueeze(-1)
#         mask = mask.expand(-1, -1, q.size(-1))
#         q = q * mask
#         mean_q = self.mean_pool(q)
#         q = self.pool(q, mean_q)
#         # print(q)
        
#         res = self.mlp(q)
#         # res = F.elu(res) + 2
#         # print(res)
        
#         return res.squeeze(-1)
    

class ACE(nn.Module):
    def __init__(self, dim, cross_att_layers, self_att_layers, mlp_layers, mlp_dim, data_num, agg_model=None):
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

        # self.analyzer1 = BasicMLP([dim * (data_num + 1)] + [mlp_dim] * (cross_att_layers - 1) + [dim])

        self.analyzer1 = nn.ModuleList([])
        for _ in range(cross_att_layers):
            self.analyzer1.append(nn.ModuleList([PostNormAttention(dim, dim, hidden_dim=dim), PostNormFFN(dim)]))
     
        self.analyzer2 = nn.ModuleList([])
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

        # MLP
        # q = self.analyzer1(q)

        for _, (cross_attn, ffn1) in enumerate(self.analyzer1):
            q = cross_attn(q, x)#, mask=q_mask
            q = ffn1(q)
        
        for _, (self_attn, ffn2) in enumerate(self.analyzer2):
            q = self_attn(q)#, mask=q_mask
            q = ffn2(q)
        # print(q, q.size())
        # exit()
        mask = q_mask.unsqueeze(-1)
        mask = mask.expand(-1, -1, q.size(-1))
        q = q * mask
        q = torch.cat([q, sel], -1)
        q = self.pool(q, q_mask)

        # q = self.agg(q, mask)

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
