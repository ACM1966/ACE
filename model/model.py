import torch
import torch.nn as nn
import torch.nn.functional as F
from model.att_modules import *
    

class Aggregator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # self.ffn = BasicMLP([dim] + [512] * (depth - 1) + [dim])
        self.ffn = nn.Linear(dim, dim)
        # mean pooling
        self.pool = nn.AdaptiveAvgPool2d((1, dim))
        # # max pooling
        # self.pool = nn.AdaptiveMaxPool2d((1, dim))

    
    def forward(self, x, m=None):
        x = self.ffn(x)
        if m is not None:
            x = x * m
        
        # used in max pooling
        # max_neg_value = torch.finfo(x.dtype).min
        # x[x == 0] = max_neg_value

        x = F.gelu(x)
        x = self.pool(x).squeeze(1)
        x = F.normalize(x)
        
        return x


class GEGLU(nn.Module):
    def __init__(self):
        super(GEGLU, self).__init__()
    

    def forward(self, x):
        x, gates = x.chunk(2, -1)
        out = x * F.gelu(gates)
        return out


class FFN(nn.Module):
    def __init__(self, dim, multi=4):
        super(FFN, self).__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim * multi * 2), GEGLU(), nn.Linear(dim * multi, dim))
    

    def forward(self, x):
        return self.net(x)


class PreNormAttention(nn.Module):
    def __init__(self, dim, data_dim=None, hidden_dim=512, num_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        if data_dim is not None:
            self.norm_data = nn.LayerNorm(data_dim)
            self.attention = Attention(dim, data_dim, hidden_dim, num_heads)
        else:
            self.attention = Attention(dim, dim, hidden_dim, num_heads)
        # self.out = nn.Linear(hidden_dim, dim)


    def forward(self, x, d=None, mask=None):
        x = self.norm(x)
        if d is not None:
            d = self.norm_data(d)
            att_out = self.attention(x, d, mask)
        else:
            att_out = self.attention(x, x, mask)
        # att_out = self.out(att_out)
        
        return att_out
    

class PostNormAttention(nn.Module):
    def __init__(self, dim, data_dim=None, hidden_dim=512, num_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        if data_dim is not None:
            self.attention = Attention(dim, data_dim, hidden_dim, num_heads)
        else:
            self.attention = Attention(dim, dim, hidden_dim, num_heads)
        # self.out = nn.Linear(hidden_dim, dim)


    def forward(self, x, d=None, mask=None):
        if d is not None:
            x = self.norm(x + self.attention(x, d, mask))
        else:
            x = self.norm(x + self.attention(x, x, mask))
        # att_out = self.out(att_out)
        
        return x
    

class PreNormFFN(nn.Module):
    def __init__(self, dim, multi=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = FFN(dim, multi)
    

    def forward(self, x):
        x = self.norm(x)
        out = self.ffn(x)

        return out


class PostNormFFN(nn.Module):
    def __init__(self, dim, multi=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ffn = FFN(dim, multi)
    

    def forward(self, x):
        x = self.norm(x + self.ffn(x))

        return x


class Perceiver(nn.Module):
    def __init__(self, depth, latent_dim, data_dim, first_share=False):
        super().__init__()
        self.cross_attn_1 = PreNormAttention(latent_dim, data_dim)
        self.cross_ffn_1 = PreNormFFN(latent_dim)
        self.self_attn_1 = PreNormAttention(latent_dim)
        self.self_ffn_1 = PreNormFFN(latent_dim)
        if not first_share:
            self.cross_attn_n = PreNormAttention(latent_dim, data_dim)
            self.cross_ffn_n = PreNormFFN(latent_dim)
            self.self_attn_n = PreNormAttention(latent_dim)
            self.self_ffn_n = PreNormFFN(latent_dim)
        
        self.norm = nn.LayerNorm(latent_dim)
        self.weight_share = first_share
        self.depth = depth
    
    
    def forward(self, data, latents):
        latents = latents.repeat(data.size(0), 1, 1)
        latents = latents + self.cross_attn_1(latents, data)
        latents = latents + self.cross_ffn_1(latents)
        latents = latents + self.self_attn_1(latents)
        latents = latents + self.self_ffn_1(latents)

        cross_attn_n = self.cross_attn_1 if self.weight_share else self.cross_attn_n
        cross_ffn_n = self.cross_ffn_1 if self.weight_share else self.cross_ffn_n
        self_attn_n = self.self_attn_1 if self.weight_share else self.self_attn_n
        self_ffn_n = self.self_ffn_1 if self.weight_share else self.self_ffn_n

        for _ in range(1, self.depth):
            latents = latents + cross_attn_n(latents, data)
            latents = latents + cross_ffn_n(latents)
            latents = latents + self_attn_n(latents)
            latents = latents + self_ffn_n(latents)
            
        latents = self.norm(latents)

        return latents.squeeze(0)


class PostDistillation(nn.Module):
    def __init__(self, depth, latent_dim, data_dim, first_share=False):
        super().__init__()
        self.cross_attn_1 = PostNormAttention(latent_dim, data_dim)
        self.cross_ffn_1 = PostNormFFN(latent_dim)
        # self.self_attn_1 = PostNormAttention(latent_dim)
        # self.self_ffn_1 = PostNormFFN(latent_dim)
        if not first_share:
            self.cross_attn_n = PostNormAttention(latent_dim, data_dim)
            self.cross_ffn_n = PostNormFFN(latent_dim)
            # self.self_attn_n = PostNormAttention(latent_dim)
            # self.self_ffn_n = PostNormFFN(latent_dim)

        self.weight_share = first_share
        self.depth = depth
    
    
    def forward(self, data, latents):
        latents = latents.repeat(data.size(0), 1, 1)
        latents = self.cross_attn_1(latents, data)
        latents = self.cross_ffn_1(latents)
        # latents = self.self_attn_1(latents)
        # latents = self.self_ffn_1(latents)

        cross_attn_n = self.cross_attn_1 if self.weight_share else self.cross_attn_n
        cross_ffn_n = self.cross_ffn_1 if self.weight_share else self.cross_ffn_n
        # self_attn_n = self.self_attn_1 if self.weight_share else self.self_attn_n
        # self_ffn_n = self.self_ffn_1 if self.weight_share else self.self_ffn_n

        for _ in range(1, self.depth):
            latents = cross_attn_n(latents, data)
            latents = cross_ffn_n(latents)
            # latents = self_attn_n(latents)
            # latents = self_ffn_n(latents)

        return latents.squeeze(0)
    

class Featurization(nn.Module):
    def __init__(self, distill_depth, data_dim, latent_dim, distill_ratio, distill_share=False):
        super().__init__()
        self.distill_ratio = distill_ratio
        self.aggregator = Aggregator(data_dim)
        # self.distillation = Perceiver(distill_depth, latent_dim, data_dim, distill_share)
        self.distillation = PostDistillation(distill_depth, latent_dim, data_dim, distill_share)
    

    def forward(self, x, pos_samples, neg_samples, mask, distilled_embs=None):
        set_embs = self.aggregator(x, mask)

        concat_samples = torch.cat((pos_samples, neg_samples), dim=1)
        link_pred = (set_embs.unsqueeze(1) * concat_samples).sum(-1)
        
        if distilled_embs is None:
            distilled_embs = torch.randperm(set_embs.size(0))[:max(int(set_embs.size(0) * self.distill_ratio), 2)]
            # distilled_embs = torch.randperm(set_embs.size(0))[:min(set_embs.size(0), self.distill_size)]
            distilled_embs = set_embs[distilled_embs].detach()
        
        distilled_embs = self.distillation(set_embs.unsqueeze(0), distilled_embs)
        
        return link_pred, set_embs, distilled_embs
