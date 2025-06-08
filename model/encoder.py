from model.att_modules import *
from model.modules import *
import os
import pickle
from sentence_transformers import SentenceTransformer

def makedirs(path) -> None:
    if not os.path.exists(path):
        os.makedirs(path)

class Aggregator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # self.ffn = BasicMLP([dim] + [512] * (depth - 1) + [dim])
        self.ffn = nn.Linear(dim, dim)
        # mean pooling
        self.pool = nn.AdaptiveAvgPool2d((1, dim))
        # # max pooling
        # self.pool = nn.AdaptiveMaxPool2d((1, dim))
        
        # 语义处理层
        self.semantic_layer = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        # 语义门控
        self.semantic_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        
        # 多头自注意力层
        self.self_attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(dim)
        
        # 估计层
        self.estimation_layer = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )

    
    def forward(self, x, m=None):
        x = self.ffn(x)
        if m is not None:
            x = x * m
        x = F.gelu(x)
        
        # used in max pooling
        # max_neg_value = torch.finfo(x.dtype).min
        # x[x == 0] = max_neg_value
        
        x = self.pool(x).squeeze(1)
        x = F.normalize(x)
        
        # 在特征聚合后添加语义处理
        semantic_features = self.semantic_layer(x)
        semantic_weights = self.semantic_gate(x)
        x = x + semantic_features * semantic_weights  # 语义残差连接
        x = F.normalize(x)  # 再次归一化

        # 多头自注意力交互
        x_attn = x.unsqueeze(1)  # [batch, 1, dim]
        attn_output, _ = self.self_attention(x_attn, x_attn, x_attn)
        x = x + attn_output.squeeze(1)  # 残差连接
        x = self.norm(x)
        
        # 基数估计
        estimation = self.estimation_layer(x)
        
        return x, estimation

current_path = os.path.dirname(os.path.abspath(__file__))
split_root_path = os.path.join(current_path, 'data/split_table')
save_root_path = os.path.join(current_path, f'data/embedding')
makedirs(save_root_path)
# ! set PLM path
model_dir = 'sentence-transformers/sentence-t5-large'
plm_model = SentenceTransformer(model_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PostDistillation(nn.Module):
    def __init__(self, depth, latent_dim, data_dim, first_share=False):
        super().__init__()
        self.cross_attn_1 = PostNormAttention(latent_dim, data_dim)
        self.cross_ffn_1 = PostNormFFN(latent_dim)

        if not first_share:
            self.cross_attn_n = PostNormAttention(latent_dim, data_dim)
            self.cross_ffn_n = PostNormFFN(latent_dim)

        self.weight_share = first_share
        self.depth = depth

        # 新增语义投影层 (仅此一处修改)
        self.semantic_projection = nn.Linear(data_dim, latent_dim)
    
    
    def forward(self, data, latents):
        # 语义增强 (新增1行)
        data = data + F.gelu(self.semantic_projection(data))  # 语义残差连接

        latents = latents.repeat(data.size(0), 1, 1)
        latents = self.cross_attn_1(latents, data)
        latents = self.cross_ffn_1(latents)

        cross_attn_n = self.cross_attn_1 if self.weight_share else self.cross_attn_n
        cross_ffn_n = self.cross_ffn_1 if self.weight_share else self.cross_ffn_n

        for _ in range(1, self.depth):
            latents = cross_attn_n(latents, data)
            latents = cross_ffn_n(latents)

        return latents.squeeze(0)
    

class Featurization(nn.Module):
    def __init__(self, distill_depth, data_dim, latent_dim, distill_ratio, distill_share=False):
        super().__init__()
        self.distill_ratio = distill_ratio
        self.aggregator = Aggregator(data_dim)
        # self.distillation = Perceiver(distill_depth, latent_dim, data_dim, distill_share)
        self.distillation = PostDistillation(distill_depth, latent_dim, data_dim, distill_share)

        # 新增语义筛选层 (仅此一处修改)
        self.semantic_gate = nn.Sequential(
            nn.Linear(data_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, pos_samples, neg_samples, mask, distilled_embs=None):
        # # 语义筛选 (新增3行)
        # semantic_weights = self.semantic_gate(x)  # [batch, seq_len, 1]
        # x = x * semantic_weights  # 语义加权
        
        # 直接使用 aggregator 进行特征聚合和语义处理
        set_embs = self.aggregator(x, mask)

        concat_samples = torch.cat((pos_samples, neg_samples), dim=1)
        link_pred = (set_embs.unsqueeze(1) * concat_samples).sum(-1)
        
        if distilled_embs is None:
            distilled_embs = torch.randperm(set_embs.size(0))[:max(int(set_embs.size(0) * self.distill_ratio), 2)]
            # distilled_embs = torch.randperm(set_embs.size(0))[:min(set_embs.size(0), self.distill_size)]
            distilled_embs = set_embs[distilled_embs].detach()
        
        distilled_embs = self.distillation(set_embs.unsqueeze(0), distilled_embs)
        
        return link_pred, set_embs, distilled_embs
