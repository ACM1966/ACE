import time
from pathlib import Path

from utils.preprocess import *
from utils.customized_dataset import *
from utils.arg_parser import *
from model.encoder import *
from model.ace import *


if __name__ == '__main__':
    args = make_args()

    dataset = args.d
    dim = args.dim
    distill_ratio = args.r
    workload_type = args.qt
    workload_freq = args.qf
    lr = args.ql
    reg_weight = args.qr
    batch_size = args.qb
    if args.dev < 0:
        dev = 'cpu'
    else:
        dev = 'cuda:%d' % args.dev
    print(args)

    seed_random(args.s)
    # TODO: change the path of the folder
    folder = os.listdir('/mnt/md1/yufan/data/dynamic/%s' % dataset)
    num_parts = len(folder) - 1
    times = []

    for idx in range(-1, num_parts):
        if idx == -1:
            data = pd.read_csv('/mnt/md1/yufan/data/dynamic/%s/base.csv' % dataset, header=None)
            num_element = data[0][0]
            data = data[1:].reset_index(drop=True)
            element_emb = nn.Embedding(num_element + 1, dim, padding_idx=0).requires_grad_(False)
        else:
            data = pd.read_csv('/mnt/md1/yufan/data/dynamic/%s/%s.csv' % (dataset, idx), header=None)
            if idx > 0:
                is_delete = data[0][0]
                data = data[1:].reset_index(drop=True)

        agg_model = Aggregator(dim)
        agg_model.load_state_dict(torch.load('save_model/%s_aggregation.pth' % (dataset)))
        dis_model = PostDistillation(args.dis_dep, dim, dim, False)
        dis_model.load_state_dict(torch.load('save_model/%s_distillation.pth' % (dataset)))
        
        data_set = SequentialDataset(data, 1, args.neg, num_element)
        if idx == -1:
            loader = DataLoader(data_set, args.db, collate_fn=SequentialDataset.collate_fn)
            group_embs = torch.tensor([])
        else:
            loader = DataLoader(data_set, len(data_set), collate_fn=SequentialDataset.collate_fn)
            start = time.perf_counter()
            group_embs = torch.load('save_model/%s/groups_%d.pth' % (dataset, idx - 1))
            if is_delete:
                group_embs = torch.load('save_model/%s/groups_%d.pth' % (dataset, idx - 2))

        with torch.no_grad():
            for _, (data, pos_idx, neg_idx) in enumerate(loader):
                data = pad_sequence(data, True)
                element_embs = element_emb(data)
                masks = torch.sign(torch.sum(torch.abs(element_embs), -1))
                masks = masks.unsqueeze(-1)
                masks = masks.expand(-1, -1, element_embs.size(-1))
                pos_sample = element_emb(pos_idx)
                neg_sample = element_emb(neg_idx)
                
                set_embs = agg_model(element_embs, masks)

                group_idx = torch.randperm(set_embs.size(0))[:max(int(set_embs.size(0) * distill_ratio), 2)]
                groups = set_embs[group_idx].detach()
                groups = dis_model(set_embs.unsqueeze(0), groups)
                group_embs = torch.cat((group_embs, groups))
        torch.save(group_embs, 'save_model/%s/groups_%d.pth' % (dataset, idx))
        if idx != -1:
            end = time.perf_counter()
            times.append((end - start) * 1000)
    
    print(np.mean(times) / 1000)
