import time

from utils.preprocess import *
from utils.criterion import *
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
    _, num_element = read_data(dataset)
    element_emb = nn.Embedding(num_element + 1, dim, padding_idx=0).requires_grad_(False)
    # element_emb.load_state_dict(torch.load('%s_elements_%s_%s.pth' % (dataset, dim, distill_ratio), map_location='cpu'))

    # group_embs = torch.load('save_model/%s_groups_%s_%s.pth' % (dataset, dim, distill_ratio))
    group_embs = torch.load('save_model/%s_groups.pth' % (dataset))
    group_embs = group_embs.detach().to(dev).requires_grad_(False)

    # MLP - used for CA ablation study
    # group_embs = group_embs.detach()
    # flatten_group_embs = torch.flatten(group_embs)

    print(group_embs.size())
    group_embs = group_embs[:int(0.7*group_embs.size(0)), :]
    print(group_embs.size())

    estimator = ACE(dim, args.cross_attn_num, args.self_attn_num, args.mlp_dep, args.mlp_dim).to(dev)

    # agg_model = Aggregator(dim)
    # agg_model.load_state_dict(torch.load('save_model/%s_aggregation.pth' % (dataset)))
    # estimator = ACE(dim, args.cross_attn_num, args.self_attn_num, args.mlp_dep, args.mlp_dim, group_embs.size(0), agg_model).to(dev)
    
    optimizer = torch.optim.Adam([{'params': estimator.parameters(), 'lr': lr}])#, {'params': group_embs, 'lr': lr}, 'weight_decay': 1e-3
    regularizer = ModelRegularizer(reg_weight)

    # train_workload = pd.read_csv('../query/%s/%s/train_%s/%s.csv' % (dataset, workload_type, workload_freq, dataset), header=None)
    # train_dataset = MyQueryDataset(train_workload)
    # train_loader = DataLoader(train_dataset, batch_size, True, collate_fn=MyQueryDataset.collate_fn)
    # val_workload = pd.read_csv('../query/%s/%s/val_%s/%s.csv' % (dataset, workload_type, workload_freq, dataset), header=None)
    # val_dataset = MyQueryDataset(val_workload)
    # val_loader = DataLoader(val_dataset, batch_size, True, collate_fn=MyQueryDataset.collate_fn)

    train_workload = pd.read_csv('F:/SciResearch/ACE/query/%s/%s/train_workload.csv' % (dataset, workload_type), header=None, nrows=200)
    train_dataset = MyQueryDataset(train_workload)
    train_loader = DataLoader(train_dataset, batch_size, True, collate_fn=MyQueryDataset.collate_fn)
    val_workload = pd.read_csv('F:/SciResearch/ACE/query/%s/%s/val_workload.csv' % (dataset, workload_type), header=None, nrows=100)
    val_dataset = MyQueryDataset(val_workload)
    val_loader = DataLoader(val_dataset, batch_size, True, collate_fn=MyQueryDataset.collate_fn)

    print(len(train_loader), len(val_loader))
    # np.save('%s_freq.npy' % dataset,lr)
    degree = np.load('%s_freq.npy' % dataset)
    degree = torch.from_numpy(degree)
    times = []

    if args.m == 'train':
        train_start = time.perf_counter()
        # training
        print('Training in progress..')
        best_loss = 1e50
        best_epoch = 0
        for epoch in range(args.qe):
            for _, batch in enumerate(train_loader):
                optimizer.zero_grad()
                qs, truth = batch
                truth = truth.float().to(dev)
                lst = [element_emb(q + 1) for q in qs]
                sel = [torch.log(degree[q]) for q in qs]
                sel = pad_sequence(sel, True).unsqueeze(-1).to(dev)
                
                # MLP - used for CA ablation study
                # lst = pad_sequence(lst, True)
                # if lst.size(1) > 200:
                #     continue
                # train_q_mask = torch.sign(torch.sum(torch.abs(lst), -1))
                # train_group_embs = flatten_group_embs.repeat(lst.size(0), lst.size(1), 1)
                # # print(train_group_embs.size())
                # mask1 = train_q_mask.unsqueeze(-1)
                # mask1 = mask1.expand(-1, -1, train_group_embs.size(-1))
                # train_group_embs = train_group_embs * mask1
                # lst = torch.cat((train_group_embs, lst), dim=-1)
                
                # lst = lst.to(dev)
                # train_q_mask = train_q_mask.to(dev)
                # pred = estimator(lst, train_q_mask, sel)

                # original ACE
                lst = pad_sequence(lst, True).to(dev)

                pred = estimator(lst, group_embs, sel)

                l_train = weight_q_error(truth, pred).sum()
                l_reg = regularizer(estimator)
                train_loss = l_train + l_reg
                train_loss.backward()
                optimizer.step()
            
            if epoch > 9:
                # torch.cuda.empty_cache()
                with torch.no_grad():
                    # validation
                    val_preds = torch.tensor([])
                    val_truth = torch.tensor([])
                    for _, val_batch in enumerate(val_loader):
                        val_qs, val_cs = val_batch
                        val_truth = torch.cat((val_truth, val_cs))
                        val_lst = [element_emb(q + 1) for q in val_qs]
                        val_sel = [torch.log(degree[q]) for q in val_qs]
                        val_sel = pad_sequence(val_sel, True).unsqueeze(-1).to(dev)

                        # MLP - used for CA ablation study
                        # val_lst = pad_sequence(val_lst, True)
                        # if val_lst.size(1) > 200:
                        #     continue
                        # val_q_mask = torch.sign(torch.sum(torch.abs(val_lst), -1))
                        # val_group_embs = flatten_group_embs.repeat(val_lst.size(0), val_lst.size(1), 1)
                        # mask1 = val_q_mask.unsqueeze(-1)
                        # mask1 = mask1.expand(-1, -1, val_group_embs.size(-1))
                        # val_group_embs = val_group_embs * mask1
                        # val_lst = torch.cat((val_group_embs, val_lst), dim=-1)
                        
                        # val_lst = val_lst.to(dev)
                        # val_q_mask = val_q_mask.to(dev)
                        # val_pred = estimator(val_lst, val_q_mask, val_sel)
                        
                        #  original ACE
                        val_lst = pad_sequence(val_lst, True).to(dev)

                        val_pred = estimator(val_lst, group_embs, val_sel)

                        val_preds = torch.cat((val_preds, val_pred.cpu()))
                    val_loss = q_error_criterion(val_truth, val_preds).mean()
                    # print(epoch, val_loss.item(), val_preds.min().item(), val_preds.max().item())
                    if val_loss.item() < best_loss and val_preds.min().item() > 0:
                        best_loss = val_loss.item()
                        best_epoch = epoch
                        torch.save(estimator.state_dict(), 'save_model/%s_estimator_%s_%s.pth' % (dataset, workload_type, distill_ratio))
                # torch.cuda.empty_cache()
        train_end = time.perf_counter()
        print(best_epoch, best_loss, 'time', (train_end - train_start) / 60)
    else:
        # test
        print('Evaluation')
        test_workload = pd.read_csv('F:/SciResearch/ACE/query/%s/%s/test_%s/%s.csv' % (dataset, workload_type, workload_freq, dataset), header=None)
        test_dataset = MyQueryDataset(test_workload)
        test_loader = DataLoader(test_dataset, batch_size, collate_fn=MyQueryDataset.collate_fn)
        estimator.load_state_dict(torch.load('save_model/%s_estimator_%s_%s.pth' % (dataset, workload_type, distill_ratio)))
        
        with torch.no_grad():   
            test_preds = torch.tensor([])
            test_truth = torch.tensor([])
            for _, test_batch in enumerate(test_loader):
                test_qs, test_cs = test_batch
                test_truth = torch.cat((test_truth, test_cs))
                test_lst = [element_emb(q + 1) for q in test_qs]
                test_sel = [torch.log(degree[q]) for q in test_qs]
                test_sel = pad_sequence(test_sel, True).unsqueeze(-1).to(dev)

                # MLP - used for CA ablation study
                # test_lst = pad_sequence(test_lst, True)
                # if test_lst.size(1) > 200:
                #     continue
                # test_q_mask = torch.sign(torch.sum(torch.abs(test_lst), -1))
                # test_group_embs = flatten_group_embs.repeat(test_lst.size(0), test_lst.size(1), 1)
                # mask1 = test_q_mask.unsqueeze(-1)
                # mask1 = mask1.expand(-1, -1, test_group_embs.size(-1))
                # test_group_embs = test_group_embs * mask1
                # test_lst = torch.cat((test_group_embs, test_lst), dim=-1)
                
                # test_lst = test_lst.to(dev)
                # test_q_mask = test_q_mask.to(dev)
                # test_pred = estimator(test_lst, test_q_mask, test_sel)

                # original ACE
                test_lst = pad_sequence(test_lst, True).to(dev)

                test_pred = estimator(test_lst, group_embs, test_sel)

                test_preds = torch.cat((test_preds, test_pred.cpu()))

            times = []
            test_loader = DataLoader(test_dataset, collate_fn=MyQueryDataset.collate_fn)
            for _, test_batch in enumerate(test_loader):
                start = time.perf_counter()
                test_qs, test_cs = test_batch
                test_lst = [element_emb(q + 1) for q in test_qs]
                test_sel = [torch.log(degree[q]) for q in test_qs]
                test_sel = pad_sequence(test_sel, True).unsqueeze(-1).to(dev)

                # MLP - used for CA ablation study
                # test_lst = pad_sequence(test_lst, True)
                # if test_lst.size(1) > 200:
                #     continue
                # test_q_mask = torch.sign(torch.sum(torch.abs(test_lst), -1))
                # test_group_embs = flatten_group_embs.repeat(test_lst.size(0), test_lst.size(1), 1)
                # mask1 = test_q_mask.unsqueeze(-1)
                # mask1 = mask1.expand(-1, -1, test_group_embs.size(-1))
                # test_group_embs = test_group_embs * mask1
                # test_lst = torch.cat((test_group_embs, test_lst), dim=-1)
                
                # test_lst = test_lst.to(dev)
                # test_pred = estimator(test_lst, test_q_mask.to(dev), test_sel)

                # original ACE
                test_lst = pad_sequence(test_lst, True).to(dev)
                
                test_pred = estimator(test_lst, group_embs, test_sel)
                end = time.perf_counter()
                times.append((end - start) * 1000)

            test_loss = q_error_criterion(test_truth, test_preds)

            print("%.3f, %.3f, %.3f, %.3f, %.3f" % (test_loss.mean().item(), test_loss.median().item(), test_loss.quantile(0.95, dim=0, interpolation='nearest').item(), test_loss.quantile(0.99, dim=0, interpolation='nearest').item(), np.mean(times)))
