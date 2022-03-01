import torch
from torch.autograd import Variable
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
import numpy as np
import sys
from MSTNEDataset import MSTNEDataSet
FType = torch.FloatTensor
LType = torch.LongTensor
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.linear_model import LogisticRegression
from dataset import _datatset
from dataset import _graph
from dataset import _edge

class MCTNE(torch.nn.Module):
    def __init__(self, file_path, emb_size=128, neg_size=5, hist_len=5, directed=False,
                 learning_rate=0.01, batch_size=1, save_step=1, epoch_num=100):
        super(MCTNE, self).__init__()
        self.emb_size = emb_size
        self.neg_size = neg_size
        self.hist_len = hist_len
        self.lr = learning_rate
        self.batch = batch_size
        self.save_step = save_step     
        self.epochs = epoch_num

        # self.data = MCTNEDataSet(file_path, neg_size, hist_len, directed)
        self.data = MSTNEDataSet(neg_size, hist_len, directed)
        self.snapshot_nums = self.data.datasets.snapshot_nums

        self.node_dim = self.data.get_node_dim()

        if torch.cuda.is_available():
            self.node_emb = Variable(torch.from_numpy(np.random.uniform(
                -1. / np.sqrt(self.node_dim), 1. / np.sqrt(self.node_dim), (self.snapshot_nums, self.node_dim, emb_size))).type(
                FType).cuda(), requires_grad=True)

            self.delta = Variable((torch.zeros(self.node_dim) + 1.).type(FType).cuda(), requires_grad=True)
            
            self.att_param = torch.nn.Parameter(torch.from_numpy(np.random.uniform(
            -1. / np.sqrt(self.node_dim), 1. / np.sqrt(self.node_dim), (self.node_dim,24))).type(
            FType).cuda(), requires_grad=True)

            self.att_param_p = torch.nn.Parameter(torch.from_numpy(np.random.uniform(
            -1. / np.sqrt(self.node_dim), 1. / np.sqrt(self.node_dim), (self.node_dim,24))).type(
            FType).cuda(), requires_grad=True)

        else:
            self.node_emb = Variable(torch.from_numpy(np.random.uniform(
                -1. / np.sqrt(self.node_dim), 1. / np.sqrt(self.node_dim), (self.node_dim, emb_size))).type(
                FType), requires_grad=True)

            self.delta = Variable((torch.zeros(self.node_dim) + 1.).type(FType), requires_grad=True)

            self.att_param = torch.nn.Parameter(torch.from_numpy(np.random.uniform(
            -1. / np.sqrt(self.node_dim), 1. / np.sqrt(self.node_dim), (self.node_dim,24))).type(
            FType), requires_grad=True)

            self.att_param_p = torch.nn.Parameter(torch.from_numpy(np.random.uniform(
            -1. / np.sqrt(self.node_dim), 1. / np.sqrt(self.node_dim), (self.node_dim,24))).type(
            FType) , requires_grad=True)

        self.opt = SGD(lr=learning_rate, params=[self.node_emb, self.att_param, self.att_param_p, self.delta])
        self.loss = torch.FloatTensor()

    def forward(self, idx, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_types, h_t_time, h_time_mask, h_t_masks):
        batch = s_nodes.size()[0]
        s_node_emb = self.node_emb[idx].index_select(0, Variable(s_nodes.view(-1))).view(batch, -1)
        t_node_emb = self.node_emb[idx].index_select(0, Variable(t_nodes.view(-1))).view(batch, -1)
        h_node_emb = self.node_emb[idx].index_select(0, Variable(h_nodes.view(-1))).view(batch, self.hist_len, -1)
        n_node_emb = self.node_emb[idx].index_select(0, Variable(n_nodes.view(-1))).view(batch, self.neg_size, -1)

        h_types = torch.nn.functional.one_hot(h_types.to(torch.int64), num_classes=24)
        att_para = self.att_param.index_select(0, Variable(s_nodes.view(-1))).view(batch, -1)
        att_param_p = self.att_param_p.index_select(0, Variable(s_nodes.view(-1))).view(batch, -1)
        att = softmax((h_types * att_para.unsqueeze(1)).sum(dim=2), dim=1)
        att_p = softmax((h_types * att_param_p.unsqueeze(1)).sum(dim=2), dim=1)

        p_mu = ((s_node_emb - t_node_emb)**2).sum(dim=1).neg()
        p_alpha = ((h_node_emb - t_node_emb.unsqueeze(1))**2).sum(dim=2).neg()
        p_t_alpha = ((h_node_emb - s_node_emb.unsqueeze(1))**2).sum(dim=2).neg()
        
        delta = self.delta.index_select(0, Variable(s_nodes.view(-1))).unsqueeze(1)
        d_time = torch.abs(t_times.unsqueeze(1) - h_times)  # (batch, hist_len)
        d_t_time = torch.abs(t_times.unsqueeze(1) - h_t_time)  # (batch, hist_len)

        a = p_alpha * torch.exp(delta * Variable(d_time)) * Variable(h_t_masks)
        b = p_t_alpha * torch.exp(delta * Variable(d_t_time)) * Variable(h_time_mask)
        p_lambda = p_mu + ((att * (a+b)).sum(dim=1))
        
        n_mu = ((s_node_emb.unsqueeze(1) - n_node_emb)**2).sum(dim=2).neg()
        n_alpha = ((h_node_emb.unsqueeze(2) - n_node_emb.unsqueeze(1))**2).sum(dim=3).neg()
        n_t_alpha = ((t_node_emb.unsqueeze(1) - n_node_emb)**2).sum(dim=2).neg()

        a = n_alpha * torch.exp(delta * Variable(d_time)).unsqueeze(2) * (Variable(h_time_mask).unsqueeze(2))
        b = n_t_alpha.unsqueeze(1) * (torch.exp(delta * Variable(d_t_time)).unsqueeze(2))
        n_lambda = n_mu + ((att.unsqueeze(2) * (a+b)).sum(dim=1))
        
        return p_lambda, n_lambda

    def loss_func(self, idx, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_types, h_t_times, h_time_mask, h_t_masks):
        if torch.cuda.is_available():
            p_lambdas, n_lambdas = self.forward(idx, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_types, h_t_times, h_time_mask, h_t_masks)
            loss = -torch.log(p_lambdas.sigmoid()+ 1e-6) - torch.log(
                n_lambdas.neg().sigmoid() + 1e-6).sum(dim=1)
        else:
            p_lambdas, n_lambdas = self.forward(s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_types, h_t_times, h_time_mask, h_t_masks)
            loss = -torch.log(torch.sigmoid(p_lambdas) + 1e-6) - torch.log(
                torch.sigmoid(torch.neg(n_lambdas)) + 1e-6).sum(dim=1)
        return loss

    def update(self, idx, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_types, h_t_times, h_time_mask, h_t_masks):
        if torch.cuda.is_available():
            self.opt.zero_grad()
            loss = self.loss_func(idx, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_types, h_t_times, h_time_mask, h_t_masks)
            loss = loss.sum()
            self.loss += loss.data
            loss.backward()
            self.opt.step()
        else:
            self.opt.zero_grad()
            loss = self.loss_func(s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_types, h_t_times, h_time_mask, h_t_masks)
            loss = loss.sum()
            self.loss += loss.data
            loss.backward()
            self.opt.step()
    def train(self):
        for epoch in range(self.epochs):
            self.loss = 0.0
            loader = DataLoader(self.data, batch_size=self.batch,
                                shuffle=True, num_workers=0)
            if epoch % self.save_step == 0 and epoch != 0:
                #torch.save(self, './model/dnrl-dblp-%d.bin' % epoch)
                self.save_node_embeddings('../emb/dblp_htne_attn_%d.emb' % (epoch))

            for i_batch, sample_batched in enumerate(loader):

                if torch.cuda.is_available():
                    self.update(sample_batched['idx'].type(LType).cuda(),
                                sample_batched['source_node'].type(LType).cuda().squeeze(0),
                                sample_batched['target_node'].type(LType).cuda().squeeze(0),
                                sample_batched['target_time'].type(FType).cuda().squeeze(0),
                                sample_batched['neg_nodes'].type(LType).cuda().squeeze(0),
                                sample_batched['history_nodes'].type(LType).cuda().squeeze(0),
                                sample_batched['history_times'].type(FType).cuda().squeeze(0),
                                sample_batched['history_types'].type(FType).cuda().squeeze(0),
                                sample_batched['history_target_times'].type(FType).cuda().squeeze(0),
                                sample_batched['history_masks'].type(FType).cuda().squeeze(0),
                                sample_batched['history_t_masks'].type(FType).cuda().squeeze(0))
                else:
                    self.update(sample_batched['source_node'].type(LType),
                                sample_batched['target_node'].type(LType),
                                sample_batched['target_time'].type(FType),
                                sample_batched['neg_nodes'].type(LType),
                                sample_batched['history_nodes'].type(LType),
                                sample_batched['history_times'].type(FType),
                                sample_batched['history_types'].type(FType),
                                sample_batched['history_target_times'].type(FType),
                                sample_batched['history_masks'].type(FType),
                                sample_batched['history_t_masks'].type(FType))
                
                if i_batch % 100 == 0 and i_batch != 0:   
                    sys.stdout.write('\r' + str(i_batch * self.batch) + '\tloss: ' + str(
                        self.loss.cpu().numpy() / (self.batch * i_batch)) + '\tdelta:' + str(
                        self.delta.mean().cpu().data.numpy()))
                    sys.stdout.flush()

            sys.stdout.write('\repoch ' + str(epoch) + ': avg loss = ' +
                             str(self.loss.cpu().numpy() / len(self.data)) + '\n')
            sys.stdout.flush()

            # self.lr_classification('./data/nid2label.txt',0.6)

        self.save_node_embeddings('../emb/dblp_htne_attn_%d.emb' % (self.epochs))

    def save_node_embeddings(self, path):
        if torch.cuda.is_available():
            embeddings = self.node_emb.cpu().data.numpy()
        else:
            embeddings = self.node_emb.data.numpy()
        writer = open(path, 'w')
        writer.write('%d %d\n' % (self.node_dim, self.emb_size))
        for n_idx in range(self.node_dim):
            writer.write(' '.join(str(d) for d in embeddings[n_idx]) + '\n')

        writer.close()
    
    def lr_classification(self,label_data,train_ratio):
        i2l = dict()
        cl_y = []
        with open(label_data, 'r') as reader:
            for line in reader:
                parts = line.strip().split()
                n_id, l_id = int(parts[0]), int(parts[1])
                cl_y.append(l_id)

        x_train, x_valid, y_train, y_valid = train_test_split(self.node_emb.cpu().data.numpy(), cl_y, test_size=1-train_ratio, random_state=9)
        lr = LogisticRegression()
        lr.fit(x_train, y_train)
        y_valid_pred = lr.predict(x_valid)
        micro_f1 = f1_score(y_valid, y_valid_pred, average='micro')
        macro_f1 = f1_score(y_valid, y_valid_pred, average='macro')
        print ('Macro_F1_score:{}'.format(macro_f1))
        print ('Micro_F1_score:{}'.format(micro_f1))
        w = self.att_param + self.att_param_p
        self.viz.heatmap(
            X=(w.sum(dim=0).view(4,6)).cpu().data.numpy(),
            opts=dict(
                colormap='Blues',
            )
        )

if __name__ == '__main__':
    mctne = MCTNE('./data/dblp.txt', directed=False)
    mctne.cuda()
    mctne.train()
