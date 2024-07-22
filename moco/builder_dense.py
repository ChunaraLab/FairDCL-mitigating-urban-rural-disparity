# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
from mmcv.cnn import kaiming_init, normal_init
from collections import OrderedDict
import torch.distributed as dist
import math
import torch.nn.functional as F

class ContrastiveHead(nn.Module):
    """Head for contrastive learning.
    Args:
        temperature (float): The temperature hyper-parameter that
            controls the concentration level of the distribution.
            Default: 0.1.
    """

    def __init__(self, temperature=0.1):
        super(ContrastiveHead, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature

    def forward(self, pos, neg):
        """Forward head.
        Args:
            pos (Tensor): Nx1 positive similarity.
            neg (Tensor): Nxk negative similarity.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        N = pos.size(0)
        logits = torch.cat((pos, neg), dim=1)
        logits /= self.temperature
        labels = torch.zeros((N, ), dtype=torch.long).cuda()
        losses = dict()
        losses['loss_contra'] = self.criterion(logits, labels)
        return losses


def _init_weights(module, init_linear='normal', std=0.01, bias=0.):
    assert init_linear in ['normal', 'kaiming'], \
        "Undefined init_linear: {}".format(init_linear)
    for m in module.modules():
        if isinstance(m, nn.Linear):
            if init_linear == 'normal':
                normal_init(m, std=std, bias=bias)
            else:
                kaiming_init(m, mode='fan_in', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d,
                            nn.GroupNorm, nn.SyncBatchNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class DenseCLNeck(nn.Module):
    '''The non-linear neck in DenseCL.
        Single and dense in parallel: fc-relu-fc, conv-relu-conv
    '''
    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 num_grid=None):
        super(DenseCLNeck, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels), nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels))

        self.with_pool = num_grid != None
        if self.with_pool:
            self.pool = nn.AdaptiveAvgPool2d((num_grid, num_grid))
        self.mlp2 = nn.Sequential(
            nn.Conv2d(in_channels, hid_channels, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hid_channels, out_channels, 1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        # assert len(x) == 1
        # x = x[0]

        avgpooled_x = self.avgpool(x)
        avgpooled_x = self.mlp(avgpooled_x.view(avgpooled_x.size(0), -1))

        if self.with_pool:
            x = self.pool(x) # sxs
        x = self.mlp2(x) # sxs: bxdxsxs
        avgpooled_x2 = self.avgpool2(x) # 1x1: bxdx1x1
        x = x.view(x.size(0), x.size(1), -1) # bxdxs^2
        avgpooled_x2 = avgpooled_x2.view(avgpooled_x2.size(0), -1) # bxd
        return [avgpooled_x, x, avgpooled_x2]


def parse_losses(losses):
    log_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)

    log_vars['loss'] = loss
    for loss_name, loss_value in log_vars.items():
        # reduce loss when distributed training
        if dist.is_available() and dist.is_initialized():
            loss_value = loss_value.data.clone()
            dist.all_reduce(loss_value.div_(dist.get_world_size()))
        log_vars[loss_name] = loss_value.item()

    return loss, log_vars    


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, base_encoder2, feat_dim=512, queue_len=11392, momentum=0.999, loss_lambda=0.5):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 2048)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.encoder_q = nn.Sequential(
            nn.Sequential(*(list(base_encoder.children())[:-2])), DenseCLNeck(2048, 2048, 512))
        self.encoder_k = nn.Sequential(
            nn.Sequential(*(list(base_encoder2.children())[:-2])), DenseCLNeck(2048, 2048, 512))
        self.backbone = self.encoder_q[0]

        for param in self.encoder_k.parameters():
            param.requires_grad = False

        self.head = ContrastiveHead(temperature=0.2)

        # Initiate weights
        self.encoder_q[1].init_weights(init_linear='kaiming')
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)

        self.queue_len = queue_len
        self.momentum = momentum
        self.loss_lambda = loss_lambda

        # create the queue
        self.register_buffer("queue", torch.randn(feat_dim, queue_len))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        # create the second queue for dense output
        self.register_buffer("queue2", torch.randn(feat_dim, queue_len))
        self.queue2 = nn.functional.normalize(self.queue2, dim=0)
        self.register_buffer("queue2_ptr", torch.zeros(1, dtype=torch.long))


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue2(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue2_ptr)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue2[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue2_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]


    def forward(self, im_q, im_k, training_mine=False):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        if(training_mine):
            with torch.no_grad():
                ll = list(self.encoder_q[0].children())
                q_l1 = nn.Sequential(*ll[:5])(im_q)
                q_l2 = ll[5](q_l1)
                q_l3 = nn.Sequential(*ll[6:7])(q_l2)
                q_l4 = ll[7](q_l3)
                
            return q_l1, q_l2, q_l3, q_l4
        


        ll = list(self.encoder_q[0].children())
        q_l1 = nn.Sequential(*ll[:5])(im_q)
        q_l2 = ll[5](q_l1)
        q_l3 = nn.Sequential(*ll[6:7])(q_l2)
        q_l4 = ll[7](q_l3)

        q_b = q_l4  # backbone features

        # print(q_b.shape): torch.Size([16, 2048, 16, 16])
        q, q_grid, q2 = self.encoder_q[1](q_b)  # queries: NxC; NxCxS^2
      
        # q_b = q_b[0]
        q_b = q_b.view(q_b.size(0), q_b.size(1), -1)
        # print(q_b.shape): torch.Size([16, 2048, 256])

        q = nn.functional.normalize(q, dim=1)
        q2 = nn.functional.normalize(q2, dim=1)
        q_grid = nn.functional.normalize(q_grid, dim=1)
        q_b = nn.functional.normalize(q_b, dim=1)
        # print(q_b.shape): torch.Size([16, 2048, 256])
        backbone_features = q2

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # # shuffle for making use of BN
            # im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k_b = self.encoder_k[0](im_k)
            k, k_grid, k2 = self.encoder_k[1](k_b)  # keys: NxC; NxCxS^2
            # k_b = k_b[0]
            k_b = k_b.view(k_b.size(0), k_b.size(1), -1)

            k = nn.functional.normalize(k, dim=1)
            k2 = nn.functional.normalize(k2, dim=1)
            k_grid = nn.functional.normalize(k_grid, dim=1)
            k_b = nn.functional.normalize(k_b, dim=1)

            # # undo shuffle
            # k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            # k2 = self._batch_unshuffle_ddp(k2, idx_unshuffle)
            # k_grid = self._batch_unshuffle_ddp(k_grid, idx_unshuffle)
            # k_b = self._batch_unshuffle_ddp(k_b, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= 0.07

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()


        # feat point set sim
        backbone_sim_matrix = torch.matmul(q_b.permute(0, 2, 1), k_b)
        densecl_sim_ind = backbone_sim_matrix.max(dim=2)[1] # NxS^2

        indexed_k_grid = torch.gather(k_grid, 2, densecl_sim_ind.unsqueeze(1).expand(-1, k_grid.size(1), -1)) # NxCxS^2
        densecl_sim_q = (q_grid * indexed_k_grid).sum(1) # NxS^2

        l_pos_dense = densecl_sim_q.view(-1).unsqueeze(-1) # NS^2X1

        # print(l_pos_dense.shape)
        
        q_grid = q_grid.permute(0, 2, 1)
        q_grid = q_grid.reshape(-1, q_grid.size(2))
        l_neg_dense = torch.einsum('nc,ck->nk', [q_grid,
                                            self.queue2.clone().detach()])

        loss_single = self.head(l_pos, l_neg)['loss_contra']
        loss_dense = self.head(l_pos_dense, l_neg_dense)['loss_contra']
        

        losses = dict()
        losses['loss_contra_single'] = loss_single * (1 - self.loss_lambda)
        losses['loss_contra_dense'] = loss_dense * self.loss_lambda

        self._dequeue_and_enqueue(k)
        self._dequeue_and_enqueue2(k2)

        all_loss, log_var = parse_losses(losses)

        return all_loss, log_var, logits, labels, backbone_features, q_l1, q_l2, q_l3, q_l4




class Mine1(nn.Module):
    def __init__(self, input_size=514, hidden_size1=514, hidden_size2=100, output_size=256):
        super().__init__()

        self.fc1_3 = nn.Conv2d(258, 258, 1)
        self.fc2_3 = nn.Conv2d(258, 20, 1)
        self.fc3_3 = nn.Conv2d(20, 1, 1)

        self.relu1_3 = nn.ReLU(inplace=True)
        self.relu2_3 = nn.ReLU(inplace=True)

        self.ma_rate = 0.01
        self.ma_et = 1.

    def forward(self, new_batch3, measure='JSD'):
        output = self.relu1_3(self.fc1_3(new_batch3))
        output = self.relu2_3(self.fc2_3(output))
        output = self.fc3_3(output)

        batch_s = int(len(output)/2)

        if (measure == 'Mine'):
            t = output[0:batch_s]
            # et = self.relu1(self.fc1(new_batch_marginal))
            # et = self.relu2(self.fc2(et))
            et = torch.exp(output[batch_s:])

            mi_lb = torch.mean(t) - torch.log(torch.mean(et))
            self.ma_et = (1-self.ma_rate)*self.ma_et + self.ma_rate*torch.mean(et)

            loss = -(torch.mean(t) - (1/self.ma_et.mean()).detach()*torch.mean(et))
            # loss = -mi_lb

            return loss, mi_lb

        elif (measure == 'JSD'):
            log_2 = math.log(2.)
            E_pos = log_2 - F.softplus(-output[0:batch_s])
            E_neg = F.softplus(-output[batch_s:]) + output[batch_s:] - log_2
            difference = E_pos.mean() - E_neg.mean()
            measure_est = 0.5 * difference

        return -difference, measure_est


class Mine2(nn.Module):
    def __init__(self, input_size=514, hidden_size1=514, hidden_size2=100, output_size=256):
        super().__init__()

        self.fc1_3 = nn.Conv2d(514, 514, 1)
        self.fc2_3 = nn.Conv2d(514, 50, 1)
        self.fc3_3 = nn.Conv2d(50, 1, 1)

        self.relu1_3 = nn.ReLU(inplace=True)
        self.relu2_3 = nn.ReLU(inplace=True)

        self.ma_rate = 0.01
        self.ma_et = 1.

    def forward(self, new_batch3, measure='JSD'):
        output = self.relu1_3(self.fc1_3(new_batch3))
        output = self.relu2_3(self.fc2_3(output))
        output = self.fc3_3(output)

        batch_s = int(len(output)/2)

        if (measure == 'Mine'):
            t = output[0:batch_s]

            # et = self.relu1(self.fc1(new_batch_marginal))
            # et = self.relu2(self.fc2(et))
            et = torch.exp(output[batch_s:])

            mi_lb = torch.mean(t) - torch.log(torch.mean(et))
            self.ma_et = (1-self.ma_rate)*self.ma_et + self.ma_rate*torch.mean(et)

            loss = -(torch.mean(t) - (1/self.ma_et.mean()).detach()*torch.mean(et))
            # loss = -mi_lb

            return loss, mi_lb

        elif (measure == 'JSD'):
            log_2 = math.log(2.)
            E_pos = log_2 - F.softplus(-output[0:batch_s])
            E_neg = F.softplus(-output[batch_s:]) + output[batch_s:] - log_2
            difference = E_pos.mean() - E_neg.mean()
            measure_est = 0.5 * difference

        return -difference, measure_est


class Mine3(nn.Module):
    def __init__(self, input_size=1026, hidden_size1=1026, hidden_size2=100, output_size=256):
        super().__init__()

        self.fc1_3 = nn.Conv2d(1026, 1026, 1)
        self.fc2_3 = nn.Conv2d(1026, 100, 1)
        self.fc3_3 = nn.Conv2d(100, 1, 1)

        self.relu1_3 = nn.ReLU(inplace=True)
        self.relu2_3 = nn.ReLU(inplace=True)

        self.ma_rate = 0.01
        self.ma_et = 1.

    def forward(self, new_batch3, measure='JSD'):
        output = self.relu1_3(self.fc1_3(new_batch3))
        output = self.relu2_3(self.fc2_3(output))
        output = self.fc3_3(output)

        batch_s = int(len(output)/2)

        if (measure == 'Mine'):
            t = output[0:batch_s]
  
            # et = self.relu1(self.fc1(new_batch_marginal))
            # et = self.relu2(self.fc2(et))
            et = torch.exp(output[batch_s:])

            mi_lb = torch.mean(t) - torch.log(torch.mean(et))
            self.ma_et = (1-self.ma_rate)*self.ma_et + self.ma_rate*torch.mean(et)

            loss = -(torch.mean(t) - (1/self.ma_et.mean()).detach()*torch.mean(et))
            # loss = -mi_lb

            return loss, mi_lb

        elif (measure == 'JSD'):
            log_2 = math.log(2.)
            E_pos = log_2 - F.softplus(-output[0:batch_s])
            E_neg = F.softplus(-output[batch_s:]) + output[batch_s:] - log_2
            difference = E_pos.mean() - E_neg.mean()
            measure_est = 0.5 * difference

        return -difference, measure_est



class Mine(nn.Module):
    def __init__(self, input_size=2050, hidden_size1=2050, hidden_size2=205, output_size=256):
        super().__init__()

        self.fc1 = nn.Conv2d(2050, 2050, 1)
        self.fc2 = nn.Conv2d(2050, 205, 1)
        self.fc3 = nn.Conv2d(205, 1, 1)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.ma_rate = 0.01
        self.ma_et = 1.


    def forward(self, new_batch, measure='JSD'):
        output = self.relu1(self.fc1(new_batch))
        output = self.relu2(self.fc2(output))
        output = self.fc3(output)

        batch_s = int(len(output)/2)

        if (measure == 'Mine'):
            t = output[0:batch_s]

            # et = self.relu1(self.fc1(new_batch_marginal))
            # et = self.relu2(self.fc2(et))
            et = torch.exp(output[batch_s:])

            mi_lb = torch.mean(t) - torch.log(torch.mean(et))
            self.ma_et = (1-self.ma_rate)*self.ma_et + self.ma_rate*torch.mean(et)

            loss = -(torch.mean(t) - (1/self.ma_et.mean()).detach()*torch.mean(et))
            # loss = -mi_lb

            return loss, mi_lb

        elif (measure == 'JSD'):
            log_2 = math.log(2.)
            E_pos = log_2 - F.softplus(-output[0:batch_s])
            E_neg = F.softplus(-output[batch_s:]) + output[batch_s:] - log_2
            difference = E_pos.mean() - E_neg.mean()
            measure_est = 0.5 * difference

        return -difference, measure_est

class mlp(nn.Module):
    def __init__(self, input_size=512, hidden_size1=100, hidden_size2=10, output_size=2):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)

        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, new_batch):
        output = self.fc1(new_batch)
        output = self.relu1(output)
        output = self.fc2(output)
        output = self.relu2(output)
        output = self.fc3(output)
        output = self.sigmoid(output)
        return output


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # tensors_gather = [torch.ones_like(tensor)
    #     for _ in range(torch.distributed.get_world_size())]
    # torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat([tensor], dim=0)
    return output

