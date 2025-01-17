# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, base_encoder2, dim=128, K=11408, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 2048)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        
        # self.encoder_q = base_encoder(num_classes=dim)
        # self.encoder_k = base_encoder2(num_classes=dim)
        self.encoder_q = base_encoder
        self.encoder_k = base_encoder2

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

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
                ll = list(self.encoder_q.children())
                q_l1 = nn.Sequential(*ll[:5])(im_q)
                q_l2 = ll[5](q_l1)
                
                
                q_l3 = nn.Sequential(*ll[6:7])(q_l2)
                q_l4 = ll[7](q_l3)
                q = ll[8](q_l4)
                q = ll[9](q.view(q.size(0), -1))
                
                #q = self.encoder_q(im_q)  # queries: NxC  # [16, 512]
            return q, q_l1, q_l2, q_l3, q_l4

        # compute query features
        #ll = list(self.encoder_q.children())
        #q_l1 = nn.Sequential(*ll[:5])(im_q)
        #q_l2 = ll[5](q_l1)
        
        #q_l3 = nn.Sequential(*ll[6:7])(q_l2)
        #q_l4 = ll[7](q_l3)
        #q = ll[8](q_l4)
        #q = ll[9](q.view(q.size(0), -1))

        q = self.encoder_q(im_q)  # queries: NxC
        backbone_feature = q
        q = nn.functional.normalize(q, dim=1)         # [16, 512]


        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels, backbone_feature


def contrastive_gradient_penalty(network, input, penalty_amount=1.):
    """Contrastive gradient penalty.
    This is essentially the loss introduced by Mescheder et al 2018.
    Args:
        network: Network to apply penalty through.
        input: Input or list of inputs for network.
        penalty_amount: Amount of penalty.
    Returns:
        torch.Tensor: gradient penalty loss.
    """
    def _get_gradient(inp, output):
        gradient = autograd.grad(outputs=output, inputs=inp,
                                 grad_outputs=torch.ones_like(output),
                                 create_graph=True, retain_graph=True,
                                 only_inputs=True, allow_unused=True)[0]
        return gradient

    if not isinstance(input, (list, tuple)):
        input = [input]

    input = [inp.detach() for inp in input]
    input = [inp.requires_grad_() for inp in input]

    with torch.set_grad_enabled(True):
        output = network(*input)[-1]
    gradient = _get_gradient(input, output)
    gradient = gradient.view(gradient.size()[0], -1)
    penalty = (gradient ** 2).sum(1).mean()

    return penalty * penalty_amount


class Mineq(nn.Module):
    def __init__(self, input_size=514, hidden_size1=514, hidden_size2=100, output_size=256):
        super().__init__()

        self.fc1_q = nn.Linear(515, 100, 1)
        self.fc2_q = nn.Linear(100, 20, 1)
        self.fc3_q = nn.Linear(20, 1, 1)

        self.relu1_q = nn.ReLU(inplace=True)
        self.relu2_q = nn.ReLU(inplace=True)

        self.ma_rate = 0.01
        self.ma_et = 1.

    def forward(self, new_batch3, measure='JSD'):
        output = self.relu1_q(self.fc1_q(new_batch3))
        output = self.relu2_q(self.fc2_q(output))
        output = self.fc3_q(output)

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
    def __init__(self, input_size=512, hidden_size1=100, hidden_size2=10, output_size=3):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.sigmoid = nn.Softmax(dim=1)

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
    
    output = torch.cat([tensor], dim=0)
    return output
