import torch
import torch.nn as nn
import torch.nn.functional as F


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

class ProtoLoss:
    def __init__(self, opt: dict):
        self.loss = opt["settings"]["train"]["loss"]
        self.n_support = opt["settings"]["few_shot"]["train"]["support_shots"]

    def __call__(self, input, target):
        if self.loss == "cosine":
            return self.cosine(input, target, self.n_support)
        elif self.loss == "euclidean":
            return self.euclidean(input, target, self.n_support)
    
    def cosine(self, input, target):
        pass

    def euclidean(self, input, target, n_support=5):

        target_cpu = target.cpu()
        input_cpu = input.cpu()


        def supp_idxs(c):
            return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)

        classes = torch.unique(target_cpu)
        n_classes = len(classes)
        n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

        support_idxs = list(map(supp_idxs, classes))
        prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])

        query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)

        query_samples = input.to('cpu')[query_idxs]
        dists = euclidean_dist(query_samples, prototypes)
 
        log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)
        target_inds = torch.arange(0, n_classes)
        target_inds = target_inds.view(n_classes, 1, 1)
        target_inds = target_inds.expand(n_classes, n_query, 1).long()
        _, y_hat = log_p_y.max(2)
        # loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        log_p_y = log_p_y.view(-1, n_classes)
        target_ = target_inds.reshape(-1).squeeze()

        nll_loss = nn.NLLLoss()
        loss_val = nll_loss(log_p_y, target_)
        
        acc_val = y_hat.eq(target_inds.squeeze(2)).float().mean()
        
        return loss_val, acc_val
