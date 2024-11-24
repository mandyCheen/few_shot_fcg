import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout

class DistanceMetric:
    """Base class for distance metrics"""
    @staticmethod
    def compute(x, y):
        raise NotImplementedError

class EuclideanDistance(DistanceMetric):
    @staticmethod
    def compute(x, y):
        '''
        Compute euclidean distance between two tensors
        x: N x D
        y: M x D
        '''
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        if d != y.size(1):
            raise Exception("Dimension mismatch")

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        return torch.pow(x - y, 2).sum(2)

class CosineDistance(DistanceMetric):
    @staticmethod
    def compute(x, y):
        '''
        Compute cosine distance between two tensors
        x: N x D
        y: M x D
        '''
        # 正規化向量
        x_norm = F.normalize(x, p=2, dim=1)
        y_norm = F.normalize(y, p=2, dim=1)
        
        # 計算余弦相似度
        similarity = torch.mm(x_norm, y_norm.t())
        
        # 轉換為距離（1 - similarity）
        return 1 - similarity

class ManhattanDistance(DistanceMetric):
    @staticmethod
    def compute(x, y):
        '''
        Compute Manhattan (L1) distance between two tensors
        x: N x D
        y: M x D
        '''
        n = x.size(0)
        m = y.size(0)
        d = x.size(1)
        if d != y.size(1):
            raise Exception("Dimension mismatch")

        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)

        return torch.abs(x - y).sum(2)
    
DISTANCE_METRICS = {
    'euclidean': EuclideanDistance,
    'cosine': CosineDistance,
    'manhattan': ManhattanDistance
}
class ProtoLoss:
    """Prototypical Networks Loss with multiple distance metrics"""
    
    def __init__(self, opt: dict):
        self.metric_name = opt["settings"]["train"]["distance"]
        self.n_support = opt["settings"]["few_shot"]["train"]["support_shots"]
        
        if self.metric_name not in DISTANCE_METRICS:
            raise ValueError(f"Unsupported distance metric: {self.metric_name}. "
                           f"Supported metrics are: {list(DISTANCE_METRICS.keys())}")
        
        self.distance_metric = DISTANCE_METRICS[self.metric_name]
        
    def compute_prototypes(self, input, support_idxs):
        """Compute prototype vectors"""
        return torch.stack([input[idx_list].mean(0) for idx_list in support_idxs])
    
    def get_support_query_idxs(self, target):
        """Split data into support and query sets"""
        def supp_idxs(c):
            return target.eq(c).nonzero()[:self.n_support].squeeze(1)
        classes = torch.unique(target)
        support_idxs = list(map(supp_idxs, classes))
        query_idxs = torch.stack(list(map(
            lambda c: target.eq(c).nonzero()[self.n_support:], classes
        ))).view(-1)
        
        return classes, support_idxs, query_idxs
    
    def compute_loss_and_acc(self, dists, n_classes, n_query):
        """
        使用CrossEntropyLoss計算損失和準確率
        
        Args:
            dists: 距離矩陣
            n_classes: 類別數量
            n_query: 每個類別的查詢樣本數
            
        Returns:
            tuple: (loss_value, accuracy_value)
        """
        # 注意：不需要手動應用log_softmax，因為CrossEntropyLoss內部會處理
        # 將距離轉換為logits (負距離作為logits)
        logits = -dists.view(n_classes, n_query, -1)
        
        # 準備目標索引
        target_inds = torch.arange(0, n_classes)
        target_inds = target_inds.view(n_classes, 1, 1)
        target_inds = target_inds.expand(n_classes, n_query, 1).long()
        
        # 計算預測類別
        # 使用softmax而不是log_softmax來獲得概率
        probs = F.softmax(logits, dim=2)
        _, y_hat = probs.max(2)
        
        # 重塑logits和目標
        logits = logits.view(-1, n_classes)  # (n_classes * n_query) x n_classes
        target_ = target_inds.reshape(-1)    # (n_classes * n_query)
        
        # 使用CrossEntropyLoss (內部包含softmax)
        criterion = nn.CrossEntropyLoss()
        loss_val = criterion(logits, target_)
        
        # 計算準確率
        acc_val = y_hat.eq(target_inds.squeeze(2)).float().mean()
        
        return loss_val, acc_val

    def __call__(self, input, target):
        """
        計算Prototypical Networks的損失值和準確率
        
        Args:
            input: 模型輸出的特徵向量
            target: 目標類別標籤
            
        Returns:
            tuple: (loss_value, accuracy_value)
        """
        # 移到CPU進行計算（可根據需要修改）
        target_cpu = target.cpu()
        input_cpu = input.cpu()
        # 獲取support和query的索引
        classes, support_idxs, query_idxs = self.get_support_query_idxs(target_cpu)
        n_classes = len(classes)
        n_query = target_cpu.eq(classes[0].item()).sum().item() - self.n_support
        
        # 計算原型
        prototypes = self.compute_prototypes(input_cpu, support_idxs)
        
        # 獲取query樣本
        query_samples = input_cpu[query_idxs]
        
        # 計算距離
        dists = self.distance_metric.compute(query_samples, prototypes)
        
        # 計算損失和準確率
        return self.compute_loss_and_acc(dists, n_classes, n_query)

    def get_metric_name(self):
        """返回當前使用的距離度量方法名稱"""
        return self.metric_name


class DCELoss:
    
    def __init__(self, opt: dict):
        
        super(DCELoss, self).__init__()