import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from models import GraphRelationNetwork
from torch_geometric.nn import global_add_pool

class DistanceMetric:
    """Base class for distance metrics"""
    @staticmethod
    def compute(x, y):
        raise NotImplementedError

class LossFunction:
    """Base class for loss functions"""
    @staticmethod
    def compute(dists, n_classes, n_query):
        raise NotImplementedError
    @staticmethod
    def compute(dists, n_classes, n_query, negative_distance=False):
        raise NotImplementedError
    @staticmethod
    def compute(matrix, dim):
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

class CosineSimilarity(DistanceMetric):
    @staticmethod
    def compute(x, y):
        '''
        Compute cosine similarity between two tensors
        x: N x D
        y: M x D
        '''
        # 正規化向量
        x_norm = F.normalize(x, p=2, dim=1)
        y_norm = F.normalize(y, p=2, dim=1)
        
        # 計算余弦相似度
        similarity = torch.mm(x_norm, y_norm.t())
        
        return similarity
    
class CrossEntropyLoss(LossFunction):
    @staticmethod
    def compute(dists, n_classes, n_query, negative_distance=False):
        """
        使用CrossEntropyLoss計算損失和準確率
        
        Args:
            dists: 距離矩陣
            n_classes: 類別數量
            n_query: 每個類別的查詢樣本數
            
        Returns:
            tuple: (loss_value, accuracy_value)
        """
        # 將距離轉換為logits (負距離作為logits)
        if negative_distance:
            logits = -dists.view(n_classes, n_query, -1)
        else:
            logits = dists.view(n_classes, n_query, -1)
        
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
    
class EntropyLoss(LossFunction):
    """熵損失計算類"""

    @staticmethod
    def compute(matrix, dim):
        """
        計算熵損失
        
        Args:
            matrix: 輸入矩陣
            dim: 計算熵的維度
            
        Returns:
            torch.Tensor: 熵損失值
        """
        # 計算每行的softmax
        probs = F.softmax(matrix, dim=dim)
        log_probs = F.log_softmax(matrix, dim=dim)
        
        entropy_loss = -torch.sum(probs * log_probs, dim=dim).mean()

        return entropy_loss
        

DISTANCE_METRICS = {
    'euclidean': EuclideanDistance,
    'cosine_distance': CosineDistance,
    'cosine_similarity': CosineSimilarity,
    'manhattan': ManhattanDistance
}

LOSS_FUNCTIONS = {
    'CrossEntropyLoss': CrossEntropyLoss
}

class Loss:
    def __init__(self, n_support):
        self.n_support = n_support
        self.openset_auroc = 0.0
        
    def compute_prototypes(self, input, support_idxs):
        """Compute prototype vectors"""
        return torch.stack([input[idx_list].mean(0) for idx_list in support_idxs])
    
    def get_nn_points(self, input, support_idxs):
        """Get nearest neighbor points"""
        return torch.cat([input[idx_list] for idx_list in support_idxs])
    
    def get_support_query_idxs(self, target, openset=False, cls_openset=None):
        """Split data into support and query sets"""
        def supp_idxs(c):
            return target.eq(c).nonzero()[:self.n_support].squeeze(1)
        classes = torch.unique(target)
        cls_num = len(classes)
        if openset:
            assert cls_openset is not None, "cls_support must be provided for openset"
            cls_support = cls_num - cls_openset
            # Randomly select classes for support
            random_classes = classes[torch.randperm(len(classes))]
            support_idxs = list(map(supp_idxs, random_classes[:cls_support]))
            query_idxs = torch.stack(list(map(
                lambda c: target.eq(c).nonzero()[self.n_support:], random_classes[:cls_support]
            ))).view(-1)
            openset_idxs = torch.stack(list(map(
                lambda c: target.eq(c).nonzero()[:], random_classes[cls_support:]
            ))).view(-1)
            return random_classes, support_idxs, query_idxs, openset_idxs
        else:
            support_idxs = list(map(supp_idxs, classes))
            query_idxs = torch.stack(list(map(
                lambda c: target.eq(c).nonzero()[self.n_support:], classes
            ))).view(-1)
        
            return classes, support_idxs, query_idxs

    def reshape_random_data(self, data, target, n_ways, n_shot, n_queries):
        """
        Reshape randomly organized data into (n_ways, n_shot + n_queries, n_features) format
        Returns:
            Reshaped data of shape (n_ways, n_shot + n_queries, n_features)
        """
        # Set n_support for the get_support_query_idxs function
        n_support = n_shot
        
        # Get classes, support indices, and query indices
        classes, support_idxs, query_idxs = self.get_support_query_idxs(target)
        
        # Get feature dimension
        n_features = data.shape[1]
        
        # Initialize the result tensor
        result = torch.zeros(n_ways, n_shot + n_queries, n_features)
        
        # Fill in support samples
        for i, c in enumerate(classes):
            # Get support samples for this class
            class_support = data[support_idxs[i]]
            # Place them in the result tensor
            result[i, :n_shot] = class_support
        
        # Fill in query samples
        query_data = data[query_idxs]
        # Reshape query data to (n_ways, n_queries, n_features)
        query_data = query_data.reshape(n_ways, n_queries, n_features)
        # Place them after the support samples
        result[:, n_shot:] = query_data
        
        return result

    def get_avg_distance_between_support(self, input, target):
        """
        Compute average distance between support samples of the same class
        """

        class_avg_distances = []

        _, support_idxs, query_idxs = self.get_support_query_idxs(target)

        for class_idx, idx_list in enumerate(support_idxs):
            # Get support samples for this class
            class_support_samples = input[idx_list]
                
            # Compute pairwise distances within this class
            class_dists = self.distance_metric.compute(class_support_samples, class_support_samples)
            
            # Create a mask to exclude self-distances (diagonal elements)
            mask = ~torch.eye(len(idx_list), dtype=torch.bool, device=class_dists.device)
            
            # Calculate mean of non-diagonal elements (actual distances between different samples)
            class_avg_dist = class_dists[mask].mean()
            print(f"Class {class_idx}: Average distance = {class_avg_dist.item()}")
            class_avg_distances.append(class_avg_dist)
        
        # Convert to tensor for easier computation
        distances_tensor = torch.stack(class_avg_distances)
        
        # Calculate Q1 (25th percentile) and Q3 (75th percentile)
        q1, q3 = torch.quantile(distances_tensor, torch.tensor([0.25, 0.75], device=distances_tensor.device))
        
        # Calculate IQR (Interquartile Range)
        iqr = q3 - q1
        
        # Define bounds for outlier detection (typically 1.5 * IQR)
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        print(f"Lower bound: {lower_bound.item()}, Upper bound: {upper_bound.item()}")

        # Filter out outliers
        valid_distances = distances_tensor[(distances_tensor >= lower_bound) & (distances_tensor <= upper_bound)]
        
        # Calculate mean of non-outlier distances
        if len(valid_distances) > 0:
            avg_distance = torch.mean(valid_distances)
        else:
            # If all are outliers (rare case), just use median
            avg_distance = torch.median(distances_tensor)

        return avg_distance
    def get_query_distance_between_support(self, input, target):
        """
        Compute distance between support samples and query samples
        """

        _, support_idxs, query_idxs = self.get_support_query_idxs(target)
        # Get the support samples
        support_samples = torch.cat([input[idx_list] for idx_list in support_idxs])
        # Get the query samples
        query_samples = input[query_idxs]
        # Compute the pairwise distances
        dists = self.distance_metric.compute(query_samples, support_samples)
        return dists

    def get_openset_distance_between_support(self, input, target, openset):
        """
        Compute distance between support samples and open set samples
        """
        _, support_idxs, query_idxs = self.get_support_query_idxs(target)
        # Get the support samples
        support_samples = torch.cat([input[idx_list] for idx_list in support_idxs])
        # Compute the pairwise distances
        dists = self.distance_metric.compute(openset, support_samples)
        return dists

    def roc_area_calc(self, dist, closed, descending=True):
        """
        計算ROC曲線下面積
        
        參數:
            dist (torch.Tensor): 分數或距離，用於識別開放集樣本
            closed (torch.Tensor): 標記樣本是closed-set(1)還是open-set(-1)的二元標籤
            descending (bool): 排序順序，True表示降序排列（高分表示closed-set）
        
        返回:
            float: ROC曲線下面積
        """
        _, p = dist.sort(descending=descending)
        closed_p = closed[p]
        
        # 計算closed-set和open-set樣本的總數
        total_closed = (closed == 1).sum().item()
        total_open = (closed == -1).sum().item()
        
        if total_closed == 0 or total_open == 0:
            return 0.0
            
        height = 0.0
        width = 0.0
        area = 0.0
        pre = 0  # (0: width; 1: height)
        
        for i in range(len(closed_p)):
            if closed_p[i] == -1:  # open-set樣本
                if pre == 0:
                    area += height * width
                    width = 0.0
                    height += 1.0
                    pre = 1
                else:
                    height += 1.0
            else:  # closed-set樣本
                pre = 0
                width += 1.0
                
        if pre == 0:
            area += height * width
            
        # 除以總高度和總寬度進行標準化
        area = area / total_open / total_closed
        return area

class ProtoLoss(Loss):
    """Prototypical Networks Loss with multiple distance metrics"""
    
    def __init__(self, opt: dict):
        self.metric_name = opt["settings"]["train"]["distance"]
        self.n_support = opt["settings"]["few_shot"]["train"]["support_shots"]
        self.loss_fn_name = opt["settings"]["train"]["loss"]

        self.enable_openset = opt.get("settings", {}).get("openset", {}).get("use", False)
        self.openset_m_samples = opt.get("settings", {}).get("openset", {}).get("test", {}).get("m_samples", 0)
        self.cls_training_openset = opt.get("settings", {}).get("openset", {}).get("train", {}).get("class_per_iter", 0)
        self.openset_loss_scale = opt.get("settings", {}).get("openset", {}).get("train", {}).get("loss_weight", 0.5)
        
        if self.metric_name not in DISTANCE_METRICS:
            raise ValueError(f"Unsupported distance metric: {self.metric_name}. "
                           f"Supported metrics are: {list(DISTANCE_METRICS.keys())}")

        if self.loss_fn_name not in LOSS_FUNCTIONS:
            raise ValueError(f"Unsupported loss function: {self.loss_fn_name}. "
                           f"Supported loss functions are: {list(LOSS_FUNCTIONS.keys())}")
        
        self.distance_metric = DISTANCE_METRICS[self.metric_name]
        self.loss_fn = LOSS_FUNCTIONS[self.loss_fn_name]
        self.opensetLoss = EntropyLoss()
        
        super().__init__(self.n_support)

    def __call__(self, input, target, opensetTesting=False):
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

        if opensetTesting: # openset testing
            num = len(target) - self.openset_m_samples
            closed_target = target[:num]
            classes, support_idxs, query_idxs = self.get_support_query_idxs(closed_target)
            n_classes = len(torch.unique(closed_target))
            num_open_samples = self.openset_m_samples
            openset_idxs = torch.arange(len(target))[num:]
            n_query = target_cpu.eq(classes[0].item()).sum().item() - self.n_support
        elif self.enable_openset: # openset training
            # get support and query indices
            classes, support_idxs, query_idxs, openset_idxs = self.get_support_query_idxs(target, openset=True, cls_openset=self.cls_training_openset)
            n_classes = len(torch.unique(target)) - self.cls_training_openset
            num_open_samples = len(openset_idxs)
            n_query = target_cpu.eq(classes[0].item()).sum().item() - self.n_support
        else: # closed set training & testing
            classes, support_idxs, query_idxs = self.get_support_query_idxs(target_cpu)
            n_classes = len(classes)
            n_query = target_cpu.eq(classes[0].item()).sum().item() - self.n_support
            num_open_samples = 0
        
        # 計算原型
        prototypes = self.compute_prototypes(input_cpu, support_idxs)
        
        # 獲取query樣本
        query_samples = input_cpu[query_idxs]
        # 計算距離
        dists = self.distance_metric.compute(query_samples, prototypes)

        if self.metric_name == 'cosine_similarity':
            # 計算損失和準確率
            loss, acc = self.loss_fn.compute(dists, n_classes, n_query)
        else:
            # 計算損失和準確率
            loss, acc = self.loss_fn.compute(dists, n_classes, n_query, negative_distance=True)
        # openset
        if self.enable_openset and num_open_samples > 0:
            openset_samples = input_cpu[openset_idxs]
            openset_dists = self.distance_metric.compute(openset_samples, prototypes)
            openset_logits = -openset_dists.view(n_classes, num_open_samples, -1)
            query_logits = -dists.view(n_classes, n_query, -1)

            if self.metric_name == 'cosine_similarity':
                openset_logits = -openset_logits
                query_logits = -query_logits

            entropy_loss = self.opensetLoss.compute(openset_logits, dim=2)
            open_loss = -entropy_loss

            loss = loss + self.openset_loss_scale * open_loss

            query_max_probs = F.softmax(query_logits, dim=2).max(dim=2)[0]
            openset_max_probs = F.softmax(openset_logits, dim=2).max(dim=2)[0]

            all_scores = torch.cat([query_max_probs, openset_max_probs], dim=0)
            closed_labels = torch.cat([
                torch.ones(query_max_probs.size(0), device=self.device),
                -torch.ones(openset_max_probs.size(0), device=self.device)
            ], dim=0)

            self.openset_auroc = self.roc_area_calc(all_scores, closed_labels, descending=False)
             
        return loss, acc

    def get_metric_name(self):
        """返回當前使用的距離度量方法名稱"""
        return self.metric_name

    def get_loss_fn(self):
        """返回當前使用的損失函數名稱"""
        return self.loss_fn_name

class NnLoss(Loss):
    def __init__(self, opt: dict):
        self.metric_name = opt["settings"]["train"]["distance"]
        self.n_support = opt["settings"]["few_shot"]["train"]["support_shots"]
        self.loss_fn_name = opt["settings"]["train"]["loss"]

        self.enable_openset = opt.get("settings", {}).get("openset", {}).get("use", False)
        self.openset_m_samples = opt.get("settings", {}).get("openset", {}).get("test", {}).get("m_samples", 0)
        self.cls_training_openset = opt.get("settings", {}).get("openset", {}).get("train", {}).get("class_per_iter", 0)
        self.openset_loss_scale = opt.get("settings", {}).get("openset", {}).get("train", {}).get("loss_weight", 0.5)
            
        if self.metric_name not in DISTANCE_METRICS:
            raise ValueError(f"Unsupported distance metric: {self.metric_name}. "
                        f"Supported metrics are: {list(DISTANCE_METRICS.keys())}")
        if self.loss_fn_name not in LOSS_FUNCTIONS:
            raise ValueError(f"Unsupported loss function: {self.loss_fn_name}. "
                        f"Supported loss functions are: {list(LOSS_FUNCTIONS.keys())}")
        
        self.distance_metric = DISTANCE_METRICS[self.metric_name]
        self.loss_fn = LOSS_FUNCTIONS[self.loss_fn_name]
        self.opensetLoss = EntropyLoss()
        
        super().__init__(self.n_support)
    
    def __call__(self, input, target, opensetTesting=False):
        target_cpu = target.cpu()
        input_cpu = input.cpu()

        if opensetTesting: # openset testing
            num = len(target) - self.openset_m_samples
            closed_target = target[:num]
            classes, support_idxs, query_idxs = self.get_support_query_idxs(closed_target)
            n_classes = len(torch.unique(closed_target))
            num_open_samples = self.openset_m_samples
            openset_idxs = torch.arange(len(target))[num:]
            n_query = target_cpu.eq(classes[0].item()).sum().item() - self.n_support
        elif self.enable_openset: # openset training
            # get support and query indices
            classes, support_idxs, query_idxs, openset_idxs = self.get_support_query_idxs(target, openset=True, cls_openset=self.cls_training_openset)
            n_classes = len(torch.unique(target)) - self.cls_training_openset
            num_open_samples = len(openset_idxs)
            n_query = target_cpu.eq(classes[0].item()).sum().item() - self.n_support
        else: # closed set training & testing
            classes, support_idxs, query_idxs = self.get_support_query_idxs(target_cpu)
            n_classes = len(classes)
            n_query = target_cpu.eq(classes[0].item()).sum().item() - self.n_support
            num_open_samples = 0
        
        nn_points = self.get_nn_points(input_cpu, support_idxs)
        
        query_samples = input_cpu[query_idxs]
        
        dists = self.distance_metric.compute(query_samples, nn_points)

        # find the nearest neighbor for each query point
        dists_by_class = dists.view(n_classes*n_query, n_classes, self.n_support)
        min_dists, _ = dists_by_class.min(dim=2)
        
        if self.metric_name == 'cosine_similarity':
            loss, acc = self.loss_fn.compute(min_dists, n_classes, n_query)
        else:
            loss, acc = self.loss_fn.compute(min_dists, n_classes, n_query, negative_distance=True)

        # openset
        if self.enable_openset and num_open_samples > 0:
            openset_samples = input_cpu[openset_idxs]
            openset_dists = self.distance_metric.compute(openset_samples, nn_points)

            # find the nearest neighbor for each query point
            openset_dists_by_class = openset_dists.view(num_open_samples, n_classes, self.n_support)
            openset_min_dists, _ = openset_dists_by_class.min(dim=2)

            openset_logits = -openset_min_dists.view(n_classes, num_open_samples, -1)
            query_logits = -min_dists.view(n_classes, n_query, -1)

            if self.metric_name == 'cosine_similarity':
                openset_logits = -openset_logits
                query_logits = -query_logits
            
            entropy_loss = self.opensetLoss.compute(openset_logits, dim=2)
            open_loss = -entropy_loss

            loss = loss + self.openset_loss_scale * open_loss

            query_max_probs = F.softmax(query_logits, dim=2).max(dim=2)[0]
            openset_max_probs = F.softmax(openset_logits, dim=2).max(dim=2)[0]

            all_scores = torch.cat([query_max_probs, openset_max_probs], dim=0)
            closed_labels = torch.cat([
                torch.ones(query_max_probs.size(0), device=self.device),
                -torch.ones(openset_max_probs.size(0), device=self.device)
            ], dim=0)

            self.openset_auroc = self.roc_area_calc(all_scores, closed_labels, descending=False)
             
        return loss, acc 
    
    def get_metric_name(self):
        return self.metric_name
    
    def get_loss_fn(self):
        return self.loss_fn_name
    
class SoftNnLoss(Loss):
    def __init__(self, opt: dict):
        self.metric_name = opt["settings"]["train"]["distance"]
        self.n_support = opt["settings"]["few_shot"]["train"]["support_shots"]
        self.loss_fn_name = opt["settings"]["train"]["loss"]
        
        self.enable_openset = opt.get("settings", {}).get("openset", {}).get("use", False)
        self.openset_m_samples = opt.get("settings", {}).get("openset", {}).get("test", {}).get("m_samples", 0)
        self.cls_training_openset = opt.get("settings", {}).get("openset", {}).get("train", {}).get("class_per_iter", 0)
        self.openset_loss_scale = opt.get("settings", {}).get("openset", {}).get("train", {}).get("loss_weight", 0.5)
            

        if self.metric_name not in DISTANCE_METRICS:
            raise ValueError(f"Unsupported distance metric: {self.metric_name}. "
                        f"Supported metrics are: {list(DISTANCE_METRICS.keys())}")
        if self.loss_fn_name not in LOSS_FUNCTIONS:
            raise ValueError(f"Unsupported loss function: {self.loss_fn_name}. "
                        f"Supported loss functions are: {list(LOSS_FUNCTIONS.keys())}")
        
        self.distance_metric = DISTANCE_METRICS[self.metric_name]
        self.loss_fn = LOSS_FUNCTIONS[self.loss_fn_name]
        self.opensetLoss = EntropyLoss()
        
        super().__init__(self.n_support)
    def __call__(self, input, target):
        """
        Compute soft nearest neighbor loss by using weighted distances to all support samples
        
        Args:
            input: Model output features
            target: Target labels
            
        Returns:
            tuple: (loss_value, accuracy_value)
        """
        target_cpu = target.cpu()
        input_cpu = input.cpu()       

        if opensetTesting: # openset testing
            num = len(target) - self.openset_m_samples
            closed_target = target[:num]
            classes, support_idxs, query_idxs = self.get_support_query_idxs(closed_target)
            n_classes = len(torch.unique(closed_target))
            num_open_samples = self.openset_m_samples
            openset_idxs = torch.arange(len(target))[num:]
            n_query = target_cpu.eq(classes[0].item()).sum().item() - self.n_support
        elif self.enable_openset: # openset training
            # get support and query indices
            classes, support_idxs, query_idxs, openset_idxs = self.get_support_query_idxs(target, openset=True, cls_openset=self.cls_training_openset)
            n_classes = len(torch.unique(target)) - self.cls_training_openset
            num_open_samples = len(openset_idxs)
            n_query = target_cpu.eq(classes[0].item()).sum().item() - self.n_support
        else: # closed set training & testing
            classes, support_idxs, query_idxs = self.get_support_query_idxs(target_cpu)
            n_classes = len(classes)
            n_query = target_cpu.eq(classes[0].item()).sum().item() - self.n_support
            num_open_samples = 0
         
        
        # Get support points 
        nn_points = self.get_nn_points(input_cpu, support_idxs)
        
        # Get query samples
        query_samples = input_cpu[query_idxs]
        
        # Compute distances between queries and all support samples
        dists = self.distance_metric.compute(query_samples, nn_points)
        
        # Reshape distances to group by class
        dists_by_class = dists.view(n_classes*n_query, n_classes, self.n_support)
        
        # For each query point, compute soft weights for all support samples
        if self.metric_name == 'cosine_similarity':
            weights = F.softmax(dists_by_class, dim=-1)
        else:
            weights = F.softmax(-dists_by_class, dim=-1) 

        weighted_dists = (weights * dists_by_class).sum(dim=2)
            
        # Compute loss and accuracy using weighted distances
        if self.metric_name == 'cosine_similarity':
            loss, acc = self.loss_fn.compute(weighted_dists, n_classes, n_query)
        else:
            loss, acc = self.loss_fn.compute(weighted_dists, n_classes, n_query, negative_distance=True)

        # openset
        if self.enable_openset and num_open_samples > 0:
            openset_samples = input_cpu[openset_idxs]
            openset_dists = self.distance_metric.compute(openset_samples, nn_points)

            # find the nearest neighbor for each query point
            openset_dists_by_class = openset_dists.view(num_open_samples, n_classes, self.n_support)
            if self.metric_name == 'cosine_similarity':
                openset_weights = F.softmax(openset_dists_by_class, dim=-1)
            else:
                openset_weights = F.softmax(-openset_dists_by_class, dim=-1)

            openset_weighted_dists = (openset_weights * openset_dists_by_class).sum(dim=2)

            openset_logits = -openset_weighted_dists.view(n_classes, num_open_samples, -1)
            query_logits = -weighted_dists.view(n_classes, n_query, -1)

            if self.metric_name == 'cosine_similarity':
                openset_logits = -openset_logits
                query_logits = -query_logits
            
            entropy_loss = self.opensetLoss.compute(openset_logits, dim=2)
            open_loss = -entropy_loss

            loss = loss + self.openset_loss_scale * open_loss

            query_max_probs = F.softmax(query_logits, dim=2).max(dim=2)[0]
            openset_max_probs = F.softmax(openset_logits, dim=2).max(dim=2)[0]

            all_scores = torch.cat([query_max_probs, openset_max_probs], dim=0)
            closed_labels = torch.cat([
                torch.ones(query_max_probs.size(0), device=self.device),
                -torch.ones(openset_max_probs.size(0), device=self.device)
            ], dim=0)

            self.openset_auroc = self.roc_area_calc(all_scores, closed_labels, descending=False)
             
        return loss, acc 

class ClassifierLoss(Loss):
    def __init__(self, opt: dict):
        self.n_support = opt["settings"]["few_shot"]["train"]["support_shots"]
        self.loss_fn = opt["settings"]["train"]["loss"]

        if self.loss_fn == "CrossEntropyLoss":
            self.criterion = nn.CrossEntropyLoss()
        elif self.loss_fn == "MSELoss":
            self.criterion = nn.MSELoss()
        elif self.loss_fn == "NLLLoss":
            self.criterion = nn.NLLLoss()
        
        super().__init__(self.n_support)
        
    def __call__(self, input, target):
        target_cpu = target.cpu()
        input_cpu = input.cpu()
        
        classes, support_idxs, query_idxs = self.get_support_query_idxs(target_cpu)
        n_classes = len(classes)
        n_query = target_cpu.eq(classes[0].item()).sum().item() - self.n_support
        
        # get support values
        support_samples = input_cpu[support_idxs]
        
        loss = self.criterion(support_samples, target_cpu[support_idxs])
        acc = torch.sum(torch.argmax(support_samples, dim=1) == target_cpu[support_idxs]).item() / len(support_idxs)
        
        return loss, acc

from torch_geometric.data import Batch
class LabelPropagation(nn.Module, Loss):
    """Label Propagation"""
    def __init__(self, opt: dict, encoder):
        super(LabelPropagation, self).__init__()
        self.opt = opt
        self.args = opt['settings']['few_shot']['parameters']
        self.device = opt["settings"]["train"]["device"]
        self.n_support = opt["settings"]["few_shot"]["train"]["support_shots"]
        self.enable_openset = opt.get("settings", {}).get("openset", {}).get("use", False)
        self.openset_m_samples = opt.get("settings", {}).get("openset", {}).get("test", {}).get("m_samples", 0)
        self.cls_training_openset = opt.get("settings", {}).get("openset", {}).get("train", {}).get("class_per_iter", 0)
        self.openset_loss_scale = opt.get("settings", {}).get("openset", {}).get("train", {}).get("loss_weight", 0.5)
        self.encoder = encoder
        self.relation = GraphRelationNetwork(self.args["dim_in"], self.args["dim_hidden"], self.args["dim_out"], self.args["relation_layer"])

        self.metric_name = opt["settings"]["train"]["distance"]
        self.distance_metric = DISTANCE_METRICS[self.metric_name]
        self.opensetLoss = EntropyLoss()
        
        if opt['settings']['few_shot']['parameters']['rn'] == 300:   # learned sigma, fixed alpha
            self.alpha = torch.tensor([opt['settings']['few_shot']['parameters']['alpha']], requires_grad=False).to(self.device)
        elif opt['settings']['few_shot']['parameters']['rn'] == 30:    # learned sigma, learned alpha
            self.alpha = nn.Parameter(torch.tensor([opt['settings']['few_shot']['parameters']['alpha']]).to(self.device), requires_grad=True)

    def forward(self, data, opensetTesting=False):
        """
            inputs are preprocessed
            support:    (N_way*N_shot)x3x84x84
            query:      (N_way*N_query)x3x84x84
            s_labels:   (N_way*N_shot)xN_way, one-hot
            q_labels:   (N_way*N_query)xN_way, one-hot
        """
        # init
        import torch.nn.functional as F
        eps = np.finfo(float).eps

        input = self.encoder(data.x, data.edge_index)
        target = data.y
        edge_index = data.edge_index
        batch = data.batch

        if opensetTesting: # openset testing
            num = len(target) - self.openset_m_samples
            closed_target = target[:num]
            _, support_idxs, query_idxs = self.get_support_query_idxs(closed_target)
            num_support_classes = len(torch.unique(closed_target))
            num_open_samples = self.openset_m_samples
            openset_idxs = torch.arange(len(target))[num:]
        elif self.enable_openset: # openset training
            # get support and query indices
            _, support_idxs, query_idxs, openset_idxs = self.get_support_query_idxs(target, openset=True, cls_openset=self.cls_training_openset)
            num_support_classes = len(torch.unique(target)) - self.cls_training_openset
            num_open_samples = len(openset_idxs)
        else: # closed set training & testing
            _, support_idxs, query_idxs = self.get_support_query_idxs(target)
            num_support_classes = len(torch.unique(target))
            num_open_samples = 0
        num_queries = 20 - self.n_support

        s_labels_ori = torch.cat([target[idx_list] for idx_list in support_idxs])
        q_labels_ori = target[query_idxs]
        unique_labels = torch.unique(torch.cat([s_labels_ori, q_labels_ori]))
        label_map = {label.item(): idx for idx, label in enumerate(unique_labels)}
        s_labels_mapped = torch.tensor([label_map[label.item()] for label in s_labels_ori]).to(self.device)
        q_labels_mapped = torch.tensor([label_map[label.item()] for label in q_labels_ori]).to(self.device)
        # generate one-hot labels
        s_labels = F.one_hot(s_labels_mapped, num_support_classes)
        q_labels = F.one_hot(q_labels_mapped, num_support_classes)

        pool_input = global_add_pool(input, batch)

        # Step1: Embedding
        N = pool_input.shape[0]

        # Step2: Graph Construction
        ## sigmma
        if self.args['rn'] in [30,300]:
            self.relation.to(self.device)
            #self.sigma = self.relation(input, edge_index, batch)
            self.sigma = F.softplus(self.relation(input, edge_index, batch))
            # sigma_pooled = scatter_mean(self.sigma, batch, dim=0)
            # self.sigma = torch.clamp(self.sigma, min=0.1, max=10.0) -> No need to clamp

            self.sigma_support = torch.cat([self.sigma[idx_list] for idx_list in support_idxs])
            self.sigma_query = self.sigma[query_idxs]

            if self.enable_openset:
                self.sigma_openset = self.sigma[openset_idxs]
                self.sigma = torch.cat([self.sigma_support, self.sigma_query, self.sigma_openset], 0)
            else:
                self.sigma = torch.cat([ self.sigma_support, self.sigma_query], 0)

            pool_input_support = torch.cat([pool_input[idx_list] for idx_list in support_idxs])
            pool_input_query = pool_input[query_idxs]

            if self.enable_openset:
                pool_input_openset = pool_input[openset_idxs]
                pool_input = torch.cat([pool_input_support, pool_input_query, pool_input_openset], 0)
            else:
                pool_input = torch.cat([ pool_input_support, pool_input_query], 0)

            pool_input = pool_input / (self.sigma + eps)
            emb1 = torch.unsqueeze(pool_input, 1)
            emb2 = torch.unsqueeze(pool_input, 0)
            W = ((emb1 - emb2) ** 2).mean(2)
            W = torch.exp(-W / 2)

        ## keep top-k values
        if self.args['k'] > 0:
            topk, indices = torch.topk(W, self.args['k'])
            mask = torch.zeros_like(W)
            mask = mask.scatter(1, indices, 1)
            mask = ((mask + torch.t(mask)) > 0).type(torch.float32)
            W = W * mask
        # 正規化
        D = W.sum(0)
        D_sqrt_inv = torch.sqrt(1.0 / (D + eps))
        D1 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, N)
        D2 = torch.unsqueeze(D_sqrt_inv, 0).repeat(N, 1)
        S = D1 * W * D2

        # Step3: Label Propagation, F = (I-\alpha S)^{-1}Y
        ys = s_labels
        yu = torch.zeros(num_support_classes * num_queries, num_support_classes).to(self.device)

        if self.enable_openset and num_open_samples > 0:
            yo = torch.zeros(num_open_samples, num_support_classes).to(self.device)
            y = torch.cat((ys, yu, yo), 0)
        else:
            y = torch.cat((ys, yu), 0)
        
        F_ = torch.matmul(torch.inverse(torch.eye(N).to(self.device) - self.alpha * S + eps), y)

        # Fq = F_[num_support_classes * self.n_support:, :]
        Fq = F_[num_support_classes * self.n_support : num_support_classes * self.n_support + num_support_classes * num_queries, :]

        if self.enable_openset and num_open_samples > 0:
            Fo = F_[num_support_classes * self.n_support + num_support_classes * num_queries:, :]

        # Step4: Cross-Entropy Loss
        ce = nn.CrossEntropyLoss().to(self.device)
        ## both support and query loss
        gt = torch.argmax(torch.cat((s_labels, q_labels), 0), 1)
        F_sq = F_[:num_support_classes * self.n_support + num_support_classes * num_queries, :]

        loss = ce(F_sq, gt)
        ## acc
        predq = torch.argmax(Fq, 1)
        gtq = torch.argmax(q_labels, 1)
        correct = (predq == gtq).sum()

        total = num_queries * num_support_classes
        acc = 1.0 * correct.float() / float(total)

        if self.enable_openset and num_open_samples > 0:
            # calculate open set loss
            entropy_loss = self.opensetLoss.compute(Fo, dim=1)
            # prob_open = F.softmax(Fo, dim=1)
            # log_prob_open = F.log_softmax(Fo, dim=1)
            # entropy_loss = -(prob_open * log_prob_open).sum(dim=1).mean()

            open_loss = -entropy_loss
            loss = loss + self.openset_loss_scale * open_loss

            # # claculate open set auroc
            # temp_log = "./temp_log.txt"
            # with open(temp_log, "a") as f:
            #     f.write("Fq:\n")
            #     f.write(f"{Fq}\n")
            #     f.write("Fo:\n")
            #     f.write(f"{Fo}\n")

            query_max_probs = F.softmax(Fq, dim=1).max(dim=1)[0]
            openset_max_probs = F.softmax(Fo, dim=1).max(dim=1)[0]

            all_scores = torch.cat([query_max_probs, openset_max_probs], dim=0)
            closed_labels = torch.cat([
                torch.ones(query_max_probs.size(0), device=self.device),
                -torch.ones(openset_max_probs.size(0), device=self.device)
            ], dim=0)

            self.openset_auroc = self.roc_area_calc(all_scores, closed_labels, descending=False)
            # with open(temp_log, "a") as f:
            #     f.write("ROC Area:\n")
            #     f.write(f"{self.openset_auroc}\n")
            #     f.write("all_scores:\n")
            #     all_scores = all_scores.cpu().numpy()
            #     all_scores_str = "\n".join([str(x.item()) for x in all_scores])
            #     f.write(all_scores_str)
            #     f.write("labels:\n")
            #     closed_labels = closed_labels.cpu().numpy()
            #     closed_labels_str = "\n".join([str(x.item()) for x in closed_labels])
            #     f.write(closed_labels_str)
            #     f.write("\n")

        return loss, acc
    
    def get_encoded_data(self, data):
        """
        Encode input data using the model and return the encoded results
        """
        input = self.encoder(data.x, data.edge_index)
        batch = data.batch

        from torch_geometric.nn import global_add_pool
        pool_input = global_add_pool(input, batch)

        return pool_input

    def get_processed_data(self, data):
            """
            Process data through the model and return the processed results
            
            Parameters:
            data: The input data
            
            Returns:
            The processed data (N_way*N_shot + N_way*N_query) x N_features
            """
            import torch.nn.functional as F
            eps = np.finfo(float).eps

            # Extract initial embeddings
            input = self.encoder(data.x, data.edge_index)
            target = data.y
            edge_index = data.edge_index
            batch = data.batch

            # Get support and query indices
            _, support_idxs, query_idxs = self.get_support_query_idxs(target)
            num_classes = len(torch.unique(target))
            num_support = self.opt["settings"]["few_shot"]["train"]["support_shots"]
            num_queries = 20 - num_support

            # Process labels
            s_labels_ori = torch.cat([target[idx_list] for idx_list in support_idxs])
            q_labels_ori = target[query_idxs]
            unique_labels = torch.unique(torch.cat([s_labels_ori, q_labels_ori]))
            label_map = {label.item(): idx for idx, label in enumerate(unique_labels)}
            s_labels_mapped = torch.tensor([label_map[label.item()] for label in s_labels_ori]).to(self.device)
            q_labels_mapped = torch.tensor([label_map[label.item()] for label in q_labels_ori]).to(self.device)
            
            # Generate one-hot labels
            s_labels = F.one_hot(s_labels_mapped, num_classes)
            q_labels = F.one_hot(q_labels_mapped, num_classes)

            # Pool embeddings
            pool_input = global_add_pool(input, batch)

            # Graph construction
            if self.args['rn'] in [30, 300]:
                self.relation.to(self.device)
                sigma = F.softplus(self.relation(input, edge_index, batch))
                
                sigma_support = torch.cat([sigma[idx_list] for idx_list in support_idxs])
                sigma_query = sigma[query_idxs]
                sigma_combined = torch.cat([sigma_support, sigma_query], 0)
                
                pool_input_support = torch.cat([pool_input[idx_list] for idx_list in support_idxs])
                pool_input_query = pool_input[query_idxs]
                pool_input_combined = torch.cat([pool_input_support, pool_input_query], 0)
                
                normalized_pool_input = pool_input_combined / (sigma_combined + eps)
            
            return normalized_pool_input