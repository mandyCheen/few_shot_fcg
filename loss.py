import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, Dropout
from models import GraphRelationNetwork
from torch_scatter import scatter_mean

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
        
    def compute_prototypes(self, input, support_idxs):
        """Compute prototype vectors"""
        return torch.stack([input[idx_list].mean(0) for idx_list in support_idxs])
    
    def get_nn_points(self, input, support_idxs):
        """Get nearest neighbor points"""
        return torch.cat([input[idx_list] for idx_list in support_idxs])
    
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

class ProtoLoss(Loss):
    """Prototypical Networks Loss with multiple distance metrics"""
    
    def __init__(self, opt: dict):
        self.metric_name = opt["settings"]["train"]["distance"]
        self.n_support = opt["settings"]["few_shot"]["train"]["support_shots"]
        self.loss_fn_name = opt["settings"]["train"]["loss"]
        
        if self.metric_name not in DISTANCE_METRICS:
            raise ValueError(f"Unsupported distance metric: {self.metric_name}. "
                           f"Supported metrics are: {list(DISTANCE_METRICS.keys())}")

        if self.loss_fn_name not in LOSS_FUNCTIONS:
            raise ValueError(f"Unsupported loss function: {self.loss_fn_name}. "
                           f"Supported loss functions are: {list(LOSS_FUNCTIONS.keys())}")
        
        self.distance_metric = DISTANCE_METRICS[self.metric_name]
        self.loss_fn = LOSS_FUNCTIONS[self.loss_fn_name]
        
        super().__init__(self.n_support)

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

        if self.metric_name == 'cosine_similarity':
            # 計算損失和準確率
            return self.loss_fn.compute(dists, n_classes, n_query)
        else:
            # 計算損失和準確率
            return self.loss_fn.compute(dists, n_classes, n_query, negative_distance=True)

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
            
        if self.metric_name not in DISTANCE_METRICS:
            raise ValueError(f"Unsupported distance metric: {self.metric_name}. "
                        f"Supported metrics are: {list(DISTANCE_METRICS.keys())}")
        if self.loss_fn_name not in LOSS_FUNCTIONS:
            raise ValueError(f"Unsupported loss function: {self.loss_fn_name}. "
                        f"Supported loss functions are: {list(LOSS_FUNCTIONS.keys())}")
        
        self.distance_metric = DISTANCE_METRICS[self.metric_name]
        self.loss_fn = LOSS_FUNCTIONS[self.loss_fn_name]
        
        super().__init__(self.n_support)
    
    def __call__(self, input, target):
        target_cpu = target.cpu()
        input_cpu = input.cpu()
        
        classes, support_idxs, query_idxs = self.get_support_query_idxs(target_cpu)
        n_classes = len(classes)
        n_query = target_cpu.eq(classes[0].item()).sum().item() - self.n_support
        
        nn_points = self.get_nn_points(input_cpu, support_idxs)
        
        query_samples = input_cpu[query_idxs]
        
        dists = self.distance_metric.compute(query_samples, nn_points)

        # find the nearest neighbor for each query point
        dists_by_class = dists.view(n_classes*n_query, n_classes, self.n_support)
        min_dists, _ = dists_by_class.min(dim=2)
        
        if self.metric_name == 'cosine_similarity':
            return self.loss_fn.compute(min_dists, n_classes, n_query)
        else:
            return self.loss_fn.compute(min_dists, n_classes, n_query, negative_distance=True)
    
    def get_metric_name(self):
        return self.metric_name
    
    def get_loss_fn(self):
        return self.loss_fn_name
    
class SoftNnLoss(Loss):
    def __init__(self, opt: dict):
        self.metric_name = opt["settings"]["train"]["distance"]
        self.n_support = opt["settings"]["few_shot"]["train"]["support_shots"]
        self.loss_fn_name = opt["settings"]["train"]["loss"]
        
        if self.metric_name not in DISTANCE_METRICS:
            raise ValueError(f"Unsupported distance metric: {self.metric_name}. "
                        f"Supported metrics are: {list(DISTANCE_METRICS.keys())}")
        if self.loss_fn_name not in LOSS_FUNCTIONS:
            raise ValueError(f"Unsupported loss function: {self.loss_fn_name}. "
                        f"Supported loss functions are: {list(LOSS_FUNCTIONS.keys())}")
        
        self.distance_metric = DISTANCE_METRICS[self.metric_name]
        self.loss_fn = LOSS_FUNCTIONS[self.loss_fn_name]
        
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

        classes, support_idxs, query_idxs = self.get_support_query_idxs(target_cpu)
        n_classes = len(classes)
        n_query = target_cpu.eq(classes[0].item()).sum().item() - self.n_support
        
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
            # For cosine similarity, larger values = more similar
            weights = F.softmax(dists_by_class, dim=-1)
            # Weight and sum the raw similarities
            weighted_dists = (weights * dists_by_class).sum(dim=2)
        else:
            # For distance metrics, smaller values = more similar
            # So we use negative distances for the softmax
            weights = F.softmax(-dists_by_class, dim=-1) 
            # Weight and sum the distances
            weighted_dists = (weights * dists_by_class).sum(dim=2)
            
        # Compute loss and accuracy using weighted distances
        if self.metric_name == 'cosine_similarity':
            return self.loss_fn.compute(weighted_dists, n_classes, n_query)
        else:
            return self.loss_fn.compute(weighted_dists, n_classes, n_query, negative_distance=True)

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

class LabelPropagation(nn.Module, Loss):
    """Label Propagation"""
    def __init__(self, opt: dict, encoder):
        super(LabelPropagation, self).__init__()
        self.opt = opt
        self.args = opt['settings']['few_shot']['parameters']
        self.device = opt["settings"]["train"]["device"]
        self.n_support = opt["settings"]["few_shot"]["train"]["support_shots"]
        self.encoder = encoder
        self.relation = GraphRelationNetwork(self.args["dim_in"], self.args["dim_hidden"], self.args["dim_out"], self.args["relation_layer"])

        if opt['settings']['few_shot']['parameters']['rn'] == 300:   # learned sigma, fixed alpha
            self.alpha = torch.tensor([opt['settings']['few_shot']['parameters']['alpha']], requires_grad=False).to(self.device)
        elif opt['settings']['few_shot']['parameters']['rn'] == 30:    # learned sigma, learned alpha
            self.alpha = nn.Parameter(torch.tensor([opt['settings']['few_shot']['parameters']['alpha']]).to(self.device), requires_grad=True)

    def forward(self, data):
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

        _, support_idxs, query_idxs = self.get_support_query_idxs(target)
        num_classes = len(torch.unique(target))
        num_support = self.opt["settings"]["few_shot"]["train"]["support_shots"]
        num_queries = 20 - num_support

        s_labels_ori = torch.cat([target[idx_list] for idx_list in support_idxs])
        q_labels_ori = target[query_idxs]
        unique_labels = torch.unique(torch.cat([s_labels_ori, q_labels_ori]))
        label_map = {label.item(): idx for idx, label in enumerate(unique_labels)}
        s_labels_mapped = torch.tensor([label_map[label.item()] for label in s_labels_ori]).to(self.device)
        q_labels_mapped = torch.tensor([label_map[label.item()] for label in q_labels_ori]).to(self.device)
        # generate one-hot labels
        s_labels = F.one_hot(s_labels_mapped, num_classes)
        q_labels = F.one_hot(q_labels_mapped, num_classes)

        from torch_geometric.nn import global_add_pool
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
            self.sigma = torch.cat([ self.sigma_support, self.sigma_query], 0)

            pool_input_support = torch.cat([pool_input[idx_list] for idx_list in support_idxs])
            pool_input_query = pool_input[query_idxs]
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
        yu = torch.zeros(num_classes * num_queries, num_classes).to(self.device)
        y = torch.cat((ys, yu), 0)
        F_ = torch.matmul(torch.inverse(torch.eye(N).to(self.device) - self.alpha * S + eps), y)
        Fq = F_[num_classes * num_support:, :]

        # Step4: Cross-Entropy Loss
        ce = nn.CrossEntropyLoss().to(self.device)
        ## both support and query loss
        gt = torch.argmax(torch.cat((s_labels, q_labels), 0), 1)
        loss = ce(F_, gt)
        ## acc
        predq = torch.argmax(Fq, 1)
        gtq = torch.argmax(q_labels, 1)
        correct = (predq == gtq).sum()

        total = num_queries * num_classes
        acc = 1.0 * correct.float() / float(total)

        return loss, acc
    
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
            from torch_geometric.nn import global_add_pool
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