import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import load_config, save_config, reshape_processed_data
from loadDataset import LoadDataset
from trainModule import TestModule
import torch
import numpy as np
from tqdm import tqdm
import math
import warnings
import time
warnings.filterwarnings("ignore")

def GenerateRunSet(test: TestModule):
    """Generate ndatas in shape of (n_runs, n_ways, n_shot + n_queries, n_features)"""
    ndatas = torch.zeros(test.iterations, test.class_per_iter_test, test.support_shots_test + test.query_shots_test, test.embeddingSize)
    iter = 0
    test.load_best_model()
    for data in tqdm(test.testLoader):
        test.model.eval()
        data = data.to(test.device)
        with torch.no_grad():
            if (test.opt["settings"]["few_shot"]["method"] == "LabelPropagation"):
                processed_data = test.model.get_processed_data(data) ### (N_way*N_shot + N_way*N_query) x N_features
                reshaped_data = reshape_processed_data(processed_data, test.class_per_iter_test, test.support_shots_test, test.query_shots_test)
                ndatas[iter] = torch.tensor(reshaped_data)
            else: 
                processed_data = test.model(data)
                processed_data = processed_data.cpu()
                reshaped_data = test.loss_fn.reshape_random_data(processed_data, data.y.cpu(), test.class_per_iter_test, test.support_shots_test, test.query_shots_test)
                ndatas[iter] = reshaped_data
                del processed_data, reshaped_data
        torch.cuda.empty_cache()
        iter += 1
    return ndatas

def scaleEachUnitaryDatas(datas):
    """Scale each unitary data"""
    norms = datas.norm(dim=2, keepdim=True)
    return datas/norms

def SVDreduction(ndatas,K):
    """Reduce the dimension of ndatas using SVD"""
    _,s,v = torch.svd(ndatas)
    ndatas = ndatas.matmul(v[:,:,:K])

    return ndatas

def centerDatas(datas): 
    """Center the datas"""
    datas= datas - datas.mean(1, keepdim=True)
    datas = datas / torch.norm(datas, dim=2, keepdim= True)

    return datas

def predictW(gamma, Z, labels):
    # #Certainty_scores = 1 + (Z*torch.log(Z)).sum(dim=2) / math.log(5)
    # Z[:,:n_lsamples].fill_(0)
    # Z[:,:n_lsamples].scatter_(2,labels[:,:n_lsamples].unsqueeze(2), 1)
    Y = torch.zeros(n_runs,n_lsamples, n_ways,device='cuda')
    Y.scatter_(2,labels[:,:n_lsamples].unsqueeze(2), 1)
    tZ_Z = torch.bmm(torch.transpose(Z,1,2), Z)
    delta = torch.sum(Z, 1)
    L = tZ_Z - torch.bmm(tZ_Z, tZ_Z/delta.unsqueeze(1))
    Z_l = Z[:,:n_lsamples]

    #A = np.dot(np.linalg.inv(torch.matmul(torch.transpose(Z_l,1,2), Z_l) + gamma * L), torch.bmm(torch.transpose(Z_l,1,2), Y))
    u = torch.linalg.cholesky(torch.bmm(torch.transpose(Z_l,1,2), Z_l) + gamma * L)# + 0.1*
    #u = torch.linalg.cholesky(gamma * L)
    A = torch.cholesky_solve(torch.bmm(torch.transpose(Z_l,1,2), Y), u)
    P = Z.bmm(A)
    _, n, m = P.shape
    r = torch.ones(n_runs, n_lsamples + n_usamples,device='cuda')
    c = torch.ones(n_runs, n_ways,device='cuda') * (n_shots + n_queries)
    u = torch.zeros(n_runs, n).cuda()
    maxiters = 1000
    iters = 1
    # normalize this matrix
    while torch.max(torch.abs(u - P.sum(2))) > 0.01:
        u = P.sum(2)
        P *= (r / u).view((n_runs, -1, 1))
        P *= (c / P.sum(1)).view((n_runs, 1, -1))
        P[:,:n_lsamples].fill_(0)
        P[:,:n_lsamples].scatter_(2,labels[:,:n_lsamples].unsqueeze(2), 1)
        if iters == maxiters:
            break
        iters = iters + 1
    return P

class Model:
    def __init__(self, n_ways):
        self.n_ways = n_ways
              
class GaussianModel(Model):
    def __init__(self, n_ways, lam):
        super(GaussianModel, self).__init__(n_ways)
        self.mus = None         # shape [n_runs][n_ways][n_nfeat]
        self.lam = lam
        
    def clone(self):
        other = GaussianModel(self.n_ways)
        other.mus = self.mus.clone()
        return self

    def cuda(self):
        self.mus = self.mus.cuda()
        
    def initFromLabelledDatas(self, ndatas, n_runs, n_shot, n_queries, n_ways, n_nfeat):
        self.mus_ori = ndatas.reshape(n_runs, n_shot+n_queries,n_ways, n_nfeat)[:,:n_shot,].mean(1)
        self.mus = self.mus_ori.clone()
        self.mus = self.mus / self.mus.norm(dim=2, keepdim=True)


    def initFromCenter(self, mus):
        self.mus = mus
        self.mus = self.mus / self.mus.norm(dim=2, keepdim=True)


    def updateFromEstimate(self, estimate, alpha, l2 = False):

        diff = self.mus_ori - self.mus
        Dmus = estimate - self.mus
        if l2 == True:
            self.mus = self.mus + alpha * (Dmus) + 0.01 * diff
        else:
            self.mus = self.mus + alpha * (Dmus)

    def compute_optimal_transport(self, M, r, c, epsilon=1e-6):
        
        r = r.cuda()
        c = c.cuda()
        n_runs, n, m = M.shape
        P = torch.exp(- self.lam * M)
        P /= P.view((n_runs, -1)).sum(1).unsqueeze(1).unsqueeze(1)
                                         
        u = torch.zeros(n_runs, n).cuda()
        maxiters = 1000
        iters = 1
        # normalize this matrix
        while torch.max(torch.abs(u - P.sum(2))) > epsilon:
            u = P.sum(2)
            P *= (r / u).view((n_runs, -1, 1))
            P *= (c / P.sum(1)).view((n_runs, 1, -1))
            if iters == maxiters:
                break
            iters = iters + 1
        return P, torch.sum(P * M)
    
    def getProbas(self, ndatas, n_runs, n_ways, n_usamples, n_lsamples):

        dist = (ndatas.unsqueeze(2)-self.mus.unsqueeze(1)).norm(dim=3).pow(2)
        
        p_xj = torch.zeros_like(dist)

        r = torch.ones(n_runs, n_usamples)
        c = torch.ones(n_runs, n_ways) * (n_queries)
       
        p_xj_test, _ = self.compute_optimal_transport(dist[:, n_lsamples:], r, c, epsilon=1e-3)
        p_xj[:, n_lsamples:] = p_xj_test
        p_xj[:,:n_lsamples].fill_(0)
        p_xj[:,:n_lsamples].scatter_(2,labels[:,:n_lsamples].unsqueeze(2), 1)
        
        return p_xj

    def estimateFromMask(self, mask, ndatas):

        emus = mask.permute(0,2,1).matmul(ndatas).div(mask.sum(dim=1).unsqueeze(2))

        return emus

class MAP:
    def __init__(self, alpha=None):
        
        self.verbose = False
        self.progressBar = False
        self.alpha = alpha
    
    def getAccuracy(self, probas):
        olabels = probas.argmax(dim=2)
        matches = labels.eq(olabels).float()
        acc_test = matches[:,n_lsamples:].mean(1)    

        m = acc_test.mean().item()
        pm = acc_test.std().item() *1.96 / math.sqrt(n_runs)
        return m, pm
    
    def performEpoch(self, model, ndatas, n_runs, n_ways, n_usamples, n_lsamples, epochInfo=None):
     
        p_xj = model.getProbas(ndatas, n_runs, n_ways, n_usamples, n_lsamples)
        self.probas = p_xj
        
        if self.verbose:
            print("accuracy from filtered probas", self.getAccuracy(self.probas))

        m_estimates = model.estimateFromMask(self.probas,ndatas)
               
        # update centroids
        model.updateFromEstimate(m_estimates, self.alpha)
        #self.alpha -= 0.001
        if self.verbose:
            op_xj = model.getProbas(ndatas, n_runs, n_ways, n_usamples, n_lsamples)
            acc = self.getAccuracy(op_xj)
            print("output model accuracy", acc)
        
    def loop(self, model, ndatas, n_runs, n_ways, n_usamples, n_lsamples, n_epochs=20):
        
        self.probas = model.getProbas(ndatas, n_runs, n_ways, n_usamples, n_lsamples)
        if self.verbose:
            print("initialisation model accuracy", self.getAccuracy(self.probas))

        if self.progressBar:
            if type(self.progressBar) == bool:
                pb = tqdm(total = n_epochs)
            else:
                pb = self.progressBar
           
        for epoch in range(1, n_epochs+1):
            if self.verbose:
                print("----- epoch[{:3d}]  lr_p: {:0.3f}".format(epoch, self.alpha))
            p_xj = model.getProbas(ndatas, n_runs, n_ways, n_usamples, n_lsamples)
            self.probas = p_xj

            if self.verbose:
                print("accuracy from filtered probas", self.getAccuracy(self.probas))
            pesudo_L = predictW(0.05, self.probas, labels)
            if self.verbose:
                print("accuracy from AnchorGraph probas", self.getAccuracy(pesudo_L))
            beta = 0.7
            m_estimates = model.estimateFromMask((beta*pesudo_L + (1-beta)*p_xj).clamp(0,1), ndatas)

            model.updateFromEstimate(m_estimates, self.alpha)
            if self.verbose:
                op_xj = model.getProbas(ndatas, n_runs, n_ways, n_usamples, n_lsamples)
                acc = self.getAccuracy(op_xj)
                print("output model accuracy", acc)
            if (self.progressBar): pb.update()
        
        # get final accuracy and return it
        op_xj = model.getProbas(ndatas, n_runs, n_ways, n_usamples, n_lsamples)
        acc = self.getAccuracy(op_xj)
        return acc
    


if __name__ == '__main__':


    config_paths = [
                    "/home/mandy/Projects/few_shot_fcg/checkpoints/x86_64_withVal_withPretrain_ghidra_6_baseline/5way_5shot_LabelPropagation_20250226_151803/config.json", 
                    "/home/mandy/Projects/few_shot_fcg/checkpoints/x86_64_withVal_withPretrain_ghidra_6_baseline/5way_10shot_LabelPropagation_20250226_153804/config.json",
                    "/home/mandy/Projects/few_shot_fcg/checkpoints/x86_64_withVal_withPretrain_ghidra_6_baseline/10way_5shot_LabelPropagation_20250226_161219/config.json",
                    "/home/mandy/Projects/few_shot_fcg/checkpoints/x86_64_withVal_withPretrain_ghidra_6_baseline/10way_10shot_LabelPropagation_20250226_164443/config.json",
                    "/home/mandy/Projects/few_shot_fcg/checkpoints/x86_64_withVal_withPretrain_ghidra_6_baseline/5way_5shot_NnNet_with_pretrain_20250104_010925/config.json",
                    "/home/mandy/Projects/few_shot_fcg/checkpoints/x86_64_withVal_withPretrain_ghidra_6_baseline/5way_10shot_NnNet_with_pretrain_20250104_113856/config.json",
                    "/home/mandy/Projects/few_shot_fcg/checkpoints/x86_64_withVal_withPretrain_ghidra_6_baseline/10way_5shot_NnNet_with_pretrain_20250105_115117/config.json",
                    "/home/mandy/Projects/few_shot_fcg/checkpoints/x86_64_withVal_withPretrain_ghidra_6_baseline/10way_10shot_NnNet_with_pretrain_20250106_153519/config.json"
                    ]
    
    for config_path in config_paths:

        options = load_config(config_path)
        n_shots = options["settings"]["few_shot"]["test"]["support_shots"]
        n_ways = options["settings"]["few_shot"]["test"]["class_per_iter"]
        n_queries = 20 - n_shots
        n_runs = 10000
        n_lsamples = n_ways * n_shots
        n_usamples = n_ways * n_queries
        n_samples = n_lsamples + n_usamples

        cfg = {'shot':n_shots, 'ways':n_ways, 'queries':n_queries}

        ### Load Dataset & Generate ndatas
        print("Loading Dataset...")

        options["settings"]["train"]["iterations"] = n_runs
        dataset = LoadDataset(options, pretrain=False)
        test = TestModule(config_path, dataset, options)
        model_folder = os.path.join(test.model_folder)
        name = os.path.basename(model_folder)
        parent_folder = os.path.basename(os.path.dirname(model_folder))
        ndata_path = os.path.join(test.embeddingFolder, "protoLP_{}_{}.pkl".format(parent_folder, name))
        if not os.path.exists(ndata_path):
            ndatas = GenerateRunSet(test)
            torch.save(ndatas, ndata_path)
        else:
            ndatas = torch.load(ndata_path)
        print("Size of the ndatas...", ndatas.size())
        ### Preprocess ndatas
        print("Preprocessing Datas...")

        ndatas = ndatas.permute(0,2,1,3).reshape(n_runs, n_samples, -1)
        labels = torch.arange(n_ways).view(1,1,n_ways).expand(n_runs,n_shots+n_queries,n_ways).clone().view(n_runs, n_samples)
        
        ndatas = scaleEachUnitaryDatas(ndatas)
        ndatas = SVDreduction(ndatas, 40)
        n_nfeat = ndatas.size(2)

        ndatas = centerDatas(ndatas)
        print("Size of the datas...", ndatas.size())

        ndatas = ndatas.to(test.device)
        labels = labels.to(test.device)

        lam = 10
        model = GaussianModel(n_ways, lam)
        model.initFromLabelledDatas(ndatas, n_runs, n_shots,n_queries,n_ways,n_nfeat)
        
        alpha = 0.2
        optim = MAP(alpha)

        optim.verbose=True
        optim.progressBar=True

        T1 = time.perf_counter()
        acc_test = optim.loop(model, ndatas, n_runs, n_ways, n_usamples, n_lsamples, n_epochs=50)
        print('running time:%s ' % (time.perf_counter() - T1))
        print("final accuracy found {:0.2f} +- {:0.2f}".format(*(100*x for x in acc_test)))
        
        