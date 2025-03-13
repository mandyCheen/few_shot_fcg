import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import os
import gc
from datetime import datetime

try:
    import psutil
    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False
    print("psutil 未安裝，將不提供系統資源監控")

# 設置隨機種子以確保結果可重現
torch.manual_seed(42)
np.random.seed(42)

# 創建一個更複雜的模型以增加計算量
class ComplexModel(nn.Module):
    def __init__(self, input_size=784, hidden_size=512, num_layers=3, output_size=10):
        super(ComplexModel, self).__init__()
        self.input_size = input_size
        self.layers = nn.ModuleList()
        
        # 輸入層
        self.layers.append(nn.Linear(input_size, hidden_size))
        
        # 隱藏層
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        
        # 輸出層
        self.output = nn.Linear(hidden_size, output_size)
        
        # 批次歸一化層
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_size) for _ in range(num_layers)])
        
    def forward(self, x):
        # 修正：確保輸入張量形狀正確
        batch_size = x.size(0)
        
        # 重新調整維度，確保形狀正確
        x = x.reshape(batch_size, -1)
        
        # 確保輸入維度適合第一個線性層
        if x.size(1) != self.input_size:
            raise ValueError(f"輸入特徵數量 {x.size(1)} 與模型期望的 {self.input_size} 不匹配")
        
        # 應用所有層
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # 批次歸一化需要正確的形狀 [batch_size, features]
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=0.3, training=self.training)
        
        # 應用輸出層
        x = self.output(x)
        return x

# 創建一個大型隨機數據集
class LargeRandomDataset(Dataset):
    def __init__(self, batch_size, input_size, num_samples):
        self.len = num_samples
        self.input_size = input_size
        self.batch_size = batch_size
        
        # 創建足夠大的數據
        print(f"正在生成大型數據集（{num_samples}樣本，每個大小為{batch_size}x{input_size}）...")
        self.data = torch.randn(num_samples, batch_size, input_size)
        self.labels = torch.randint(0, 10, (num_samples, batch_size))
        print("數據集生成完成！")
        
    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
    def __len__(self):
        return self.len

# 獲取系統資訊的函數
def get_system_info():
    if not HAVE_PSUTIL:
        return {"message": "psutil未安裝，無法監控系統資源"}
    
    # CPU信息
    cpu_usage = psutil.cpu_percent(interval=0.1)
    # 內存信息
    memory = psutil.virtual_memory()
    # GPU信息
    gpu_info = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_info.append({
                "id": i,
                "name": torch.cuda.get_device_name(i),
                "memory_allocated": torch.cuda.memory_allocated(i) / 1024**3, # GB
                "memory_reserved": torch.cuda.memory_reserved(i) / 1024**3,  # GB
                "max_memory_allocated": torch.cuda.max_memory_allocated(i) / 1024**3 # GB
            })
    
    return {
        "cpu_usage": cpu_usage,
        "memory_used_gb": memory.used / 1024**3,
        "memory_percent": memory.percent,
        "gpu_info": gpu_info
    }

# 打印模型參數和張量設備位置
def print_model_device_info(model):
    print("\n===== 模型參數設備信息 =====")
    if isinstance(model, nn.DataParallel):
        print(f"DataParallel設備IDs: {model.device_ids}")
        print(f"DataParallel輸出設備: {model.output_device}")
        module = model.module
    else:
        module = model
    
    for name, param in module.named_parameters():
        print(f"參數 {name}: 形狀 {param.shape}, 設備 {param.device}")

# 顯示CUDA事件的詳細計時
def measure_execution_time(model, sample_input, target, num_warmup=3, num_measurements=10):
    # 確保輸入和模型在相同設備上
    device = next(model.parameters()).device
    sample_input = sample_input.to(device)
    target = target.to(device)
    
    # 確保輸入形狀正確
    if isinstance(model, nn.DataParallel):
        input_size = model.module.input_size
    else:
        input_size = model.input_size
        
    batch_size = sample_input.size(0)
    sample_input = sample_input.reshape(batch_size, -1)
    
    if sample_input.size(1) != input_size:
        print(f"警告：調整輸入形狀從 {sample_input.shape} 到 {(batch_size, input_size)}")
        # 創建新的、形狀正確的輸入
        sample_input = torch.randn(batch_size, input_size, device=device)
    
    # 預熱
    print("預熱中...")
    for _ in range(num_warmup):
        with torch.no_grad():
            model(sample_input)
    
    # 測量前向傳播時間
    forward_times = []
    print("測量前向傳播時間...")
    for _ in range(num_measurements):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        with torch.no_grad():
            output = model(sample_input)
        end_event.record()
        
        torch.cuda.synchronize()
        forward_times.append(start_event.elapsed_time(end_event))
    
    # 測量反向傳播時間
    backward_times = []
    criterion = nn.CrossEntropyLoss().to(device)
    print("測量反向傳播時間...")
    
    for _ in range(num_measurements):
        # 清除漸變
        if isinstance(model, nn.DataParallel):
            model.module.zero_grad()
        else:
            model.zero_grad()
        
        # 前向傳播
        output = model(sample_input)
        loss = criterion(output, target)
        
        # 測量反向傳播時間
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        loss.backward()
        end_event.record()
        
        torch.cuda.synchronize()
        backward_times.append(start_event.elapsed_time(end_event))
    
    # 計算統計數據
    forward_mean = np.mean(forward_times)
    forward_std = np.std(forward_times)
    backward_mean = np.mean(backward_times)
    backward_std = np.std(backward_times)
    
    print(f"\n===== 執行時間分析 (ms) =====")
    print(f"前向傳播: {forward_mean:.2f} ± {forward_std:.2f}")
    print(f"反向傳播: {backward_mean:.2f} ± {backward_std:.2f}")
    print(f"總計: {forward_mean + backward_mean:.2f} ± {(forward_std**2 + backward_std**2)**0.5:.2f}")
    
    return {
        "forward_mean": forward_mean,
        "forward_std": forward_std,
        "backward_mean": backward_mean,
        "backward_std": backward_std,
        "total_mean": forward_mean + backward_mean
    }

# 測試函數，專注於平行計算性能
def test_parallel_performance(model, device_ids=None, batch_size=64, input_size=784, num_samples=100, num_epochs=3):
    print("\n" + "="*50)
    print(f"開始測試 - 批次大小: {batch_size}, 輸入維度: {input_size}, 樣本數: {num_samples}, Epochs: {num_epochs}")
    print("="*50)
    
    # 設置設備
    if torch.cuda.is_available() and device_ids:
        device_count = len(device_ids)
        print(f"使用 {device_count} 個GPU: {device_ids}")
        primary_device = f"cuda:{device_ids[0]}"
        
        # 確保模型先移到主GPU上
        model = model.to(primary_device)
        
        # 只有多個GPU時才使用DataParallel
        if device_count > 1:
            model = nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])
            print(f"模型使用DataParallel包裝，主GPU為: cuda:{device_ids[0]}")
        else:
            print(f"使用單GPU: {primary_device}")
            
        device = torch.device(primary_device)
    else:
        print("使用CPU進行訓練 (可能會非常慢)")
        device = torch.device("cpu")
        model = model.to(device)
    
    # 顯示當前系統狀態
    if HAVE_PSUTIL:
        print("\n===== 初始系統狀態 =====")
        sys_info = get_system_info()
        print(f"CPU使用率: {sys_info['cpu_usage']}%")
        print(f"內存使用: {sys_info['memory_used_gb']:.2f} GB ({sys_info['memory_percent']}%)")
        for gpu in sys_info['gpu_info']:
            print(f"GPU {gpu['id']} ({gpu['name']}): 已分配 {gpu['memory_allocated']:.2f} GB, 已保留 {gpu['memory_reserved']:.2f} GB")
    
    # 創建大型數據集以增加計算負荷
    dataset = LargeRandomDataset(batch_size, input_size, num_samples)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True)
    
    # 打印模型資訊
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型總參數數量: {total_params:,}")
    
    # 顯示模型參數所在的設備
    print_model_device_info(model)
    
    # 準備一個樣本進行測量
    sample_data, sample_labels = next(iter(dataloader))
    sample_input = sample_data.reshape(batch_size, -1).to(device)  # 確保形狀正確
    sample_targets = sample_labels.reshape(-1).to(device)  # 展平標籤
    
    # 確保輸入維度與模型匹配
    if isinstance(model, nn.DataParallel):
        expected_size = model.module.input_size
    else:
        expected_size = model.input_size
        
    if sample_input.size(1) != expected_size:
        print(f"調整樣本輸入從 {sample_input.shape} 到 {(batch_size, expected_size)}")
        sample_input = torch.randn(batch_size, expected_size, device=device)
    
    # 測量單批次執行時間
    print("\n進行單批次執行時間測量...")
    time_stats = measure_execution_time(model, sample_input, sample_targets)
    
    # 定義損失函數和優化器
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 訓練循環
    print("\n===== 開始訓練循環 =====")
    total_batches = len(dataloader)
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        running_loss = 0.0
        batch_times = []
        
        for i, (inputs, labels) in enumerate(dataloader):
            batch_start = time.time()
            
            # 將輸入和標籤移到正確的設備上並確保形狀正確
            inputs = inputs.reshape(batch_size, -1).to(device, non_blocking=True)
            labels = labels.reshape(-1).to(device, non_blocking=True)
            
            # 前向傳播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向傳播和優化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 記錄損失和時間
            running_loss += loss.item()
            batch_end = time.time()
            batch_time = batch_end - batch_start
            batch_times.append(batch_time)
            
            # 定期報告進度和性能
            if (i + 1) % 10 == 0 or (i + 1) == total_batches:
                # 計算批次處理速度
                samples_per_sec = batch_size / np.mean(batch_times[-10:] if len(batch_times) >= 10 else batch_times)
                eta = (total_batches - i - 1) * np.mean(batch_times[-10:] if len(batch_times) >= 10 else batch_times)
                
                # 獲取系統狀態
                if HAVE_PSUTIL:
                    sys_info = get_system_info()
                
                print(f"\rEpoch {epoch+1}/{num_epochs} - 批次 {i+1}/{total_batches} - "
                      f"損失: {loss.item():.4f} - "
                      f"批次時間: {batch_time:.4f}秒 - "
                      f"速度: {samples_per_sec:.2f} 樣本/秒 - "
                      f"剩餘時間: {eta:.2f}秒", end="")
                
                # 每50個批次顯示詳細資訊
                if (i + 1) % 50 == 0 and HAVE_PSUTIL:
                    print("\n")
                    print(f"時間戳: {datetime.now().strftime('%H:%M:%S')}")
                    print(f"CPU使用率: {sys_info['cpu_usage']}%")
                    print(f"內存使用: {sys_info['memory_used_gb']:.2f} GB ({sys_info['memory_percent']}%)")
                    
                    for gpu in sys_info['gpu_info']:
                        print(f"GPU {gpu['id']} ({gpu['name']}): "
                              f"已分配 {gpu['memory_allocated']:.2f} GB, "
                              f"已保留 {gpu['memory_reserved']:.2f} GB, "
                              f"峰值 {gpu['max_memory_allocated']:.2f} GB")
        
        # 每個epoch的統計
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        avg_loss = running_loss / total_batches
        avg_batch_time = np.mean(batch_times)
        avg_samples_per_sec = batch_size / avg_batch_time
        
        print(f"\n[Epoch {epoch+1}/{num_epochs}] "
              f"平均損失: {avg_loss:.4f} - "
              f"Epoch時間: {epoch_time:.2f}秒 - "
              f"平均批次時間: {avg_batch_time:.4f}秒 - "
              f"平均處理速度: {avg_samples_per_sec:.2f} 樣本/秒")
    
    # 訓練完成，顯示總結
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "="*50)
    print(f"訓練完成 - 總時間: {total_time:.2f} 秒")
    print(f"每個epoch平均時間: {total_time/num_epochs:.2f} 秒")
    print("="*50)
    
    # 顯示最終系統狀態
    if HAVE_PSUTIL:
        print("\n===== 最終系統狀態 =====")
        sys_info = get_system_info()
        print(f"CPU使用率: {sys_info['cpu_usage']}%")
        print(f"內存使用: {sys_info['memory_used_gb']:.2f} GB ({sys_info['memory_percent']}%)")
        for gpu in sys_info['gpu_info']:
            print(f"GPU {gpu['id']} ({gpu['name']}): "
                  f"已分配 {gpu['memory_allocated']:.2f} GB, "
                  f"已保留 {gpu['memory_reserved']:.2f} GB, "
                  f"峰值 {gpu['max_memory_allocated']:.2f} GB")
    
    # 清理內存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return {
        "total_time": total_time,
        "avg_epoch_time": total_time / num_epochs,
        "batches_per_sec": total_batches / total_time,
        "samples_per_sec": batch_size * total_batches / total_time
    }

if __name__ == "__main__":
    print(f"PyTorch版本: {torch.__version__}")
    
    # 檢查可用的GPU數量
    num_gpus = torch.cuda.device_count()
    print(f"系統上可用的GPU數量: {num_gpus}")
    
    if num_gpus > 0:
        for i in range(num_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # 配置參數 - 減小維度以避免形狀不匹配問題
    input_size = 784     # 標準MNIST大小，以確保兼容性
    hidden_size = 512    # 中等大小的隱藏層
    num_layers = 3       # 適中的層數
    batch_size = 64      # 標準批次大小
    num_samples = 100    # 減少樣本數量，但保持足夠的工作量
    num_epochs = 2       # 減少epoch數以加快測試
    
    # 創建複雜模型
    model = ComplexModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
    print(f"\n創建了複雜模型：{num_layers} 層，"
          f"輸入維度 {input_size}，"
          f"隱藏維度 {hidden_size}")
    
    # 只執行平行GPU測試
    results = {}
    
    # 1. 單GPU測試
    if num_gpus > 0:
        print("\n\n===== 單GPU測試 =====")
        results['single_gpu'] = test_parallel_performance(
            model, device_ids=[0], 
            batch_size=batch_size, input_size=input_size, 
            num_samples=num_samples, num_epochs=num_epochs
        )
    
    # 2. 多GPU測試 (如果有多個GPU)
    if num_gpus > 1:
        print("\n\n===== 多GPU測試 =====")
        # 使用所有可用的GPU
        device_ids = list(range(num_gpus))
        results['multi_gpu'] = test_parallel_performance(
            model, device_ids=device_ids, 
            batch_size=batch_size, input_size=input_size, 
            num_samples=num_samples, num_epochs=num_epochs
        )
    
    # 比較結果
    if len(results) > 1:
        speedup = results['single_gpu']['total_time'] / results['multi_gpu']['total_time']
        print("\n" + "="*50)
        print(f"性能比較: 單GPU vs 多GPU ({num_gpus}個)")
        print(f"加速比: {speedup:.2f}x")
        print(f"單GPU樣本處理速度: {results['single_gpu']['samples_per_sec']:.2f} 樣本/秒")
        print(f"多GPU樣本處理速度: {results['multi_gpu']['samples_per_sec']:.2f} 樣本/秒")
        print("="*50)
        
        # 理論加速比vs實際加速比
        theoretical_speedup = num_gpus
        efficiency = (speedup / theoretical_speedup) * 100
        print(f"並行效率: {efficiency:.2f}% (理論加速比: {theoretical_speedup:.1f}x, 實際加速比: {speedup:.2f}x)")