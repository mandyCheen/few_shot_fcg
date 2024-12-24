#!/bin/bash

config_dir="./config"

# 設定實驗參數陣列
declare -A factors
factors["expSet"]="1 2 3 4"          # 實驗設定：5way5shot, 5way10shot, 10way5shot, 10way10shot
factors["decisionNet"]="1 2"         # 決策網路：ProtoNet, NnNet
factors["pretrain"]="1 2"            # 預訓練：with_pretrain, without_pretrain


show_help() {
    cat << EOF
使用方法：
    $0 --seed <random_seed> --cuda <cuda_device>

參數說明：
    --help          顯示此幫助訊息
    --seed          設定隨機種子
    --cuda          設定 CUDA 裝置編號

範例：
    $0 --seed 42 --cuda 0
EOF
}

generate_config() {
    local datasetName=$1
    local expSet=$2
    local decisionNet=$3
    local pretrain=$4
    local cudaDevice=$5
    local seed=$6
    local logFile=$7

    local expName=""
    local decisionNetName=""
    local pretrainName=""
    local lr=0.0
    local projection_lr=0.0
    local weight=""
    local support_shots=0
    local query_shots=0
    local class_per_iter=0

    
    config_file="${config_dir}/config_${datasetName}.json"
    
    if [ -f $config_file ]; then
        rm $config_file
    fi

    if [ $expSet -eq 1 ]; then
        expName="5way_5shot"
        support_shots=5
        query_shots=15
        class_per_iter=5
    elif [ $expSet -eq 2 ]; then
        expName="5way_10shot"
        support_shots=10
        query_shots=10
        class_per_iter=5
    elif [ $expSet -eq 3 ]; then
        expName="10way_5shot"
        support_shots=5
        query_shots=15
        class_per_iter=10
    elif [ $expSet -eq 4 ]; then
        expName="10way_10shot"
        support_shots=10
        query_shots=10
        class_per_iter=10
    fi

    if [ $decisionNet -eq 1 ]; then
        decisionNetName="ProtoNet"
    elif [ $decisionNet -eq 2 ]; then
        decisionNetName="NnNet"
    fi

    if [ $pretrain -eq 1 ]; then
        pretrainName="with_pretrain"
        lr=0.0005
        projection_lr=0.001
        weight="x86_pretrained_20241121_1653"
    elif [ $pretrain -eq 2 ]; then
        pretrainName="without_pretrain"
        lr=0.001
        projection_lr=0.001
        weight=""
    fi

    # 記錄到日誌檔案
    {
        echo "======================================"
        echo "Experiment Set: $expName"
        echo "Decision Network: $decisionNetName"
        echo "Pretrain: $pretrainName"
        echo "Timestamp: $(date)"
        echo "======================================"
    } >> $logFile
    


    # 創建配置檔案
    cat > $config_file << EOL
{
    "dataset": {
        "pack_filter": "diec",
        "cpu_arch": "x86_64",
        "reverse_tool": "ghidra",
        "raw": "malware_diec_ghidra_x86_64_fcg_dataset.csv",
        "split_by_cpu": false,
        "pretrain_family": [
            "gafgyt",
            "ngioweb",
            "mirai",
            "tsunami"
        ]
    },
    "pretrain": {
        "name": "x86_pretrained",
        "use": true,
        "raw_dataset": "malware_diec_ghidra_x86_64_fcg_pretrain_dataset.csv",
        "batch_size": 128
    },
    "settings": {
        "name": "${expName}_${decisionNetName}_${pretrainName}",
        "model": {
            "model_name": "GraphSAGE",
            "input_size": 128,
            "hidden_size": 128,
            "output_size": 128,
            "num_layers": 2,
            "projection": true,
            "load_weights": "${weight}"
        },
        "train": {
            "training": true,
            "validation": true,
            "num_epochs": 500,
            "device": "cuda:${cudaDevice}",
            "parallel": false,
            "parallel_device": [],
            "iterations": 100,
            "lr": ${lr},
            "projection_lr": ${projection_lr},
            "lr_scheduler": {
                "use": true,
                "method": "ReduceLROnPlateau",
                "step_size": 20,
                "gamma": 0.5,
                "patience": 10,
                "factor": 0.5
            },
            "early_stopping": {
                "use": true,
                "patience":  30
            },
            "loss": "CrossEntropyLoss",
            "distance": "euclidean",
            "optimizer": "AdamW",
            "save_model": true
        },
        "few_shot": {
            "use": false,
            "method": "${decisionNetName}",
            "train": {
                "support_shots": ${support_shots},
                "query_shots": ${query_shots},
                "class_per_iter": ${class_per_iter}
            },
            "test": {
                "support_shots": ${support_shots},
                "query_shots": ${query_shots},
                "class_per_iter": ${class_per_iter}
            }
        },
        "vectorize": {
            "node_embedding_method": "word2vec",
            "node_embedding_size": 128,
            "num_workers": 4
        },
        "seed": ${seed}
    },
    "paths": {
        "data": {
            "fcg_dataset": "./dataset/data_ghidra_fcg",
            "csv_folder": "./dataset/raw_csv",
            "split_folder": "./dataset/split",
            "embedding_folder": "./embeddings",
            "pretrain_dataset": "./dataset/data_ghidra_fcg_pretrain"
        },
        "model": {
            "model_folder": "./checkpoints",
            "pretrained_folder": "./pretrained"
        }
    }
}
EOL

}

# 執行實驗的函數
run_experiment() {
    local config_file=$1
    local expSet=$2
    local decisionNet=$3
    local pretrain=$4
    local logFile=$5
    

    echo "Experiment: ${expSet}"
    echo "Decision Network: ${decisionNet}"
    echo "Pretrain: ${pretrain}"

    # 執行 Python 腳本
    python main.py --config $config_file
    
    # 檢查實驗是否成功
    if [ $? -eq 0 ]; then
        echo "Successfully completed the experiment" >> $logFile
    else
        echo "Failed to complete the experiment" >> $logFile
    fi

}

while [[ $# -gt 0 ]]; do
    case $1 in
        --help)
            show_help
            exit 0
            ;;
        --seed)
            seed="$2"
            shift 2
            ;;
        --cuda)
            cuda="$2"
            shift 2
            ;;
        *)
            echo "未知的參數: $1"
            show_help
            exit 1
            ;;
    esac
done

# 檢查必要參數
if [ -z "$seed" ] || [ -z "$cuda" ]; then
    echo "錯誤：需要設定 --seed 和 --cuda"
    show_help
    exit 1
fi

datasetName="NICT_Ghidra_x86_64"

echo "Implementing Few-Shot Learning for Malware Classification"
echo "Dataset: $datasetName"
echo "Random Seed: $seed"
echo "CUDA Device Number: $cuda"

datasetName="${datasetName}_${seed}"
logFile="./logs/log_${datasetName}.txt"

> $logFile

# 迴圈執行實驗
for expSet in ${factors["expSet"]}; do
    for decisionNet in ${factors["decisionNet"]}; do
        for pretrain in ${factors["pretrain"]}; do
            echo "Experiment Set: $expSet"
            echo "Decision Network: $decisionNet"
            echo "Pretrain: $pretrain"

            # 生成配置檔案
            generate_config "$datasetName" "$expSet" "$decisionNet" "$pretrain" "$cuda" "$seed" "$logFile"
            
            # 執行實驗
            run_experiment "${config_dir}/config_${datasetName}.json" "$expSet" "$decisionNet" "$pretrain" "$logFile"
            
            # 休息一下
            sleep 2
        done
    done
done

echo "All experiments have been completed"