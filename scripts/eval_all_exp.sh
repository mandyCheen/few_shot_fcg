#!/bin/bash

# 顯示使用說明
usage() {
    echo "Usage: $0 --expFolder <experiment_folder>"
    echo "This script recursively finds all config.json files in subfolders and runs RunEval.py with each config"
    exit 1
}

# 檢查參數
if [ "$#" -ne 2 ]; then
    usage
fi

if [ "$1" != "--expFolder" ]; then
    usage
fi

exp_folder="$2"

# 檢查資料夾是否存在
if [ ! -d "$exp_folder" ]; then
    echo "Error: Directory '$exp_folder' does not exist"
    exit 1
fi

# 檢查 RunEval.py 是否存在
if [ ! -f "../RunEval.py" ]; then
    echo "Error: ../RunEval.py not found in current directory"
    exit 1
fi

# 計數器
total_configs=0
processed_configs=0
failed_configs=0

# 找出所有 config.json 檔案
echo "Searching for config.json files..."
config_files=$(find "$exp_folder" -type f -name "config.json")

# 計算總數
total_configs=$(echo "$config_files" | wc -l)

if [ $total_configs -eq 0 ]; then
    echo "No config.json files found in $exp_folder"
    exit 1
fi

echo "Found $total_configs config.json files"
echo "Starting evaluation..."

# 處理每個 config.json
while IFS= read -r config_file; do
    echo "Processing config: $config_file"
    if python ../RunEval.py --config "$config_file"; then
        ((processed_configs++))
        echo "Successfully processed: $config_file"
    else
        ((failed_configs++))
        echo "Failed to process: $config_file"
    fi
    echo "Progress: $processed_configs/$total_configs completed"
    echo "----------------------------------------"
done <<< "$config_files"

# 顯示執行結果摘要
echo "Evaluation completed!"
echo "Total configs processed: $processed_configs"
echo "Failed configs: $failed_configs"
echo "Success rate: $(( (processed_configs * 100) / total_configs ))%"

if [ $failed_configs -gt 0 ]; then
    exit 1
fi

exit 0