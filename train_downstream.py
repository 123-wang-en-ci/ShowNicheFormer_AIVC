import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from model_engine import NicheformerEngine  # 复用你现在的引擎来提取特征

# ================= 配置 =================
# H5AD 文件中存储真实标签的列名 (请根据你的数据修改!)
CELL_TYPE_COL = "cell_type"  
REGION_COL = "clust_annot"        # 如果没有区域标签，可以设为 None

# 训练参数
BATCH_SIZE = 64
EPOCHS = 50
LR = 0.001
# =======================================

class ClassifierHead(nn.Module):
    def __init__(self, input_dim=256, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

def train_classifier(features, labels, save_name, device):
    print(f"\n🚀 开始训练分类器: {save_name}")
    
    # 1. 标签编码 (String -> Int)
    le = LabelEncoder()
    targets = le.fit_transform(labels)
    num_classes = len(le.classes_)
    print(f"检测到 {num_classes} 个类别: {le.classes_[:num_classes+1]}...")
    
    # 保存 LabelEncoder (推理时要把 Int 转回 String)
    with open(f"{save_name}_labels.pkl", "wb") as f:
        pickle.dump(le.classes_.tolist(), f)
    
    # 2. 准备数据
    X_train, X_val, y_train, y_val = train_test_split(features, targets, test_size=0.2, random_state=42)
    
    train_ds = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long())
    val_ds = TensorDataset(torch.tensor(X_val).float(), torch.tensor(y_val).long())
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    
    # 3. 初始化模型
    model = ClassifierHead(input_dim=256, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # 4. 训练循环
    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        acc = 100 * correct / total
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Val Acc = {acc:.2f}%")
            
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"{save_name}.pth")
            
    print(f"✅ 训练完成。最佳准确率: {best_acc:.2f}%。模型已保存至 {save_name}.pth")

def main():
    # 1. 初始化引擎并加载数据
    engine = NicheformerEngine()
    engine.load_data("train.h5ad") # 请改为你的文件名
    engine.build_spatial_graph()
    engine.load_model("nicheformer_weights.pth") # 确保这个文件存在
    
    # 2. 获取/计算 Embeddings (这是最重要的特征)
    # 如果已有缓存会自动加载，没有会计算
    engine._precompute_embeddings() 
    embeddings = engine.embeddings_cache
    
    if CELL_TYPE_COL in engine.adata.obs:
        print(f"\n正在处理细胞类型数据 ({CELL_TYPE_COL})...")
        
        # 定义不需要的垃圾标签列表 (根据你的观察添加)
        INVALID_LABELS = ['cell', 'Unknown', 'nan', 'N/A']
        
        # 获取原始标签列
        raw_labels = engine.adata.obs[CELL_TYPE_COL].astype(str)
        
        # 构建过滤掩码：既不是 NaN，也不在垃圾列表中
        # ~ 表示“非”，isin 表示“在列表中”
        valid_mask = (engine.adata.obs[CELL_TYPE_COL].notna()) & \
                     (~raw_labels.isin(INVALID_LABELS))
        
        # 统计一下过滤了多少
        n_total = len(raw_labels)
        n_keep = valid_mask.sum()
        print(f"原始细胞数: {n_total}, 过滤后: {n_keep} (剔除了 {n_total - n_keep} 个模糊细胞)")

        if n_keep > 0:
            features = embeddings[valid_mask]
            labels = raw_labels[valid_mask].values
            
            train_classifier(features, labels, "cell_type_model", engine.device)
        else:
            print("错误: 过滤后没有剩余细胞，请检查过滤条件！")

    # 4. 训练区域分割分类器
    if REGION_COL and REGION_COL in engine.adata.obs:
        print("\n正在准备区域数据...")
        valid_mask = engine.adata.obs[REGION_COL].notna()
        features = embeddings[valid_mask]
        labels = engine.adata.obs[REGION_COL][valid_mask].values.astype(str)
        
        train_classifier(features, labels, "region_model", engine.device)
    else:
        print(f"⚠️ 跳过区域分割训练 (列 '{REGION_COL}' 不存在)。")

if __name__ == "__main__":
    main()