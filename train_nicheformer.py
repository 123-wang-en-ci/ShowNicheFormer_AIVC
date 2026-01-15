import os
import sys
import torch
import numpy as np
import scanpy as sc
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import NearestNeighbors
import warnings

# ==============================================================================
# 0. 显存优化设置 (关键修复)
# ==============================================================================
# 释放未使用的显存
torch.cuda.empty_cache()
# 降低矩阵乘法精度以节省显存并加速 (针对 30系显卡)
torch.set_float32_matmul_precision('medium')

# ==============================================================================
# 1. 环境与路径设置
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
nicheformer_src = os.path.join(current_dir, "Nicheformer", "src")
if nicheformer_src not in sys.path:
    sys.path.append(nicheformer_src)

try:
    from nicheformer.models._nicheformer import Nicheformer
except ImportError:
    try:
        from model_engine import Nicheformer
    except ImportError:
        pass

# ==============================================================================
# 2. 缺失工具函数补全 (Mock Utils)
# ==============================================================================
def complete_masking(batch, masking_p, n_tokens):
    x = batch['X'].clone()
    batch_size, seq_len = x.shape
    probability_matrix = torch.full(x.shape, masking_p).to(x.device)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    mask_token_id = 0 
    labels = x.clone()
    
    indices_replaced = torch.bernoulli(torch.full(x.shape, 0.8)).bool().to(x.device) & masked_indices
    x[indices_replaced] = mask_token_id

    indices_random = torch.bernoulli(torch.full(x.shape, 0.5)).bool().to(x.device) & masked_indices & ~indices_replaced
    random_tokens = torch.randint(1, n_tokens, x.shape, dtype=torch.long).to(x.device)
    x[indices_random] = random_tokens[indices_random]

    new_batch = {
        'masked_indices': x,
        'mask': x, 
        'X': labels,
        'attention_mask': batch.get('attention_mask', None)
    }
    mask_for_loss = torch.ones_like(x)
    mask_for_loss[masked_indices] = 0
    new_batch['mask'] = mask_for_loss
    
    for k, v in batch.items():
        if k not in new_batch:
            new_batch[k] = v
    return new_batch

import types
if 'nicheformer.models._utils' not in sys.modules:
    mod = types.ModuleType('nicheformer.models._utils')
    mod.complete_masking = complete_masking
    sys.modules['nicheformer.models._utils'] = mod

# ==============================================================================
# 3. 数据集构建
# ==============================================================================
class SpatialNicheDataset(Dataset):
    def __init__(self, h5ad_path, context_length=1024, n_neighbors=20):
        print(f"正在加载数据: {h5ad_path}...")
        self.adata = sc.read_h5ad(h5ad_path)
        self.context_length = context_length
        self.n_neighbors = n_neighbors
        
        if 'counts' not in self.adata.layers:
            self.adata.layers['counts'] = self.adata.X.copy()
        
        self.gene_names = self.adata.var_names.tolist()
        self.gene_to_id = {gene: i + 3 for i, gene in enumerate(self.gene_names)}
        self.n_tokens = len(self.gene_names) + 3
        
        print("构建空间邻域图...")
        if 'spatial' in self.adata.obsm:
            coords = self.adata.obsm['spatial']
        else:
            coords = self.adata.X[:, :2]
            
        self.nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree').fit(coords)
        self.distances, self.indices = self.nbrs.kneighbors(coords)
        print(f"数据集就绪。细胞数: {self.adata.n_obs}, 基因数: {len(self.gene_names)}")

    def __len__(self):
        return self.adata.n_obs

    def __getitem__(self, idx):
        neighbor_indices = self.indices[idx]
        local_expression = self.adata.layers['counts'][neighbor_indices].sum(axis=0)
        
        if hasattr(local_expression, "A1"):
            local_expression = local_expression.A1
        else:
            local_expression = np.array(local_expression).flatten()
            
        expressed_gene_indices = np.where(local_expression > 0)[0]
        
        if len(expressed_gene_indices) > self.context_length:
            top_indices = np.argsort(local_expression[expressed_gene_indices])[-self.context_length:]
            selected_gene_indices = expressed_gene_indices[top_indices]
        else:
            selected_gene_indices = expressed_gene_indices

        token_ids = selected_gene_indices + 3
        
        padding_len = self.context_length - len(token_ids)
        if padding_len > 0:
            token_ids = np.pad(token_ids, (0, padding_len), 'constant', constant_values=1)
            attention_mask = np.concatenate([np.zeros(len(selected_gene_indices)), np.ones(padding_len)])
        else:
            attention_mask = np.zeros(self.context_length)

        return {
            'X': torch.tensor(token_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.bool),
            'cell_type': 0
        }

# ==============================================================================
# 4. 训练主流程
# ==============================================================================
def train():
    # --- 配置参数 (已针对 3050Ti 4GB 优化) ---
    H5AD_PATH = "train.h5ad" 
    OUTPUT_PATH = "nicheformer_weights.pth"
    
    # 【显存优化 1】降低 Batch Size
    # 既然之前报错显存不足，我们这里极端一点，设为 2
    # 如果还是报错，请改为 1
    BATCH_SIZE = 2  
    
    MAX_EPOCHS = 100
    LR = 1e-4
    
    # 模型超参数
    CONTEXT_LENGTH = 1024 
    DIM_MODEL = 256
    N_HEADS = 8
    N_LAYERS = 6
    
    if not os.path.exists(H5AD_PATH):
        print(f"❌ 错误: 找不到数据文件 {H5AD_PATH}")
        return

    dataset = SpatialNicheDataset(H5AD_PATH, context_length=CONTEXT_LENGTH)
    
    # num_workers=0 避免 Windows 多进程开销和报错
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    
    print("初始化模型...")
    model = Nicheformer(
        dim_model=DIM_MODEL,
        nheads=N_HEADS,
        dim_feedforward=DIM_MODEL * 4,
        nlayers=N_LAYERS,
        dropout=0.1,
        batch_first=True,
        masking_p=0.15,
        n_tokens=dataset.n_tokens,
        context_length=CONTEXT_LENGTH,
        lr=LR,
        warmup=100,
        batch_size=BATCH_SIZE,
        max_epochs=MAX_EPOCHS,
        learnable_pe=True
    )
    
    # 检查是否有 GPU
    use_gpu = torch.cuda.is_available()
    accelerator = "gpu" if use_gpu else "cpu"
    devices = 1 if use_gpu else "auto"
    
    # 【修正点】根据你的 Lightning 版本自动选择 precision 写法
    # 旧版本只接受 16，新版本接受 "16-mixed"
    precision_val = 16 if use_gpu else 32

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='train_loss',
        dirpath='checkpoints',
        filename='nicheformer-{epoch:02d}-{train_loss:.2f}',
        save_top_k=1,
        mode='min'
    )
    
    print(f"配置 Trainer (Accelerator: {accelerator}, Precision: {precision_val})...")
    
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=MAX_EPOCHS,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
        enable_progress_bar=True,
        # 【关键修正】这里改为 16 (整数)，兼容旧版 Lightning
        precision=precision_val
    )
    
    print(f"开始训练 (Batch Size: {BATCH_SIZE})...")
    trainer.fit(model, dataloader)
    
    print(f"训练完成，正在保存权重到 {OUTPUT_PATH} ...")
    state_dict = model.state_dict()
    clean_state_dict = {k: v for k, v in state_dict.items() if "loss" not in k}
    torch.save(clean_state_dict, OUTPUT_PATH)
    print("✅ 权重保存成功！")
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    train()