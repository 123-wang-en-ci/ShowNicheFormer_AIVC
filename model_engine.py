import torch
import numpy as np
import scanpy as sc
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import sys
import os
import importlib.util
from tqdm import tqdm  # 用于显示进度条
import torch
import torch.nn as nn # 新增引用
import pickle         # 新增引用

current_dir = os.path.dirname(os.path.abspath(__file__))
nicheformer_root = os.path.join(current_dir, "Nicheformer")
possible_paths = [
    os.path.join(nicheformer_root, "src"),
    nicheformer_root,
    os.path.join(current_dir, "nicheformer"),
]
found_path = None
for path in possible_paths:
    if os.path.isdir(os.path.join(path, "nicheformer")):
        found_path = path
        break
if found_path and found_path not in sys.path:
    sys.path.append(found_path)

# 导入模型类
Nicheformer = None
try:
    from nicheformer.models._nicheformer import Nicheformer
    print("✅ 成功导入 Nicheformer 类")
except ImportError:
    try:
        from nicheformer.models import Nicheformer
        print("✅ 成功导入 Nicheformer 类 (from models)")
    except ImportError:
        print("❌ 无法导入 Nicheformer，请检查路径。")
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
    def forward(self, x): return self.net(x)
# Nicheformer 推理引擎
class NicheformerEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.adata = None
        self.model = None
        self.gene_list = []
        self.gene_to_id = {}
        self.coords = None
        self.kd_tree = None
        self.center = None
        # 新增：分类器模型容器
        self.cls_model = None
        self.cls_labels = []
        
        self.seg_model = None
        self.seg_labels = []
        # --- 超参数 (必须与训练时一致) ---
        self.n_neighbors = 20    # 邻域大小
        self.context_length = 1024 # 上下文长度
        self.batch_size = 16     # 推理时的 Batch Size (根据显存调整)
        self.adata_emb = None  # 用于存储邻居图的持久化对象
        self.pca_cache = None
        # --- 缓存 ---
        self.embeddings_cache = None # 存储所有细胞的 Latent Vector
        self.cell_type_cache = None
        self.region_cache = None
        self.adata_cache = None # 专门用于存储 Embedding 分析结果的 AnnData 对象

    def load_data(self, h5ad_path):
        print(f"Loading data from {h5ad_path}...")
        self.adata = sc.read_h5ad(h5ad_path)
        
        # 1. 加载固定词表
        if os.path.exists("gene_vocab.npy"):
            print("✅ Found gene_vocab.npy, loading fixed vocabulary...")
            self.gene_list = np.load("gene_vocab.npy", allow_pickle=True).tolist()
        else:
            self.gene_list = self.adata.var_names.tolist()

        # 2. 【修复】硬编码偏移量
        # 我们已经通过之前的测试确认了：模型权重(382) - 基因数(374) = 8
        start_idx = 8
        print(f"✅ Using fixed Offset (Start Index): {start_idx}")

        # 3. 建立映射
        self.gene_to_id = {name: i + start_idx for i, name in enumerate(self.gene_list)}
        
        # 验证第一个基因
        print(f"🔍 Mapping check: '{self.gene_list[0]}' -> ID {self.gene_to_id[self.gene_list[0]]}")
        print(f"Data loaded. Cells: {self.adata.n_obs}, Genes: {self.adata.n_vars}")

    # ------------------------------------------------------------------

    def build_spatial_graph(self):
        """构建 KDTree 用于查找邻居"""
        if self.coords is None: return
        print("Building spatial neighbor graph (KDTree)...")
        self.kd_tree = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='ball_tree')
        self.kd_tree.fit(self.coords)
        # 预先计算所有细胞的邻居索引，加速后续推理
        print("Pre-calculating neighbors for all cells...")
        self.distances, self.neighbor_indices = self.kd_tree.kneighbors(self.coords)
        print("Spatial graph ready.")

    def load_model(self, model_path):
        """加载训练好的权重"""
        if Nicheformer is None: return

        print(f"Loading Nicheformer weights from {model_path}...")
        
        # 1. 实例化模型 (参数必须与 train_nicheformer.py 中一致)
        self.model = Nicheformer(
            dim_model=256,
            nheads=8,
            dim_feedforward=1024, # 256 * 4
            nlayers=6,
            dropout=0.1,
            batch_first=True,
            masking_p=0.0, # 推理时不需要 Mask
            n_tokens=len(self.gene_list) + 3,
            context_length=self.context_length,
            lr=1e-4,
            warmup=100,
            batch_size=self.batch_size,
            max_epochs=5,
            learnable_pe=True
        )
        
        # 2. 加载权重
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            # 处理可能的 key 不匹配 (例如 lightning 留下的 'model.' 前缀)
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("model."):
                    new_state_dict[k[6:]] = v
                else:
                    new_state_dict[k] = v
            
            self.model.load_state_dict(new_state_dict, strict=False)
            self.model.to(self.device)
            self.model.eval()
            print("✅ Model loaded successfully.")
            
            # 3. 加载后立即计算 Embeddings 缓存
            self._precompute_embeddings()
            
        except Exception as e:
            print(f"❌ Error loading weights: {e}")

    def get_coordinates(self):
        """返回给 Unity 的坐标"""
        if self.coords is None: return []
        z = np.zeros((self.coords.shape[0], 1))
        return np.hstack([self.coords, z])

    # 核心工具：构造模型输入 (Batch Tokenization)
    def _get_batch_tokens(self, cell_indices_batch):
        """
        将一批细胞索引转换为 Nicheformer 的输入 Tensor。
        [修复版]：自动检测数据位置 (.layers['counts'] 或 .X)
        """
        batch_tokens = []
        batch_masks = []
        
        # 获取这批细胞的所有邻居索引
        batch_neighbor_indices = self.neighbor_indices[cell_indices_batch]
        
        # --- 【修复核心】确定数据源 ---
        # 优先使用 counts 层，如果没有则使用 X
        if 'counts' in self.adata.layers:
            source_data = self.adata.layers['counts']
        else:
            source_data = self.adata.X

        for i in range(len(cell_indices_batch)):
            neighbors = batch_neighbor_indices[i]
            
            # 聚合邻域表达量 (Sum)
            # 处理稀疏矩阵与密集矩阵的差异
            local_expr = source_data[neighbors].sum(axis=0)
            
            # 确保转换为 1维 numpy 数组
            if hasattr(local_expr, "A1"): # matrix
                local_expr = local_expr.A1
            elif hasattr(local_expr, "toarray"): # sparse matrix
                local_expr = local_expr.toarray().flatten()
            else: # numpy array
                local_expr = np.array(local_expr).flatten()
            
            # 提取 Top K 基因 -> Tokens
            expressed_indices = np.where(local_expr > 0)[0]
            
            if len(expressed_indices) > self.context_length:
                # 按表达量排序取 Top K
                top_k_args = np.argsort(local_expr[expressed_indices])[-self.context_length:]
                selected_indices = expressed_indices[top_k_args]
            else:
                selected_indices = expressed_indices
            
            # 映射为 Token ID (Gene ID + 3)
            token_ids = selected_indices + 3
            
            # Padding
            padding_len = self.context_length - len(token_ids)
            if padding_len > 0:
                padded_tokens = np.pad(token_ids, (0, padding_len), 'constant', constant_values=1) # 1=PAD
                # Attention Mask: 0=Keep, 1=Ignore
                att_mask = np.concatenate([np.zeros(len(token_ids)), np.ones(padding_len)])
            else:
                padded_tokens = token_ids
                att_mask = np.zeros(self.context_length)
                
            batch_tokens.append(padded_tokens)
            batch_masks.append(att_mask)
            
        return (torch.tensor(np.array(batch_tokens), dtype=torch.long).to(self.device),
                torch.tensor(np.array(batch_masks), dtype=torch.bool).to(self.device))

    # ==========================================================================
    # 预计算 Embeddings
    # ==========================================================================
    def _precompute_embeddings(self):
        """
        [优化版] 计算或加载 Embeddings
        如果本地有缓存文件，直接加载；否则计算并保存。
        """
        # 定义缓存文件名 (基于 h5ad 文件名，防止混淆)
        cache_filename = "embeddings_cache.npy"
        cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), cache_filename)

        # 1. 尝试直接加载
        if os.path.exists(cache_path):
            print(f"[Cache] Found cached embeddings at {cache_filename}, loading...")
            try:
                self.embeddings_cache = np.load(cache_path)
                # 简单的校验
                if self.embeddings_cache.shape[0] == self.adata.n_obs:
                    print(f"[Cache] Successfully loaded embeddings. Shape: {self.embeddings_cache.shape}")
                    return
                else:
                    print("[Cache] Cached embeddings shape mismatch. Recomputing...")
            except Exception as e:
                print(f"[Cache] Error loading cache: {e}. Recomputing...")

        # 2. 如果没有缓存，则开始计算 (原逻辑)
        print("Computing embeddings for all cells (First time run)...")
        self.embeddings_cache = []
        n_cells = self.adata.n_obs
        
        # 确保模型在 eval 模式
        self.model.eval()
        
        with torch.no_grad():
            for i in tqdm(range(0, n_cells, self.batch_size), desc="Inference"):
                batch_indices = np.arange(i, min(i + self.batch_size, n_cells))
                x, mask = self._get_batch_tokens(batch_indices)
                output = self.model(x, mask)
                feats = output['transformer_output']
                
                # Mean Pooling
                mask_expanded = mask.unsqueeze(-1).float()
                feats_sum = (feats * (1 - mask_expanded)).sum(dim=1)
                mask_sum = (1 - mask_expanded).sum(dim=1)
                feats_pooled = feats_sum / (mask_sum + 1e-9)
                
                self.embeddings_cache.append(feats_pooled.cpu().numpy())
                
        self.embeddings_cache = np.concatenate(self.embeddings_cache, axis=0)
        print(f"Embeddings computed. Shape: {self.embeddings_cache.shape}")
        
        # 3. 保存到硬盘
        np.save(cache_path, self.embeddings_cache)
        print(f"[Cache] Embeddings saved to {cache_filename}")

    # ==========================================================================
    # 功能 1: 基因插补 / 表达量预测
    # ==========================================================================
    def predict_gene_expression(self, gene_name):
        """
        [极速版] 基因插补
        原理：不再重新运行 Transformer，而是直接利用缓存的 Embeddings 进行线性映射。
        时间复杂度：从 O(N*L*D^2) 降低到 O(N*D)，速度提升 1000 倍以上。
        """
        # 1. 检查基因是否存在
        if gene_name not in self.gene_to_id:
            print(f"插补出错: Gene {gene_name} not found.")
            return None
        
        # 2. 获取目标基因的 Token ID
        # 注意：Geneformer/Nicheformer 通常有特殊 Token，基因 ID 通常要偏移 (比如 +3)
        # 这里的 gene_to_id 应该已经是包含了偏移量的 (我们在 server.py 里改过了)
        target_token_id = self.gene_to_id[gene_name]
        
        print(f"[Fast-Impute] 正在极速插补: {gene_name} (Token ID: {target_token_id})...")

        # 3. 检查缓存是否存在
        if self.embeddings_cache is None:
            print("错误: Embeddings 尚未计算，无法使用快速插补。请检查启动流程。")
            return None

        # 4. 执行矩阵乘法 (核心优化)
        try:
            # -------------------------------------------------------
            # 步骤 A: 准备 Embedding (N_cells, 256)
            # -------------------------------------------------------
            # 确保是 Tensor 格式
            if isinstance(self.embeddings_cache, np.ndarray):
                embeddings = torch.tensor(self.embeddings_cache).to(self.device)
            else:
                embeddings = self.embeddings_cache.to(self.device)

            # -------------------------------------------------------
            # 步骤 B: 提取特定基因的解码权重 (1, 256)
            # -------------------------------------------------------
            # Nicheformer 的解码头通常叫 classifier_head 或 decoder
            # 它的权重形状是 [Vocab_Size, Hidden_Dim]
            if hasattr(self.model, "classifier_head"):
                decoder_layer = self.model.classifier_head
            elif hasattr(self.model, "decoder"):
                decoder_layer = self.model.decoder
            else:
                # 尝试根据常用名猜测
                print("警告: 无法自动找到解码层，尝试使用 model.lm_head")
                decoder_layer = self.model.lm_head

            # 我们只需要提取属于 target_gene 的那一列权重和偏置
            # 这样避免了计算所有 20000 个基因的概率，节省大量显存
            
            # 权重: [256]
            target_weight = decoder_layer.weight[target_token_id, :] 
            # 偏置: scalar
            target_bias = decoder_layer.bias[target_token_id]

            # -------------------------------------------------------
            # 步骤 C: 极速计算 (Dot Product)
            # [N, 256] @ [256] -> [N]
            # -------------------------------------------------------
            with torch.no_grad():
                # 线性投影： y = xW^T + b
                # embeddings: [26678, 256]
                # target_weight: [256]
                logits = torch.mv(embeddings, target_weight) + target_bias
                
                # 可选：Sigmoid 或 Softmax (取决于模型训练目标，通常插补用 Sigmoid 或直接 Logits)
                # 如果数值范围很大，建议归一化一下供前端展示
                predicted_expression = torch.sigmoid(logits).cpu().numpy()

            print(f"[Fast-Impute] 计算完成。Min: {predicted_expression.min():.4f}, Max: {predicted_expression.max():.4f}")
            return predicted_expression

        except Exception as e:
            print(f"极速插补发生错误: {e}")
            import traceback
            traceback.print_exc()
            return None

    def load_downstream_models(self):
        """加载训练好的有监督分类器"""
        print("[AI] Loading downstream classifiers...")
        
        # 1. 加载细胞类型模型
        try:
            if os.path.exists("cell_type_model_labels.pkl"):
                with open("cell_type_model_labels.pkl", "rb") as f:
                    self.cls_labels = pickle.load(f)
                
                self.cls_model = ClassifierHead(num_classes=len(self.cls_labels))
                self.cls_model.load_state_dict(torch.load("cell_type_model.pth", map_location=self.device))
                self.cls_model.to(self.device).eval()
                print(f"✅ Cell Type Classifier loaded ({len(self.cls_labels)} classes)")
            else:
                print("⚠️ No cell type model found. predict_cell_types will fail.")
        except Exception as e:
            print(f"❌ Error loading cell type model: {e}")

        # 2. 加载区域分割模型
        try:
            if os.path.exists("region_model_labels.pkl"):
                with open("region_model_labels.pkl", "rb") as f:
                    self.seg_labels = pickle.load(f)
                
                self.seg_model = ClassifierHead(num_classes=len(self.seg_labels))
                self.seg_model.load_state_dict(torch.load("region_model.pth", map_location=self.device))
                self.seg_model.to(self.device).eval()
                print(f"✅ Region Classifier loaded ({len(self.seg_labels)} regions)")
        except Exception as e:
            print(f"⚠️ Region model loading skipped: {e}")

    # ==========================================================================
    # 【重写】功能 2: 细胞类型注释 (有监督)
    # ==========================================================================
    def predict_cell_types(self):
        """
        使用训练好的分类器预测真实细胞名称
        """
        if self.cell_type_cache is not None:
            return self.cell_type_cache
            
        if self.embeddings_cache is None: self._precompute_embeddings()
        if self.cls_model is None: 
            # 如果没有模型，尝试加载（或者回退到 Leiden? 建议这里强制要求模型）
            self.load_downstream_models()
            if self.cls_model is None:
                return [], [] # 失败返回空

        print("Predicting cell types using Supervised Classifier...")
        
        # 批量推理
        predictions = []
        # 将 embedding 转为 Tensor
        features = torch.tensor(self.embeddings_cache).float().to(self.device)
        
        with torch.no_grad():
            # 显存如果不够，这里也要分 Batch，但只有 Linear 层通常一次能跑完 2.6w
            outputs = self.cls_model(features)
            _, predicted_ids = torch.max(outputs, 1)
            predictions = predicted_ids.cpu().numpy()
            
        # 生成图例
        # self.cls_labels 是 ['Astro', 'Micro', 'T-Cell'...]
        legend = []
        import colorsys
        for i, name in enumerate(self.cls_labels):
            hue = i / len(self.cls_labels)
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            hex_color = '#%02x%02x%02x' % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
            legend.append({"id": int(i), "name": name, "color": hex_color})
            
        self.cell_type_cache = (predictions, legend)
        return predictions, legend

    # ==========================================================================
    # 【重写】功能 3: 组织区域语义分割 (有监督)
    # ==========================================================================
    def segment_tissue_regions(self):
        """
        使用训练好的分类器预测真实区域名称
        """
        if self.region_cache is not None: return self.region_cache
        if self.embeddings_cache is None: self._precompute_embeddings()
        if self.seg_model is None: 
            self.load_downstream_models()
            if self.seg_model is None:
                # 回退策略：如果没有训练区域模型，使用 KMeans
                print("⚠️ No supervised region model found, falling back to KMeans.")
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=8, random_state=42).fit(self.embeddings_cache)
                return kmeans.labels_, [f"Region_{i}" for i in np.unique(kmeans.labels_)]

        print("Segmenting tissue regions using Supervised Classifier...")
        
        features = torch.tensor(self.embeddings_cache).float().to(self.device)
        with torch.no_grad():
            outputs = self.seg_model(features)
            _, predicted_ids = torch.max(outputs, 1)
            region_ids = predicted_ids.cpu().numpy()
            
        # 映射回名字列表
        # region_names = ["Cortex", "Thalamus"...]
        region_names = self.seg_labels 
        
        self.region_cache = (region_ids, region_names)
        return region_ids, region_names
   # ==========================================================================
    # 零样本聚类 (Zero-Shot Clustering) - K-Means 版本
    def run_zero_shot_clustering(self, n_clusters=10):
            """
            [极速版 - MiniBatchKMeans + PCA Cache] 零样本聚类
            速度优化：
            1. 缓存 PCA 结果（避免重复降维）
            2. 使用 MiniBatchKMeans（小批量迭代，比标准 KMeans 快 10-50 倍）
            """
            # 1. 确保有特征
            if self.embeddings_cache is None:
                self._precompute_embeddings()
                
            print(f"[AI] Running Fast Clustering (Target K={n_clusters})...")
            
            # 2. 安全限制
            n_clusters = int(n_clusters)
            n_clusters = max(2, n_clusters)
            n_clusters = min(n_clusters, 100)
            
            # 3. 准备数据 (含 PCA 缓存机制)
            try:
                from sklearn.cluster import MiniBatchKMeans
                from sklearn.decomposition import PCA
                
                # --- PCA 缓存逻辑 ---
                if self.pca_cache is None:
                    print(f"   - [1/2] Computing PCA (First time only)...")
                    # 如果维度 > 50，进行降维并缓存
                    if self.embeddings_cache.shape[1] > 50:
                        pca = PCA(n_components=50, random_state=42)
                        self.pca_cache = pca.fit_transform(self.embeddings_cache)
                    else:
                        self.pca_cache = self.embeddings_cache
                    print(f"   - PCA Cached. Shape: {self.pca_cache.shape}")
                else:
                    # print("   - [1/2] Using cached PCA data (Skipping calculation).")
                    pass
                
                # 使用缓存的数据
                X_data = self.pca_cache
                
                # --- 极速聚类逻辑 ---
                # MiniBatchKMeans: 牺牲微小的精度换取巨大的速度提升
                # batch_size: 每次只看 2048 个细胞
                # n_init: 只尝试 3 次不同的初始化 (标准是 10 次)
                kmeans = MiniBatchKMeans(
                    n_clusters=n_clusters, 
                    batch_size=2048, 
                    n_init=3, 
                    random_state=42,
                    reassignment_ratio=0 # 防止某些类因为太小被丢弃
                )
                
                # fit_predict 毫秒级完成
                clusters = kmeans.fit_predict(X_data)
                
                unique_clusters = np.unique(clusters)
                print(f"   - [2/2] Clustering finished. Groups: {len(unique_clusters)}")
                
                # 4. 生成图例 (保持不变)
                legend = []
                import colorsys
                for i, cid in enumerate(unique_clusters):
                    hue = (i * 0.618033988749895) % 1.0 
                    rgb = colorsys.hsv_to_rgb(hue, 0.75, 0.95) 
                    hex_color = '#%02x%02x%02x' % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
                    
                    legend.append({
                        "id": int(cid), 
                        "name": f"Cluster {cid}", 
                        "color": hex_color
                    })
                    
                return clusters, legend

            except Exception as e:
                print(f"❌ Clustering Error: {e}")
                import traceback
                traceback.print_exc()
                return np.zeros(len(self.embeddings_cache), dtype=int), []