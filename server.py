# server.py
from fastapi.responses import JSONResponse
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import scanpy as sc
import pandas as pd
import numpy as np
import torch
from scipy.spatial import KDTree
from scipy.sparse import issparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD 
import os
import datetime
import sys
from contextlib import asynccontextmanager
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_engine import NicheformerEngine 

# H5AD_FILENAME = "Allen2022Molecular_aging_MsBrainAgingSpatialDonor_10_0.h5ad" 
# H5AD_FILENAME = "Allen2022Molecular_lps_MsBrainAgingSpatialDonor_14_1.h5ad" 
H5AD_FILENAME = "train.h5ad" 
# H5AD_FILENAME = "data.h5ad" 
CSV_FILENAME = "unity_cell_data.csv"
CELL_TYPE_COLUMN = "cell_type" 

# Nicheformer 模型路径 (请确保文件存在，或修改为你实际的 .pth 路径)
NICHEFORMER_MODEL_PATH = "nicheformer_weights.pth" 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 请求体定义 (保持完全一致) ---
class GeneRequest(BaseModel):
    gene_name: str
    use_imputation: bool = False 

class PerturbRequest(BaseModel):
    target_id: str
    perturb_type: str = "KO"
    target_gene: str = "ENSMUSG00000037010"

class ClusteringRequest(BaseModel):
    n_clusters: int = 10

# 数据管理类 (逻辑核心)
class DataManager:
    def __init__(self):
        self.adata = None
        self.spatial_tree = None
        self.coords = None
        self.indices_map = None
        self.scaler = MinMaxScaler()
        
        self.cached_total_counts = None
        # self.cached_features = None # Nicheformer 内部处理特征，不再需要显式缓存 SVD
        self.current_view_gene = "RESET"
        
        self.base_dir = os.path.dirname(os.path.abspath(__file__))

        # 【核心修改】初始化 Nicheformer 引擎，而非 Geneformer
        print("[System] 初始化 Nicheformer 引擎...")
        self.ai_engine = NicheformerEngine() 
        
        # 这里的 model_dir 如果你需要，可以指向 Nicheformer 的权重文件夹
        self.model_path = os.path.join(self.base_dir, NICHEFORMER_MODEL_PATH)

    def load_and_sync_data(self):
        print(f"[后端] 加载数据: {H5AD_FILENAME} ...")
        h5ad_path = os.path.join(self.base_dir, H5AD_FILENAME)

        if not os.path.exists(h5ad_path):
            print(f"Error: File not found {h5ad_path}")
            return

        # 1. 加载 Scanpy 数据
        self.adata = sc.read_h5ad(h5ad_path)
        self.h5ad_path = h5ad_path
        # =========================================================
        # 【新增调试代码】打印基因信息，排查 "Not Found" 问题
        # =========================================================
        print("\n" + "="*40)
        print("[调试] 基因索引检查")
        print(f"总基因数: {self.adata.n_vars}")
        
        # 1. 打印前 10 个基因名 (看看是 Gene Symbol 还是 Ensembl ID)
        top_10_genes = self.adata.var_names[:10].tolist()
        print(f"示例基因 (Index): {top_10_genes}")
        
        # 2. 检查报错的那个特定 ID 是否存在
        target_debug_id = "ENSMUSG00000037010"
        if target_debug_id in self.adata.var_names:
            print(f"✅ 目标基因 {target_debug_id} 在索引中存在！")
        else:
            print(f"❌ 目标基因 {target_debug_id} 不在索引中！")
            
            # 3. 尝试在其他列里寻找 (有时候 ID 藏在 var 的某一列里)
            found_in_col = False
            for col in self.adata.var.columns:
                # 检查这一列是否包含该 ID
                if self.adata.var[col].astype(str).str.contains(target_debug_id).any():
                    print(f"⚠️ 发现 {target_debug_id} 存在于列 '{col}' 中，而不是索引中。")
                    print(f"   (前端发送的是ID，但模型目前使用 '{col}' 之外的索引)")
                    found_in_col = True
            
            if not found_in_col:
                print(f"❌ 在整个数据表中都找不到 {target_debug_id}，可能是数据过滤掉了该基因。")
        print("="*40 + "\n")
        # =========================================================

        # 【新增修复】确保 layers['counts'] 存在，防止 Nicheformer 报错
        if 'counts' not in self.adata.layers:
            self.adata.layers['counts'] = self.adata.X.copy()

        # --- 将数据同步给 Nicheformer 引擎 ---
        print("[Nicheformer] 同步数据到 AI 引擎...")
        self.ai_engine.adata = self.adata
        self.ai_engine.gene_list = self.adata.var_names.tolist()
        
        # 【核心修复】变量名必须是 gene_to_id，且 ID 偏移量要与 model_engine 一致 (i + 3)
        self.ai_engine.gene_to_id = {name: i + 3 for i, name in enumerate(self.ai_engine.gene_list)}
        
        # 打印一下验证是否注入成功
        print(f"[调试] 引擎映射表已构建，包含 {len(self.ai_engine.gene_to_id)} 个基因。")
        # -----------------------------------

        # 2. 处理坐标
        if 'spatial' in self.adata.obsm:
            self.coords = self.adata.obsm['spatial']
        else:
            self.coords = self.adata.X[:, :2] if self.adata.X.shape[1] >=2 else np.zeros((self.adata.n_obs, 2))

        if issparse(self.coords): self.coords = self.coords.toarray()
        if not isinstance(self.coords, np.ndarray): self.coords = np.array(self.coords)
        
        # 中心化坐标 (供 Unity 使用)
        self.center = np.mean(self.coords, axis=0)
        self.coords_centered = self.coords - self.center

        # --- 同步坐标给 AI 引擎并构建图 ---
        self.ai_engine.coords = self.coords_centered # 使用中心化后的坐标
        self.ai_engine.center = np.zeros(2) # 引擎内部已经不需要再偏移了
        
        # 【关键步骤】构建 Nicheformer 所需的空间邻域图
        self.ai_engine.build_spatial_graph()
        # -----------------------------------
            
        self.spatial_tree = KDTree(self.coords_centered) # 用于简单的距离查询
        self.indices_map = {idx: i for i, idx in enumerate(self.adata.obs.index)}

        # 3. 缓存 Total Counts (用于 RESET 视图)
        if issparse(self.adata.X):
            raw_counts = np.ravel(self.adata.X.sum(axis=1))
        else:
            raw_counts = np.ravel(self.adata.X.sum(axis=1))
        self.cached_total_counts = self.scaler.fit_transform(raw_counts.reshape(-1, 1)).flatten()

        # 4. 加载 Nicheformer 模型权重
        if os.path.exists(self.model_path):
            try:
                self.ai_engine.load_model(self.model_path)
                print("[System] Nicheformer 权重加载成功。")
            except Exception as e:
                print(f"[Warning] Nicheformer 加载失败: {e}")
        else:
            print(f"[Warning] 未找到权重文件: {self.model_path}，将使用未训练模型运行(仅测试流程)。")

        print(f"[后端] 数据加载成功。Cells: {self.adata.n_obs}")

        # 5. 生成 Unity CSV
        self.export_csv_for_unity()
    def update_clusters(self, cluster_ids, legend_info):
        """
        将 AI 算出的聚类结果保存到 adata.obs 中，
        以便后续点击“保存”按钮时能写入硬盘。
        """
        if self.adata is None:
            print("[Error] DataManager: adata is None, cannot update clusters.")
            return

        try:
            # 1. 确保长度匹配
            if len(cluster_ids) != self.adata.n_obs:
                print(f"[Warning] Cluster count ({len(cluster_ids)}) != Cell count ({self.adata.n_obs})")
                return
            
            # 2. 将结果写入 obs (列名为 'zero_shot_cluster')
            self.adata.obs['zero_shot_cluster'] = cluster_ids
            
            # 强制转为分类类型 (Categorical)
            self.adata.obs['zero_shot_cluster'] = self.adata.obs['zero_shot_cluster'].astype(str).astype('category')
            
            # =========================================================
            # 【修复】H5AD 不支持直接存字典列表，必须转为 JSON 字符串
            # =========================================================
            import json
            # 将复杂的 list[dict] 转换为纯字符串存入，避免报错
            self.adata.uns['zero_shot_legend'] = json.dumps(legend_info) 
            # =========================================================
            
            print("[System] Zero-shot clusters updated in RAM.")
            
        except Exception as e:
            print(f"[Error] Failed to update clusters in DataManager: {e}")
            
        except Exception as e:
            print(f"[Error] Failed to update clusters in DataManager: {e}")
    def export_csv_for_unity(self):
        print("[Sync] 为 Unity 生成 CSV (执行坐标中心化)...")
        ids = self.adata.obs.index
        
        # 使用 load_and_sync_data 中计算好的中心化坐标
        norm_x = self.coords_centered[:, 0]
        norm_y = self.coords_centered[:, 1]
        
        expression_norm = self.cached_total_counts 

        if CELL_TYPE_COLUMN in self.adata.obs:
            cell_type_names = self.adata.obs[CELL_TYPE_COLUMN].values
            cell_type_codes, uniques = pd.factorize(cell_type_names)
        else:
            cell_type_names = ["Unknown"] * len(ids)
            cell_type_codes = [0] * len(ids)

        df_export = pd.DataFrame({
            'id': ids, 
            'x': norm_x,  
            'y': norm_y, 
            'z': 0,
            'expression_level': expression_norm,
            'cell_type_id': cell_type_codes,
            'cell_type_name': cell_type_names
        })

        unity_csv_path = os.path.join(self.base_dir, "..", "Assets", "StreamingAssets", CSV_FILENAME)
        os.makedirs(os.path.dirname(unity_csv_path), exist_ok=True)

        try:
            df_export.to_csv(unity_csv_path, index=False)
            print(f"[成功] CSV 已保存至: {unity_csv_path}")
        except Exception as e:
            print(f"[失败] CSV save error: {e}")

    # --- 核心修改：基于 Nicheformer 的插补 ---
    def impute_data(self, gene_values):
        """
        调用 Nicheformer 进行基因表达插补
        注意：gene_values 参数在这里可能用不上，因为 Nicheformer 是从隐空间重建
        """
        gene_name = self.current_view_gene
        if gene_name == "RESET": return gene_values
        
        print(f"[Nicheformer] 正在插补基因: {gene_name}")
        
        try:
            # 直接调用引擎的方法
            imputed_vals = self.ai_engine.predict_gene_expression(gene_name)
            
            # 简单的后处理/归一化，防止数值过大
            # imputed_vals = np.clip(imputed_vals, 0, None)
            return imputed_vals
        except Exception as e:
            print(f"插补出错: {e}")
            return gene_values # 回退到原始数据

    def get_gene_data(self, gene_name):
        if gene_name.upper() in ["RESET", "TOTAL", "DEFAULT", "HARD_RESET"]:
            base_values = self.cached_total_counts 
        else:
            if gene_name not in self.adata.var_names: return None
            
            if self.adata.raw is not None:
                try: vals = self.adata.raw[:, gene_name].X
                except: vals = self.adata[:, gene_name].X
            else:
                vals = self.adata[:, gene_name].X
            
            if issparse(vals): vals = vals.toarray()
            base_values = self.scaler.fit_transform(vals.reshape(-1, 1)).flatten()

        return np.clip(base_values, 0.0, 5.0)

    # --- 保存插补数据 ---
    def save_imputed_data(self, gene_name):
        if gene_name == "RESET": return None, "Cannot save RESET view"
        
        print(f"[保存] Nicheformer 插补数据 {gene_name}...")
        try:
            imputed_values = self.ai_engine.predict_gene_expression(gene_name)
            
            df_result = pd.DataFrame({
                'cell_id': self.adata.obs.index,
                'x': self.coords_centered[:, 0],
                'y': self.coords_centered[:, 1],
                f'{gene_name}_niche_imputed': imputed_values
            })

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"niche_imputed_{gene_name}_{timestamp}.csv"
            save_path = os.path.join(self.base_dir, "..", "Assets", "StreamingAssets", filename)
            
            df_result.to_csv(save_path, index=False)
            return filename, "Success"
        except Exception as e:
            return None, str(e)

    # --- 保存注释结果 ---
    def save_annotation_result(self):
        print("[保存] 保存 Nicheformer 注释结果...")
        try:
            # 1. 获取预测
            pred_ids, legend = self.ai_engine.predict_cell_types()
            
            # 2. 映射名称 (legend 是 [{'id':0, 'name':...}, ...])
            # 将 legend 转换为 ID->Name 的字典
            id_to_name = {item['id']: item['name'] for item in legend}
            predicted_names = [id_to_name.get(pid, "Unknown") for pid in pred_ids]

            data_dict = {
                'cell_id': self.adata.obs.index,
                'predicted_type_id': pred_ids,
                'predicted_type_name': predicted_names
            }
            
            df_result = pd.DataFrame(data_dict)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"niche_annotation_{timestamp}.csv"
            save_path = os.path.join(self.base_dir, "..", "Assets", "StreamingAssets", filename)
            
            df_result.to_csv(save_path, index=False)
            return filename, "Success"
        except Exception as e:
            return None, str(e)

    # --- 保存区域分割结果 ---
    def save_region_result(self):
        print("[保存] 保存组织区域分割结果...")
        try:
            region_ids, region_names = self.ai_engine.segment_tissue_regions()
            
            # region_names 是列表 ["Region_0", "Region_1"...]
            predicted_region_names = [region_names[rid] for rid in region_ids]

            data_dict = {
                'cell_id': self.adata.obs.index,
                'x_coord': self.coords_centered[:, 0],
                'y_coord': self.coords_centered[:, 1],
                'region_id': region_ids,
                'region_name': predicted_region_names
            }

            df_result = pd.DataFrame(data_dict)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"niche_segmentation_{timestamp}.csv"
            save_path = os.path.join(self.base_dir, "..", "Assets", "StreamingAssets", filename)
            
            df_result.to_csv(save_path, index=False)
            return filename, "Success"
        except Exception as e:
            return None, str(e)
    def save_zero_shot_result(self):
        print("[保存] 保存零样本聚类结果为 CSV...")
        
        # 1. 检查是否有聚类数据
        if 'zero_shot_cluster' not in self.adata.obs:
            return None, "No clustering results found in memory. Please run clustering first."
            
        try:
            # 2. 获取基础数据
            cluster_ids = self.adata.obs['zero_shot_cluster'].values
            
            # 尝试解析图例信息以获取颜色和名称 (之前存入 uns 的 JSON)
            import json
            legend_json = self.adata.uns.get('zero_shot_legend', '[]')
            
            cluster_names = []
            cluster_colors = []
            
            try:
                # 解析 JSON: [{'id':0, 'name':'Cluster 0', 'color':'#aabbcc'}, ...]
                legend_list = json.loads(legend_json)
                
                # 构建映射字典
                id_to_name = {str(item['id']): item['name'] for item in legend_list}
                id_to_color = {str(item['id']): item['color'] for item in legend_list}
                
                # 映射到每个细胞
                for cid in cluster_ids:
                    cid_str = str(cid)
                    cluster_names.append(id_to_name.get(cid_str, f"Cluster {cid}"))
                    cluster_colors.append(id_to_color.get(cid_str, "#ffffff"))
            except Exception as parse_e:
                print(f"[Warning] Failed to parse legend json: {parse_e}")
                # 降级处理：直接用 ID
                cluster_names = [f"Cluster {c}" for c in cluster_ids]
                cluster_colors = ["#ffffff"] * len(cluster_ids)

            # 3. 构建 DataFrame
            data_dict = {
                'cell_id': self.adata.obs.index,
                'x_coord': self.coords_centered[:, 0],
                'y_coord': self.coords_centered[:, 1],
                'cluster_id': cluster_ids,
                'cluster_name': cluster_names,
                'cluster_color': cluster_colors
            }
            
            df_result = pd.DataFrame(data_dict)
            
            # 4. 生成文件名和路径 (保存到 Assets/StreamingAssets)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"zero_shot_clustering_{timestamp}.csv"
            
            # 路径回退两级到 Unity 的 StreamingAssets
            save_path = os.path.join(self.base_dir, "..", "Assets", "StreamingAssets", filename)
            
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 5. 保存
            df_result.to_csv(save_path, index=False)
            print(f"[Success] CSV Saved to: {save_path}")
            
            return filename, "Success"
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, str(e)


# 全局 DataManager 实例
dm = DataManager()



@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- 启动阶段 ---
    print("[LifeSpan] Server starting up...")
    dm.load_and_sync_data()
    
    # 【新增】加载有监督分类器
    dm.ai_engine.load_downstream_models()
    
    # 预热任务
    print("[LifeSpan] Pre-calculating downstream tasks (Warming up)...")
    try:
        dm.ai_engine.predict_cell_types()
        dm.ai_engine.segment_tissue_regions()
        print("[LifeSpan] Tasks ready.")
    except Exception as e:
        print(f"[LifeSpan] Warm-up warning: {e}")

    yield
    print("[LifeSpan] Server shutting down...")

# 初始化 APP 时注入 lifespan
app = FastAPI(lifespan=lifespan)

@app.post("/switch_gene")
async def switch_gene(req: GeneRequest):
    if dm.adata is None: raise HTTPException(500, "Data not loaded")
    
    target_gene = req.gene_name
    if target_gene in ["HARD_RESET", "RESET", "TOTAL"]:
        target_gene = "RESET"
    
    dm.current_view_gene = target_gene
    values = dm.get_gene_data(target_gene) # 获取原始值作为基准
    
    if values is None: return {"status": "error", "message": "Gene not found"}
    
    # 默认消息
    msg = "View Switched"

    # 【核心逻辑】如果请求了插补 (AI Imputation)
    if req.use_imputation and target_gene != "RESET":
        # 调用 Nicheformer 逻辑
        values = dm.impute_data(values)
        # 必须保留这个消息字符串，前端靠它识别
        msg = f"AI Imputation : {target_gene}"
    
    updates = []
    ids = dm.adata.obs.index
    # 转换为 JSON 友好的格式
    # 假设 values 已经是 numpy array
    vals_list = values.tolist() if isinstance(values, np.ndarray) else values
    
    for i, val in enumerate(vals_list):
        updates.append({"id": str(ids[i]), "new_expr": round(float(val), 3)})
        
    return {"status": "success", "message": msg, "updates": updates}

@app.post("/save_imputation")
async def save_imputation(req: GeneRequest):
    filename, msg = dm.save_imputed_data(req.gene_name)
    if filename:
        return {"status": "success", "message": f"Saved to {filename}"}
    else:
        return {"status": "error", "message": f"Save failed: {msg}"}

@app.post("/get_annotation")
async def get_annotation():
    if dm.adata is None: return {"status": "error", "message": "Data not loaded"}
    
    # 调用 Nicheformer 预测
    # 注意：这里不再需要传入 cached_features，引擎内部自己处理
    try:
        pred_ids, legend_info = dm.ai_engine.predict_cell_types()
        
        # 将 legend_info 里的 name 列表提取出来给 legend 字段
        # legend_info 结构: [{'id':0, 'name':'T-Cell', 'color':'...'}, ...]
        class_names = [item['name'] for item in legend_info]
        
        updates = []
        ids = dm.adata.obs.index
        for i, pid in enumerate(pred_ids):
            updates.append({
                "id": str(ids[i]),
                "pred_id": int(pid) 
            })
            
        return {
            "status": "success",
            "legend": class_names, 
            "updates": updates
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

@app.post("/save_annotation")
async def save_annotation():
    filename, msg = dm.save_annotation_result()
    if filename:
        return {"status": "success", "message": f"Saved to {filename}"}
    else:
        return {"status": "error", "message": f"Save failed: {msg}"}

@app.get("/annotation_legend")
async def get_annotation_legend():
    """
    获取详细的图例信息（包含颜色）
    """
    try:
        # 复用 predict_cell_types 里的缓存
        _, legend_data = dm.ai_engine.predict_cell_types()
        return {
            "status": "success",
            "legend": legend_data
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/get_tissue_regions")
async def get_tissue_regions():
    if dm.adata is None: return {"status": "error", "message": "Data not loaded"}
            
    try:
        # 调用 Nicheformer 区域分割
        region_ids, region_names = dm.ai_engine.segment_tissue_regions()
        
        final_regions = region_ids.tolist() if hasattr(region_ids, "tolist") else region_ids
        final_names = region_names.tolist() if hasattr(region_names, "tolist") else list(region_names)

        return {
            "status": "success",
            "regions": final_regions,
            "names": final_names
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/save_tissue_regions")
async def save_tissue_regions():
    filename, msg = dm.save_region_result()
    if filename:
        return {"status": "success", "message": f"Results saved to {filename}"}
    else:
        return {"status": "error", "message": f"Save failed: {msg}"}
# =========================================================
# 零样本聚类
# =========================================================
@app.post("/zero_shot_cluster")
async def zero_shot_cluster(req: ClusteringRequest):
    """
    零样本聚类：返回带有 Cell ID 的更新列表
    """
    try:
        if dm.ai_engine is None:
            return JSONResponse(content={"status": "error", "message": "Model not loaded"}, status_code=500)

        # 1. 运行聚类 (得到的是纯数字数组，例如 [0, 1, 0, ...])
        cluster_ids_raw, legend_info = dm.ai_engine.run_zero_shot_clustering(req.n_clusters)
        
        # 2. 获取所有细胞的 ID (obs_names)
        # 这一步非常重要，确保 ID 和 聚类结果一一对应
        if dm.adata is None:
            raise Exception("Data not loaded in DataManager")
            
        cell_ids = dm.adata.obs_names.tolist()
        
        # 3. 【关键修复】构建 Unity 需要的 "updates" 列表
        # 将 ID 和 类别 组合在一起： [{"id": "cell_0", "cluster_id": 1}, ...]
        updates_list = []
        for cid, cluster_val in zip(cell_ids, cluster_ids_raw):
            updates_list.append({
                "id": str(cid),
                "cluster_id": int(cluster_val)
            })
        
        # 4. 更新内存中的 AnnData (方便保存)
        dm.update_clusters(cluster_ids_raw, legend_info)
        
        # 5. 返回给 Unity
        return {
            "status": "success",
            "message": f"Clustering finished. Found {len(legend_info)} clusters.",
            "legend": legend_info,
            "updates": updates_list  # <--- Unity 需要这个名字的字段！
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(content={"status": "error", "message": str(e)}, status_code=500)
@app.post("/save_zero_shot")
async def save_zero_shot(req: dict):
    """
    保存零样本聚类结果为 CSV 文件，路径与其他功能保持一致。
    """
    if dm.adata is None:
        raise HTTPException(500, "Data not loaded")
    
    # 调用刚刚新写的方法
    filename, msg = dm.save_zero_shot_result()
    
    if filename:
        return {"status": "success", "message": f"Clustering saved to {filename}"}
    else:
        return {"status": "error", "message": f"Save failed: {msg}"}
@app.post("/perturb")
async def calculate_perturbation(req: PerturbRequest): return {} 
@app.post("/clear_perturbation")
async def clear_perturbation(): return {} 
@app.post("/save_manual")
async def save_manual(): return {} 
@app.post("/impute_all")
async def impute_all(): return {}
@app.post("/disable_imputation")
async def disable_imputation(): return {}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)