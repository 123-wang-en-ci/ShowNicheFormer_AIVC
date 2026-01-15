using UnityEngine;
using System.Collections.Generic;
using System.IO;
using System.Globalization;
using System.Runtime.InteropServices; // 必须引用，用于计算 struct 大小

public class GPURenderer : MonoBehaviour
{
    [Header("文件设置")]
    public string csvFileName = "unity_cell_data.csv";

    [Header("渲染设置")]
    public Mesh cellMesh;           // 拖入 Sphere Mesh
    public Material cellMaterial;   // 拖入刚才创建的 Shader 材质
    public Gradient colorGradient;  // 颜色渐变

    [Header("参数")]
    public float positionScale = 0.5f;
    public float heightMultiplier = 10.0f;
    public float baseScale = 0.5f;

    // --- 对应 HLSL 的 struct ---
    public struct CellDataGPU
    {
        public Vector3 position;
        public float scale;
        public Vector4 color;
    }

    // --- 核心变量 ---
    private ComputeBuffer cellBuffer;     // 数据 Buffer
    private ComputeBuffer argsBuffer;     // 参数 Buffer (告诉 GPU 画多少个)
    private uint[] args = new uint[5] { 0, 0, 0, 0, 0 };
    private int cellCount = 0;
    private Bounds bounds;                // 渲染边界
    // 保存内存中的数据副本，以便修改
    public List<CellDataGPU> cellDataList = new List<CellDataGPU>();
    void Start()
    {
        // 1. 读取数据 (和之前一样)
        List<CellDataGPU> dataList = LoadDataFromCSV();
        cellCount = dataList.Count;

        if (cellCount == 0) return;

        // 2. 初始化 Buffer
        InitializeBuffers(dataList);

        // 3. 设置巨大的边界，防止相机看别处时模型消失
        bounds = new Bounds(Vector3.zero, new Vector3(10000, 10000, 10000));
    }

    void Update()
    {
        // 每一帧命令 GPU 渲染
        if (cellCount > 0 && cellMaterial != null && cellMesh != null)
        {
            Graphics.DrawMeshInstancedIndirect(
                cellMesh,
                0,
                cellMaterial,
                bounds,
                argsBuffer
            );
        }
    }

    void InitializeBuffers(List<CellDataGPU> data)
    {
        // A. 释放旧的 (如果存在)
        if (cellBuffer != null) cellBuffer.Release();
        if (argsBuffer != null) argsBuffer.Release();

        // B. 创建 Cell Buffer
        // 32 是 stride (步长): Vector3(12) + float(4) + Vector4(16) = 32 bytes
        cellBuffer = new ComputeBuffer(cellCount, 32);
        cellBuffer.SetData(data); // 把数据推送到 GPU

        // C. 创建 Args Buffer (固定格式)
        argsBuffer = new ComputeBuffer(1, args.Length * sizeof(uint), ComputeBufferType.IndirectArguments);
        args[0] = (uint)cellMesh.GetIndexCount(0); // 网格顶点数
        args[1] = (uint)cellCount;                 // 实例数量 (几万个)
        args[2] = (uint)cellMesh.GetIndexStart(0);
        args[3] = (uint)cellMesh.GetBaseVertex(0);
        args[4] = 0;
        argsBuffer.SetData(args);

        // D. 把 Buffer 绑定到材质上
        cellMaterial.SetBuffer("_CellBuffer", cellBuffer);
        // 【新增这一行！】强制开启 Shader 的过程化实例化功能
        cellMaterial.EnableKeyword("PROCEDURAL_INSTANCING_ON");
    }

    // 清理显存 (非常重要！否则每次运行显存都会涨)
    void OnDisable()
    {
        if (cellBuffer != null) cellBuffer.Release();
        if (argsBuffer != null) argsBuffer.Release();
        cellBuffer = null;
        argsBuffer = null;
    }

    // 复用之前的 CSV 读取逻辑，但直接转为 GPU Struct
    List<CellDataGPU> LoadDataFromCSV()
    {
        List<CellDataGPU> list = new List<CellDataGPU>();
        string path = Path.Combine(Application.streamingAssetsPath, csvFileName);

        if (!File.Exists(path)) { Debug.LogError("找不到文件"); return list; }
        else Debug.Log("找到文件");

        string[] lines = File.ReadAllLines(path);
        for (int i = 1; i < lines.Length; i++)
        {
            if (string.IsNullOrEmpty(lines[i])) continue;
            string[] values = lines[i].Split(',');
            if (values.Length < 6) continue;

            try
            {
                float x = float.Parse(values[1], CultureInfo.InvariantCulture);
                float y = float.Parse(values[2], CultureInfo.InvariantCulture);
                float expr = float.Parse(values[4], CultureInfo.InvariantCulture);

                CellDataGPU cell = new CellDataGPU();
                // 坐标转换
                cell.position = new Vector3(x * positionScale, expr * heightMultiplier, y * positionScale);
                // 颜色转换
                Color c = colorGradient.Evaluate(expr);
                cell.color = new Vector4(c.r, c.g, c.b, 1);
                // 大小
                cell.scale = baseScale * (0.8f + expr);

                list.Add(cell);
            }
            catch { }
        }
        Debug.Log($"CSV加载完成，共读取到 {list.Count} 个细胞数据。");
        return list;
    }


    // 修改：将数据加载逻辑稍作调整，保存 list
    public void InitializeData(List<CellDataGPU> data)
    {
        this.cellDataList = data;
        this.cellCount = data.Count;
        // 更新 Buffer (原有逻辑)
        if (cellBuffer != null) cellBuffer.Release();
        cellBuffer = new ComputeBuffer(cellCount, Marshal.SizeOf(typeof(CellDataGPU)));
        cellBuffer.SetData(cellDataList.ToArray());
        cellMaterial.SetBuffer("_CellData", cellBuffer);
    }

    // 【核心新增】：供 DataLoader 调用来刷新区域颜色
    public void UpdateColorsForRegions(List<int> regionIds, Color[] palette)
    {
        if (cellDataList == null || cellDataList.Count == 0) return;

        for (int i = 0; i < cellDataList.Count; i++)
        {
            if (i >= regionIds.Count) break;

            // 1. 根据 Region ID 取色
            int rId = regionIds[i];
            Color c = palette[rId % palette.Length];

            // 2. 更新内存副本
            CellDataGPU temp = cellDataList[i];
            temp.color = new Vector4(c.r, c.g, c.b, 1.0f); // 确保 Alpha 是 1
            cellDataList[i] = temp;
        }

        // 3. 关键：将更新后的全量数据重新上传到显卡
        cellBuffer.SetData(cellDataList.ToArray());

        Debug.Log($"[GPU] 已刷新 {cellCount} 个细胞的区域颜色");
    }
}