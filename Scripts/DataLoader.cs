using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using System.Globalization;
using UnityEngine.Networking;
using System.Xml.Serialization;
using UnityEngine.UI;
using TMPro;

public class DataLoader : MonoBehaviour
{
    [Header("设置")]
    public string csvFileName = "unity_cell_data.csv";
    public GameObject cellPrefab;

    [Header("显示参数")]
    public float positionScale = 0.5f;
    public float heightMultiplier = 1.0f;
    public float CellScale = 5.0f;

    [Header("视觉增强设置")]
    public Gradient colorGradient;
    public float emissionIntensity = 2.0f;

    public Dictionary<string, GameObject> cellMap = new Dictionary<string, GameObject>();

    [Header("图例面板")]
    public GameObject legendPanel;
    public GameObject legendItemPrefab;
    public Transform legendContent;

    [Header("语义分割分区")]
    public TMP_Dropdown regionDropdown;

    private List<string> currentRegionNames = new List<string>();
    private List<int> savedRegionIds = new List<int>();

    public enum ViewMode
    {
        Expression,
        CellType,
        AI_Annotation,
        TissueRegion,
        ZeroShot // [新增] 零样本聚类模式
    }
    public ViewMode currentMode = ViewMode.Expression;

    private Dictionary<string, CellData> currentDataMap = new Dictionary<string, CellData>();
    private Dictionary<string, int> aiPredictionMap = new Dictionary<string, int>();

    // [新增] 零样本聚类数据缓存
    private Dictionary<string, int> zeroShotClusterMap = new Dictionary<string, int>();
    private Dictionary<int, Color> zeroShotColorMap = new Dictionary<int, Color>();

    public int highlightedTypeID = -1;
    public List<string> annotationLegend = new List<string>();
    private List<GameObject> legendItems = new List<GameObject>();

    struct CellData
    {
        public string id;
        public float x;
        public float y;
        public float expression;
        public int typeId;
        public string typeName;
    }

    [Header("色彩系统设置")]
    [Range(1, 100)]
    public int typeColorCount = 45;
    public Color[] typeColors;

    [Header("可视化配置")]
    public float saturation = 0.8f;
    public float brightness = 0.9f;

    void Awake()
    {
        GenerateTypeColors();
    }

    public void GenerateTypeColors()
    {
        typeColors = new Color[typeColorCount];
        for (int i = 0; i < typeColorCount; i++)
        {
            float hue = (float)i / typeColorCount;
            typeColors[i] = Color.HSVToRGB(hue, saturation, brightness);
        }
        Debug.Log($"[Unity] 已自动生成 {typeColorCount} 种区分颜色。");
    }

    void Start()
    {
        string filePath = Path.Combine(Application.streamingAssetsPath, csvFileName);
        if (File.Exists(filePath))
        {
            List<CellData> dataList = ParseCSV(filePath);
            SpawnCells(dataList);
        }
        else
        {
            Debug.LogError("找不到CSV文件！" + filePath);
        }

        if (legendPanel != null)
            legendPanel.SetActive(false);
    }

    List<CellData> ParseCSV(string path)
    {
        List<CellData> list = new List<CellData>();
        string[] lines = File.ReadAllLines(path);

        for (int i = 1; i < lines.Length; i++)
        {
            string line = lines[i];
            if (string.IsNullOrEmpty(line)) continue;
            string[] values = line.Split(',');
            if (values.Length < 6) continue;

            CellData data = new CellData();
            try
            {
                data.id = values[0];
                data.x = float.Parse(values[1], CultureInfo.InvariantCulture);
                data.y = float.Parse(values[2], CultureInfo.InvariantCulture);
                data.expression = float.Parse(values[4], CultureInfo.InvariantCulture);
                data.typeId = int.Parse(values[5]);
                if (values.Length > 6) data.typeName = values[6];

                list.Add(data);

                if (!currentDataMap.ContainsKey(data.id))
                {
                    currentDataMap.Add(data.id, data);
                }
            }
            catch (System.Exception e) { Debug.LogWarning(e.Message); }
        }
        return list;
    }

    void SpawnCells(List<CellData> cells)
    {
        GameObject root = new GameObject("Cell_Container");
        root.transform.position = Vector3.zero;
        MaterialPropertyBlock props = new MaterialPropertyBlock();

        foreach (var cell in cells)
        {
            GameObject obj = Instantiate(cellPrefab, root.transform);
            obj.name = cell.id;
            if (!cellMap.ContainsKey(cell.id)) cellMap.Add(cell.id, obj);
            UpdateObjectVisuals(obj, cell, props);
        }
    }

    void UpdateObjectVisuals(GameObject obj, CellData cell, MaterialPropertyBlock props, bool isImputation = false, float previousExpr = 0f)
    {
        float targetValue = 0f;
        Color baseColor = Color.white;
        float scale = 0.5f;
        if (!obj.activeSelf)
        {
            obj.SetActive(true);
        }
        // --- 模式分支 ---
        if (currentMode == ViewMode.Expression)
        {
            targetValue = cell.expression;
            baseColor = colorGradient.Evaluate(cell.expression);
            scale = 0.5f + cell.expression;
        }
        else if (currentMode == ViewMode.CellType)
        {
            targetValue = 1.0f;
            int safeId = Mathf.Clamp(cell.typeId, 0, typeColors.Length - 1);
            baseColor = typeColors[safeId];
            scale = 0.5f;
        }
        else if (currentMode == ViewMode.AI_Annotation)
        {
            targetValue = 0.5f;
            int predId = 0;
            if (aiPredictionMap.ContainsKey(cell.id))
            {
                predId = aiPredictionMap[cell.id];
            }

            if (highlightedTypeID == -1 || predId == highlightedTypeID)
            {
                int safeId = Mathf.Clamp(predId, 0, typeColors.Length - 1);
                baseColor = typeColors[safeId];
                scale = 0.8f;
            }
            else
            {
                scale = 0.0f;
            }
        }
        // [新增] 零样本聚类可视化逻辑
        else if (currentMode == ViewMode.ZeroShot)
        {
            targetValue = 0.5f;
            scale = 0.7f; // 默认大小

            if (zeroShotClusterMap.ContainsKey(cell.id))
            {
                int cId = zeroShotClusterMap[cell.id];
                // 优先使用后端传回的特定颜色
                if (zeroShotColorMap.ContainsKey(cId))
                {
                    baseColor = zeroShotColorMap[cId];
                }
                else
                {
                    // 兜底：使用默认色板
                    int safeId = Mathf.Clamp(cId, 0, typeColors.Length - 1);
                    baseColor = typeColors[safeId];
                }
            }
            else
            {
                baseColor = Color.gray; // 未聚类的细胞显示灰色
                scale = 0.3f;
            }
        }

        Vector3 targetPos = new Vector3(
            cell.x * positionScale,
            targetValue * heightMultiplier,
            cell.y * positionScale
        );

        if (isImputation && cell.expression > previousExpr + 0.05f)
        {
            StartCoroutine(AnimateGrowth(obj, targetPos, scale, cell.expression, props));
        }
        else
        {
            obj.transform.position = targetPos;
            obj.transform.localScale = Vector3.one * CellScale * scale;

            props.SetColor("_BaseColor", baseColor);
            props.SetColor("_EmissionColor", baseColor * emissionIntensity);

            obj.GetComponent<Renderer>().SetPropertyBlock(props);
        }
    }


    IEnumerator AnimateGrowth(GameObject obj, Vector3 targetPos, float targetScale, float expressionValue, MaterialPropertyBlock props)
    {
        float duration = 1.5f;
        float timer = 0f;

        Vector3 startPos = obj.transform.position;
        Vector3 startScale = obj.transform.localScale;

        Renderer rend = obj.GetComponent<Renderer>();

        while (timer < duration)
        {
            timer += Time.deltaTime;
            float t = timer / duration;
            t = Mathf.Sin(t * Mathf.PI * 0.5f);

            obj.transform.position = Vector3.Lerp(startPos, targetPos, t);
            obj.transform.localScale = Vector3.Lerp(startScale, Vector3.one * targetScale, t);

            if (rend != null)
            {
                float flash = Mathf.PingPong(Time.time * 5.0f, 1.0f);
                Color magicColor = Color.cyan;
                Color finalColor = Color.Lerp(magicColor, colorGradient.Evaluate(expressionValue), t);
                props.SetColor("_BaseColor", finalColor);
                props.SetColor("_EmissionColor", finalColor * (3.0f + flash * 5.0f));
                rend.SetPropertyBlock(props);
            }
            yield return null;
        }
        Color c = colorGradient.Evaluate(expressionValue);
        props.SetColor("_BaseColor", c);
        props.SetColor("_EmissionColor", c * emissionIntensity);
        rend.SetPropertyBlock(props);
    }

    [System.Serializable]
    public class UpdateData { public string id; public float new_expr; }

    [System.Serializable]
    public class ServerResponse
    {
        public string status;
        public string message;
        public UpdateData[] updates;
    }

    public void UpdateVisuals(string jsonResponse)
    {
        ServerResponse response = JsonUtility.FromJson<ServerResponse>(jsonResponse);
        if (response == null || response.updates == null) return;

        MaterialPropertyBlock props = new MaterialPropertyBlock();

        bool isImputationAnim = false;
        if (!string.IsNullOrEmpty(response.message))
        {
            isImputationAnim = response.message.Contains("Imputation") || response.message.Contains("Denoise");
        }

        foreach (var update in response.updates)
        {
            if (cellMap.ContainsKey(update.id))
            {
                float oldExpr = 0f;
                if (currentDataMap.ContainsKey(update.id)) oldExpr = currentDataMap[update.id].expression;

                if (currentDataMap.ContainsKey(update.id))
                {
                    CellData data = currentDataMap[update.id];
                    data.expression = update.new_expr;
                    currentDataMap[update.id] = data;
                }

                GameObject obj = cellMap[update.id];
                UpdateObjectVisuals(obj, currentDataMap[update.id], props, isImputationAnim, oldExpr);
            }
        }
    }

    [System.Serializable]
    public class AnnotationUpdate { public string id; public int pred_id; }
    [System.Serializable]
    public class AnnotationResponse { public string status; public string[] legend; public AnnotationUpdate[] updates; }

    public void ApplyAnnotationData(string jsonResponse)
    {
        AnnotationResponse res = JsonUtility.FromJson<AnnotationResponse>(jsonResponse);
        if (res.status != "success") return;

        annotationLegend.Clear();
        annotationLegend.AddRange(res.legend);

        foreach (var update in res.updates)
        {
            if (aiPredictionMap.ContainsKey(update.id))
                aiPredictionMap[update.id] = update.pred_id;
            else
                aiPredictionMap.Add(update.id, update.pred_id);
        }

        currentMode = ViewMode.AI_Annotation;
        highlightedTypeID = -1;
        RefreshAllCells();

        StartCoroutine(FetchAnnotationLegend((success) =>
        {
            if (!success) Debug.LogError("Failed to fetch annotation legend");
        }));
    }

    // -------------------------------------------------------------
    // [核心新增] 零样本聚类处理函数
    // -------------------------------------------------------------
    public void ApplyZeroShotClustering(List<ClusterLegendItem> legend, List<ClusterUpdateItem> updates)
    {
        // 1. 解析图例和颜色
        zeroShotColorMap.Clear();
        LegendItem[] uiLegendItems = new LegendItem[legend.Count];

        for (int i = 0; i < legend.Count; i++)
        {
            var item = legend[i];

            // 解析 Hex 颜色
            Color color;
            if (ColorUtility.TryParseHtmlString(item.color, out color))
            {
                // 确保 Alpha 也是 1
                color.a = 1.0f;
                if (!zeroShotColorMap.ContainsKey(item.id))
                    zeroShotColorMap.Add(item.id, color);
            }
            else
            {
                Debug.LogWarning($"无法解析颜色: {item.color}");
                if (!zeroShotColorMap.ContainsKey(item.id))
                    zeroShotColorMap.Add(item.id, typeColors[item.id % typeColors.Length]);
            }

            // 准备 UI 显示用的数据
            uiLegendItems[i] = new LegendItem { id = item.id, name = item.name };
        }

        // 2. 更新细胞数据映射
        zeroShotClusterMap.Clear();
        foreach (var update in updates)
        {
            if (!zeroShotClusterMap.ContainsKey(update.id))
                zeroShotClusterMap.Add(update.id, update.cluster_id);
            else
                zeroShotClusterMap[update.id] = update.cluster_id;
        }

        // 3. 切换模式并刷新
        currentMode = ViewMode.ZeroShot;
        RefreshAllCells();

        // 4. 更新图例面板 (使用自定义颜色)
        CreateLegendPanel(uiLegendItems, zeroShotColorMap);
    }
    // -------------------------------------------------------------

    public void SwitchMode(int modeIndex)
    {
        ViewMode oldMode = currentMode;
        currentMode = (ViewMode)modeIndex;

        // 如果从AI注释或零样本模式切换出去，隐藏图例面板
        if ((oldMode == ViewMode.AI_Annotation || oldMode == ViewMode.ZeroShot || oldMode == ViewMode.TissueRegion) &&
            (currentMode != ViewMode.AI_Annotation && currentMode != ViewMode.ZeroShot && currentMode != ViewMode.TissueRegion))
        {
            ClearLegendPanel();
        }
        else if (currentMode == ViewMode.AI_Annotation)
        {
            if (annotationLegend.Count > 0)
                StartCoroutine(FetchAnnotationLegend(null));
        }
        // [新增] 切换回零样本模式时重新显示图例
        else if (currentMode == ViewMode.ZeroShot)
        {
            // 这里我们假设 zeroShotColorMap 还在内存中
            // 如果需要重新生成图例 UI，需要保存上次的 LegendItems
            // 为了简化，目前逻辑是：如果用户点击"Start Clustering"，会重刷。
            // 简单的 SwitchMode 可能暂时无法恢复 ZeroShot 图例，除非我们把 LegendItems 也缓存下来
            // 建议通过再次点击按钮触发
        }

        RefreshAllCells();
    }

    public void ToggleViewMode() { int nextMode = (currentMode == ViewMode.Expression) ? 1 : 0; SwitchMode(nextMode); }

    void RefreshAllCells()
    {
        MaterialPropertyBlock props = new MaterialPropertyBlock();
        foreach (var kvp in currentDataMap)
        {
            if (cellMap.ContainsKey(kvp.Key)) UpdateObjectVisuals(cellMap[kvp.Key], kvp.Value, props);
        }
    }

    public bool GetCellDetails(string id, out string typeName, out Vector2 pos, out float expr)
    {
        if (currentDataMap.ContainsKey(id))
        {
            CellData data = currentDataMap[id];
            typeName = string.IsNullOrEmpty(data.typeName) ? "Unknown" : data.typeName;
            pos = new Vector2(data.x, data.y);
            expr = data.expression;
            return true;
        }
        typeName = "Unknown"; pos = Vector2.zero; expr = 0;
        return false;
    }

    public float GetAverageExpression()
    {
        if (currentDataMap.Count == 0) return 0;
        float sum = 0;
        foreach (var kvp in currentDataMap) sum += kvp.Value.expression;
        return sum / currentDataMap.Count;
    }

    [System.Serializable]
    public class LegendItem
    {
        public int id;
        public string name;
    }

    [System.Serializable]
    public class LegendResponse
    {
        public string status;
        public LegendItem[] legend;
    }

    public IEnumerator FetchAnnotationLegend(System.Action<bool> onComplete)
    {
        UnityWebRequest request = UnityWebRequest.Get("http://localhost:8000/annotation_legend");
        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            string jsonRespone = request.downloadHandler.text;
            ProcessLegendData(jsonRespone);
            if (onComplete != null) onComplete(true);
        }
        else
        {
            Debug.LogError("获取图例数据失败：" + request.error);
            if (onComplete != null) onComplete(false);
        }
    }

    private void ProcessLegendData(string jsonRespone)
    {
        var response = JsonUtility.FromJson<LegendResponse>(jsonRespone);
        if (response.status == "success")
        {
            CreateLegendPanel(response.legend);
        }
    }

    // [修改] 增加可选参数 overrideColors，支持自定义颜色（用于零样本聚类）
    private void CreateLegendPanel(LegendItem[] legendData, Dictionary<int, Color> overrideColors = null)
    {
        ClearLegendPanel();

        if (legendPanel != null)
            legendPanel.SetActive(true);

        foreach (var item in legendData)
        {
            if (legendItemPrefab != null && legendContent != null)
            {
                GameObject legendItemObj = Instantiate(legendItemPrefab, legendContent);
                legendItems.Add(legendItemObj);

                // 获取 Image 组件
                Image colorBox = null;
                Transform colorBoxTransform = legendItemObj.transform.Find("ColorBox");
                if (colorBoxTransform != null) colorBox = colorBoxTransform.GetComponent<Image>();
                else colorBox = legendItemObj.GetComponentInChildren<Image>();

                // 获取 Text 组件 (支持 TMP 和 UI Text)
                TMPro.TMP_Text tmpTextLabel = legendItemObj.GetComponentInChildren<TMPro.TMP_Text>();
                Text uiTextLabel = legendItemObj.GetComponentInChildren<Text>();

                // [修改] 设置颜色逻辑
                Color finalColor = Color.white;

                if (overrideColors != null && overrideColors.ContainsKey(item.id))
                {
                    // 1. 优先使用传入的覆盖颜色 (零样本模式)
                    finalColor = overrideColors[item.id];
                }
                else if (item.id < typeColors.Length)
                {
                    // 2. 否则使用默认色板
                    finalColor = typeColors[item.id];
                }

                // 修正 Alpha
                if (finalColor.a <= 0f) finalColor.a = 1f;

                if (colorBox != null) colorBox.color = finalColor;

                // 设置文本
                if (tmpTextLabel != null) tmpTextLabel.text = item.name;
                else if (uiTextLabel != null) uiTextLabel.text = item.name;
            }
        }
        Canvas.ForceUpdateCanvases();
        if (legendContent.TryGetComponent<VerticalLayoutGroup>(out var layout))
        {
            layout.enabled = false;
            layout.enabled = true;
        }
    }

    private void ClearLegendPanel()
    {
        foreach (var item in legendItems)
        {
            if (item != null) DestroyImmediate(item);
        }
        legendItems.Clear();

        if (legendPanel != null) legendPanel.SetActive(false);
    }

    public void ApplyRegionSegmentation(List<int> regionIds, List<string> regionNames)
    {
        currentMode = ViewMode.TissueRegion;
        Debug.Log($"[Unity] 语义分割染色与平面化对齐开始，数据量: {regionIds.Count}");

        MaterialPropertyBlock propBlock = new MaterialPropertyBlock();
        int colorID = Shader.PropertyToID("_BaseColor");

        float flatY = 0f;
        int index = 0;
        foreach (var kvp in cellMap)
        {
            if (index >= regionIds.Count) break;

            GameObject cellObj = kvp.Value;
            Vector3 currentPos = cellObj.transform.localPosition;
            cellObj.transform.localPosition = new Vector3(currentPos.x, flatY, currentPos.z);

            MeshRenderer mr = cellObj.GetComponent<MeshRenderer>();
            if (mr != null)
            {
                int rId = regionIds[index];
                Color targetColor = typeColors[rId % typeColors.Length];

                mr.GetPropertyBlock(propBlock);
                propBlock.SetColor(colorID, targetColor);
                mr.SetPropertyBlock(propBlock);
                cellObj.transform.localScale = Vector3.one * 1.5f;
            }
            index++;
        }

        if (regionNames != null && regionNames.Count > 0)
        {
            LegendItem[] legendData = new LegendItem[regionNames.Count];
            for (int i = 0; i < regionNames.Count; i++)
            {
                legendData[i] = new LegendItem { id = i, name = regionNames[i] };
            }
            CreateLegendPanel(legendData);
        }
        currentRegionNames = regionNames;
        savedRegionIds = regionIds;
        InitRegionDropdown(regionNames);
    }

    private void InitRegionDropdown(List<string> names)
    {
        if (regionDropdown == null) return;
        regionDropdown.ClearOptions();
        List<string> options = new List<string> { "Show All" };
        options.AddRange(names);
        regionDropdown.AddOptions(options);
        regionDropdown.onValueChanged.RemoveAllListeners();
        regionDropdown.onValueChanged.AddListener(OnDropdownValueChanged);
    }

    private void OnDropdownValueChanged(int index)
    {
        FilterRegions(index - 1);
    }

    public void FilterRegions(int targetRegionId)
    {
        int index = 0;
        foreach (var kvp in cellMap)
        {
            int cellRegionId = savedRegionIds[index];
            GameObject cellObj = kvp.Value;
            if (targetRegionId == -1 || cellRegionId == targetRegionId)
                cellObj.SetActive(true);
            else
                cellObj.SetActive(false);
            index++;
        }
    }
}