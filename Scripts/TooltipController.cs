using UnityEngine;
using TMPro; // 引用 TextMeshPro

public class TooltipController : MonoBehaviour
{
    [Header("UI 组件引用")]
    public GameObject tooltipObj;       // 拖入 Tooltip 物体
    public TextMeshProUGUI idText;      // 拖入 Txt_ID 物体
    public RectTransform canvasRect;    // 拖入 Canvas 物体

    [Header("设置")]
    public Vector2 offset = new Vector2(15f, -15f); // 鼠标偏移量，防止挡住鼠标

    void Start()
    {
        // 游戏开始时确保隐藏
        if (tooltipObj != null) tooltipObj.SetActive(false);
    }

    void Update()
    {
        // 1. 发射射线检测
        Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
        RaycastHit hit;

        // 注意：一定要给你的细胞 Prefab 加 Collider 才能检测到！
        if (Physics.Raycast(ray, out hit))
        {
            // 2. 如果打到了物体 (显示 Tooltip)
            if (tooltipObj != null && !tooltipObj.activeSelf)
            {
                tooltipObj.SetActive(true);
            }

            // 3. 更新文字 (读取物体名字作为 ID)
            if (idText != null)
            {
                idText.text = hit.transform.name;
            }

            // 4. 让 UI 跟随鼠标移动
            // 将屏幕上的鼠标坐标转换为 Canvas 内部的坐标
            Vector2 localPoint;
            RectTransformUtility.ScreenPointToLocalPointInRectangle(
                canvasRect,
                Input.mousePosition,
                null, // 如果 Canvas 是 Overlay 模式，这里填 null
                out localPoint
            );

            // 设置位置 + 偏移量
            tooltipObj.transform.localPosition = localPoint + offset;
        }
        else
        {
            // 5. 如果没打到物体 (隐藏 Tooltip)
            if (tooltipObj != null && tooltipObj.activeSelf)
            {
                tooltipObj.SetActive(false);
            }
        }
    }
}