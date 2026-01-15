using UnityEngine;
using TMPro;
using UnityEngine.UI; // 引用 Image 组件

public class UIManager : MonoBehaviour
{
    [Header("UI 组件")]
    public TextMeshProUGUI infoTitleText;
    public TextMeshProUGUI infoBodyText;

    [Header("系统消息组件")]
    public GameObject messagePanel; // 拖入 SystemMessagePanel
    public TextMeshProUGUI messageText; // 拖入 Txt_Message
    public Image messageBg; // 拖入 SystemMessagePanel (为了改背景色)

    public static UIManager Instance;


    void Awake()
    {
        Instance = this;
    }

    // 严谨的显示方法：接收具体的数据字段，而不是一串乱糟糟的字符串
    public void ShowCellDetails(string id, string cellType, Vector2 coordinates, float expression, float avgExpression)
    {
        // 1. 设置标题
        infoTitleText.text = ":: SINGLE  CELL  ANALYSIS ::";

        // 2. 格式化内容 (使用富文本 Rich Text)
        // <color=#888888> 是灰色标签，<b> 是加粗数值
        string content = "";

        content += $"<color=#FFFFFF>ID Ref:</color>\n";
        content += $"  <b><color=#FFFFFF>{id}</color></b>\n";

        content += $"<color=#FFFFFF>Cell Type:</color>\n";
        content += $"  <b><color=#00FF00>{cellType}</color></b>\n"; // 绿色高亮类型

        content += $"<color=#FFFFFF>Spatial Coords (um):</color>\n";
        content += $"  X: <b>{coordinates.x:F2}</b>  Y: <b>{coordinates.y:F2}</b>\n"; // F2 保留两位小数

        content += $"<color=#FFFFFF>Gene Expression:</color>\n";
        
        // 根据表达量高低改变颜色 (红色=高，蓝色=低)
        string exprColor = expression > 0.5f ? "#FF4444" : "#4444FF";
        content += $"  Value: <b><color={exprColor}>{expression:F4}</color></b>\n"; // F4 保留四位小数，体现精度
        
        // 计算偏差百分比
        float deviation = ((expression - avgExpression) / avgExpression) * 100f;
        string sign = deviation >= 0 ? "+" : "";
        content += $"  Dev:   <size=80%>{sign}{deviation:F1}% vs Avg</size>";

        infoBodyText.text = content;
    }
    // 屏幕中心显示系统提示消息的方法
    public void ShowSystemMessage(string msg, bool isError)
    {
        if (messagePanel == null) return;

        // 1. 设置内容
        messagePanel.SetActive(true);
        messageText.text = msg;

        // 2. 根据状态改变颜色 (例如：错误用红色背景，成功用绿色或黑色背景)
        if (messageBg != null)
        {
            if (isError)
                messageBg.color = new Color(0.8f, 0.2f, 0.2f, 0.9f); // 红色警告
            else
                messageBg.color = new Color(0.1f, 0.1f, 0.1f, 0.8f); // 黑色提示
        }

        // 3. 自动消失 (3秒后)
        CancelInvoke("HideSystemMessage"); // 如果上一个还没消失，先取消，防止闪烁
        Invoke("HideSystemMessage", 3.0f);
    }

    void HideSystemMessage()
    {
        if (messagePanel != null)
            messagePanel.SetActive(false);
    }
}