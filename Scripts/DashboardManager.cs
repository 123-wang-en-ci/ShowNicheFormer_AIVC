using UnityEngine;
using UnityEngine.UI; // 引用 UI
using TMPro;
using System.Linq; // 用于计算平均值

public class DashboardManager : MonoBehaviour
{
    [Header("UI 组件")]
    public RectTransform barCurrent; // 拖入红柱子
    public RectTransform barAverage; // 拖入灰柱子
    public TextMeshProUGUI valCurrentText;
    public TextMeshProUGUI valAverageText;

    [Header("设置")]
    public float maxHeight = 200f; // 柱子最大高度

    // 单例方便调用
    public static DashboardManager Instance;

    void Awake() { Instance = this; }

    // 更新图表
    public void UpdateChart(float currentVal, float allCellsAverage)
    {
        // 1. 设置文本
        valCurrentText.text = currentVal.ToString("F2");
        valAverageText.text = allCellsAverage.ToString("F2");

        // 2. 设置柱子高度 (假设最大值是 1.0)
        // 简单的动画效果可以用 Mathf.Lerp，这里直接设置
        barCurrent.sizeDelta = new Vector2(barCurrent.sizeDelta.x, currentVal * maxHeight);
        barAverage.sizeDelta = new Vector2(barAverage.sizeDelta.x, allCellsAverage * maxHeight);
    }
}