using UnityEngine;

public class CameraOrbit : MonoBehaviour
{
    public Transform target; // 拖入 Cell_Container (或者手动建一个空物体放在细胞中心)
    public float distance = 50.0f;
    public float xSpeed = 120.0f;
    public float ySpeed = 120.0f;

    private float x = 0.0f;
    private float y = 0.0f;

    void Start()
    {
        Vector3 angles = transform.eulerAngles;
        x = angles.y;
        y = angles.x;

        // 如果没有目标，就创建一个临时的中心点
        if (target == null)
        {
            GameObject t = new GameObject("CamTarget");
            t.transform.position = new Vector3(50, 0, 50); // 假设你的模型中心大概在这里
            target = t.transform;
        }
    }

    void LateUpdate()
    {
        // 按住鼠标右键旋转
        if (target && Input.GetMouseButton(1))
        {
            x += Input.GetAxis("Mouse X") * xSpeed * 0.02f;
            y -= Input.GetAxis("Mouse Y") * ySpeed * 0.02f;

            Quaternion rotation = Quaternion.Euler(y, x, 0);
            Vector3 position = rotation * new Vector3(0.0f, 0.0f, -distance) + target.position;

            transform.rotation = rotation;
            transform.position = position;
        }

        // 滚轮缩放
        distance -= Input.GetAxis("Mouse ScrollWheel") * 10f;
    }
}