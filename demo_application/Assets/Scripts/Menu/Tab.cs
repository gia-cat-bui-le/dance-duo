using TMPro;
using UnityEngine;
using UnityEngine.UI;

[RequireComponent(typeof(Image))]
public class Tab : MonoBehaviour
{
    public TabGroup tabGroup;

    public Image background;

    public GameObject textObject;
    public TextMeshProUGUI text;
    // Start is called before the first frame update
    void Start()
    {
        background = GetComponent<Image>();
        textObject = transform.GetChild(0).gameObject;
        text = textObject.GetComponent<TextMeshProUGUI>();
        tabGroup.Subscribe(this);
    }

    public void SetButtonColor(Color backgroundColor, Color textColor)
    {
        background.color = backgroundColor;
        text.color = textColor;
    }
}
