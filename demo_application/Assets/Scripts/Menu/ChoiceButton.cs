using TMPro;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

public class ChoiceButton : MonoBehaviour, IPointerEnterHandler, IPointerClickHandler, IPointerExitHandler
{
    public ChoiceGroup choiceGroup;

    public Image background;

    public GameObject textObject;
    public TextMeshProUGUI text;
    public string choice;
    public bool isChosen = false;
    // Start is called before the first frame update
    void Start()
    {
        background = GetComponent<Image>();
        textObject = transform.GetChild(0).gameObject;
        text = textObject.GetComponent<TextMeshProUGUI>();
    }

    public void OnPointerEnter(PointerEventData eventData)
    {
        choiceGroup.OnButtonEnter(this);
    }

    public void OnPointerClick(PointerEventData eventData)
    {
        choiceGroup.OnButtonSelected(this);
    }

    public void OnPointerExit(PointerEventData eventData)
    {
        choiceGroup.OnButtonExit(this);
    }

    public void SetButtonColor(Color textColor, Color backgroundColor)
    {
        background.color = backgroundColor;
        text.color = textColor;
    }
}
