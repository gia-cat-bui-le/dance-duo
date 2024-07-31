using System.Collections.Generic;
using UnityEngine;

public class ChoiceGroup : MonoBehaviour
{
    public string choice;
    public List<ChoiceButton> choiceButtons;
    public Color buttonIdleTextColor;
    public Color buttonIdleBackgroundColor;
    public Color buttonHoverTextColor;
    public Color buttonHoverBackgroundColor;
    public Color buttonActiveTextColor;
    public Color buttonActiveBackgroundColor;

    private void Start()
    {
        UIInit.SetChoiceGroupsTheme(this);
        choice = choiceButtons[0].choice;
        choiceButtons[0].isChosen = true;
        choiceButtons[0].SetButtonColor(buttonActiveTextColor, buttonActiveBackgroundColor);
    }

    public void OnButtonEnter(ChoiceButton button)
    {
        button.SetButtonColor(buttonHoverTextColor, buttonHoverBackgroundColor);
    }

    public void OnButtonSelected(ChoiceButton selectButton)
    {
        foreach (var button in choiceButtons)
        {
            button.SetButtonColor(buttonIdleTextColor, buttonIdleBackgroundColor);
            button.isChosen = false;
        }
        choice = selectButton.choice;
        selectButton.SetButtonColor(buttonActiveTextColor, buttonActiveBackgroundColor);
        selectButton.isChosen = true;
    }

    public void OnButtonExit(ChoiceButton button)
    {
        if (button.isChosen)
        {
            button.SetButtonColor(buttonActiveTextColor, buttonActiveBackgroundColor);
        }
        else
        {
            button.SetButtonColor(buttonIdleTextColor, buttonIdleBackgroundColor);
        }
    }
}
