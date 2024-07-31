using UnityEngine;

public class UIInit : MonoBehaviour
{
    static public void SetChoiceGroupsTheme(ChoiceGroup choiceGroup)
    {
        choiceGroup.buttonIdleTextColor = ThemeData.TextColor;
        choiceGroup.buttonIdleBackgroundColor = ThemeData.BackgroundColor;
        choiceGroup.buttonHoverTextColor = ThemeData.BackgroundColor;
        choiceGroup.buttonHoverBackgroundColor = ThemeData.PrimaryColor;
        choiceGroup.buttonActiveTextColor = ThemeData.BackgroundColor;
        choiceGroup.buttonActiveBackgroundColor = ThemeData.TextColor;
    }
}
