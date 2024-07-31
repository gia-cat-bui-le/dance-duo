using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

public class NextPageButton : MonoBehaviour
{
    public TabGroup tabGroup;
    public int step;

    private Button _thisButton;
    public NextPageButton otherButton;

    public ChoiceController choiceController;
    public MainMenuManager mainMenuManager;

    private void Start()
    {
        _thisButton = GetComponent<Button>();
    }

    public void ChangePage()
    {
        if (step > 0 && tabGroup.selectedTabIndex == tabGroup.tabs.Count - 1)
        {
            /*
            Debug.Log($"Choices: {choiceController.SaveChoice()}");
            SceneManager.LoadScene("Show Dance");
            */
            //TODO: Go to visualize
            mainMenuManager.AskConfirm();
        }
        else
        {
            tabGroup.UpdateCurrentTab(step);
            this._thisButton.interactable = otherButton._thisButton.interactable = true;
            switch (step)
            {
                case < 0 when tabGroup.selectedTabIndex <= 0:
                case > 0 when tabGroup.selectedTabIndex >= tabGroup.tabs.Count:
                    _thisButton.interactable = false;
                    break;
            }
        }
    }
}
