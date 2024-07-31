using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

public class MainMenuManager : MonoBehaviour
{
    public Button nextButton;
    public Button backButton;

    public Button showDanceButton;
    public Button cancelButton;

    public GameObject mainMenu;
    public GameObject confirmationMenu;

    public ChoiceController choiceController;
    // Start is called before the first frame update
    void Start()
    {
        nextButton.interactable = true;
        backButton.interactable = false;

        mainMenu.SetActive(true);
        confirmationMenu.SetActive(false);
    }

    public void AskConfirm()
    {
        nextButton.interactable = false;
        backButton.interactable= false;
        confirmationMenu.SetActive(true);
    }

    public void ShowDance()
    {
        choiceController.SaveChoice();
        SceneManager.LoadScene("Show Dance");
    }

    public void Cancel()
    {
        nextButton.interactable = true;
        backButton.interactable = true;

        confirmationMenu.SetActive(false);
    }
}
