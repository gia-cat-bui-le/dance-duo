using UnityEngine;
using UnityEngine.UI;

public class ChoiceController : MonoBehaviour
{
    public ChoiceGroup musicChoice;
    public ChoiceGroup modelChoice;
    public Toggle togleDanceMatching;

    public string SaveChoice()
    {
        ChoiceSaver.modelChoice = modelChoice.choice;
        ChoiceSaver.musicChoice = musicChoice.choice;
        ChoiceSaver.isDanceMatching = togleDanceMatching.isOn;
        return "music: " + musicChoice.choice + ", model: " + modelChoice.choice + ", dance matching: " + togleDanceMatching.isOn;
    }
}
