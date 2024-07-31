using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class PostDanceMenu : MonoBehaviour
{
    public void QuitApplication()
    {
        Application.Quit();
    }

    public void NewDance()
    {
        SceneManager.LoadScene("Main Menu", LoadSceneMode.Single);
    }

    public void ReplayDance()
    {
        SceneManager.LoadScene("Show Dance", LoadSceneMode.Single);
    }
}
