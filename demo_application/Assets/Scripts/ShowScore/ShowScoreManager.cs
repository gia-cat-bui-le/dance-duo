using SFB;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class ShowScoreManager : MonoBehaviour
{
    public VideoManager videoManager;
    public GameObject optionButtonsHolder;
    public GameObject postDanceMenu;

    public Button playButton;
    public Button nextButton;

    public AudioSource audioSource;

#if UNITY_WEBGL && !UNITY_EDITOR

    //WebGL
    [DllImport(*__Internal*)]
    private static extern void UploadFile(string gameObjectName, string methodName, string filter, boll multiple);

    public void OnClickOpen() 
    {
        UploadFile(gameObject.name, "OnfileUpload", ".obj", false);
    }

    // call from browser
    public void OnFileUpload(string url) 
    {
        StartCoroutine(OutputRoutineOpen(url));
    }
#else

    public void OnClickOpen()
    {
        string[] paths = StandaloneFileBrowser.OpenFilePanel("Select dance video", "", "mp4", false);
        if (paths.Length > 0)
        {
            videoManager.LoadVideo(paths[0]);
        }
    }

#endif

    // Start is called before the first frame update
    void Start()
    {
        optionButtonsHolder.SetActive(false);
    }

    // Update is called once per frame
    void Update()
    {

    }

    public void EnableOption()
    {
        optionButtonsHolder.SetActive(true);
    }

    public void EnablePostDanceOption()
    {
        postDanceMenu.SetActive(true);
    }

    public void DisableButtonsAndStopPlaying()
    {
        playButton.interactable = false;
        nextButton.interactable = false;
        audioSource.Stop();
        videoManager.StopAnimationAndVideo();
    }
}
