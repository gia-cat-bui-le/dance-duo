using Newtonsoft.Json;
using System.Collections;
using System.Collections.Generic;
using TMPro;
using Unity.VisualScripting;
using UnityEditor;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Video;

public class VideoManager : MonoBehaviour
{
    string path = "";
    public VideoPlayer videoPlayer;
    public RenderTexture videoTexture;
    UdpSocket udpSocket;
    bool sendData = false;
    bool receiveData = false;
    Dictionary<string, string> resultData;
    public GameObject loadingUI;
    public GameObject videoCanvas;
    public GameObject videoLoader;
    public TextMeshProUGUI scoringText;

    public Animator animator;
    public Button playButton;
    public RawImage videoRenderPlace;
    public RectTransform videoHolderPanel;

    public ShowScoreManager showScoreManager;

    public AudioSource audioSource;

    private void Start()
    {
        udpSocket = FindObjectOfType<UdpSocket>();
        loadingUI.SetActive(false);
        videoCanvas.SetActive(false);
        playButton.interactable = false;
        videoLoader.SetActive(true);
    }

    public void LoadVideo(string path)
    {
        /*
        path = EditorUtility.OpenFilePanel("Choose your dance video", "", "mp4");

        if (path.Length > 0)
        {
        */
        sendData = true;
        this.path = path;
        Debug.Log("path: " + path);
        StartCoroutine(ProcessVideo());
        /*
        }
        else
        {
            Debug.Log("not found file");
        }
        */
    }

    IEnumerator ProcessVideo()
    {
        if (sendData)
        {
            resultData = new Dictionary<string, string>();
            sendData = false;
            Dictionary<string, string> data = new Dictionary<string, string>();
            data.Add("video", path);
            data.Add("genre", "hiphop");
            string dataJson = JsonConvert.SerializeObject(data);
            Debug.Log("json data:" + dataJson);
            udpSocket.SendData(dataJson);
            loadingUI.SetActive(true);
        }
        while (!receiveData)
        {
            yield return null;
        }
        receiveData = false;
        path = resultData["video"];
        videoPlayer.url = "file://" + path;
        videoPlayer.Prepare();

        while (!PrepareVidAndTexture())
        {
            Debug.Log("video preparation not done");
            yield return null;
        }

        scoringText.text = resultData["score"];

        loadingUI.SetActive(false);
        playButton.interactable = true;
        videoCanvas.SetActive(true);
        videoLoader.SetActive(false);

        showScoreManager.EnableOption();

        yield break;
    }

    public void ReceiveResult(string result)
    {
        resultData = JsonConvert.DeserializeObject<Dictionary<string, string>>(result);
        receiveData = true;
        Debug.Log(resultData);
    }

    private bool PrepareVidAndTexture()
    {
        if (!videoPlayer.isPrepared)
        {
            return false;
        }

        return true;
    }

    public void PlayAnimationAndRecord()
    {
        animator.SetBool("playing", true);
        videoPlayer.Play();
        audioSource.PlayOneShot((AudioClip)Resources.Load(ChoiceSaver.MusicPath()));
        playButton.interactable = false;
    }

    public void StopAnimationAndVideo()
    {
        animator.SetBool("playing", false);
        videoPlayer.Stop();
    }
}
