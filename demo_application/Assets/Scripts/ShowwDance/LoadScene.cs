using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

public class LoadScene : MonoBehaviour
{
    // Start is called before the first frame update
    public GameObject danceModel;
    public Animator animator;
    public bool isDanceFinish;
    public Button nextButton;

    public string animationName = "Armature|SMPL motion";
    public GameObject postDanceMenu;
    public AudioSource audioSource;

    void Start()
    {
        ChoiceSaver.musicChoice = "break";
        ChoiceSaver.isDanceMatching = true;
        ChoiceSaver.modelChoice = "girl";
        var danceAsset = (GameObject) Resources.Load(ChoiceSaver.ModelPath());
        if (danceAsset != null)
        {
            danceModel = Instantiate(danceAsset);
            animator = danceModel.GetComponent<Animator>();
        }
        else
        {
            Debug.Log("not found prefab");
        }
        nextButton.gameObject.SetActive(false);
        postDanceMenu.SetActive(false);
        Debug.Log(ChoiceSaver.musicChoice);
        AudioClip music = (AudioClip) Resources.Load(ChoiceSaver.MusicPath());
        if (music != null) {
            audioSource.clip = music;
        }
        else
        {
            Debug.Log("No clip found");
        }
    }

    // Update is called once per frame
    void Update()
    {
        if (!isDanceFinish)
        {
            var animationState = animator.GetCurrentAnimatorStateInfo(0);
            if (animationState.IsName(animationName) && animationState.normalizedTime >= 1.0)
            {
                FinishDance();
            }
        }
    }

    public void StartDance(Button button)
    {
        animator.SetBool("playing", true);
        button.interactable = false;
    }

    public void FinishDance()
    {
        isDanceFinish = true;
        nextButton.gameObject.SetActive(true);
        audioSource.Stop();
    }

    public void NextScene()
    {
        if (ChoiceSaver.isDanceMatching)
        {
            SceneManager.LoadScene("Show Score");
        }
        else
        {
            nextButton.interactable = false;
            postDanceMenu.SetActive(true);
        }
    }

    public void PlaySound()
    {
        audioSource.Play();
    }
}
