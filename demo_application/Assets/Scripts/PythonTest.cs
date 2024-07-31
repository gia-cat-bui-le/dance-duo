using System.Collections.Generic;
using TMPro;
using UnityEngine;

public class PythonTest : MonoBehaviour
{
    [SerializeField] TextMeshProUGUI pythonRcvdText = null;
    [SerializeField] TextMeshProUGUI sendToPythonText = null;

    string tempStr = "Sent from Python xxxx";
    int numToSendToPython = 0;
    UdpSocket udpSocket;

    public void QuitApp()
    {
        print("Quitting");
        Application.Quit();
    }

    public void UpdatePythonRcvdText(string str)
    {
        tempStr = str;
    }

    public void SendToPython()
    {
        Dictionary<string, string> data = new Dictionary<string, string>();
        data.Add("video", "");
        udpSocket.SendData("Make video");
        sendToPythonText.text = "On working";
    }

    private void Start()
    {
        udpSocket = FindObjectOfType<UdpSocket>();
        sendToPythonText.text = "Wait for starting";
    }

    void Update()
    {
        pythonRcvdText.text = tempStr;
    }
}
