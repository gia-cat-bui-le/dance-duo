using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ChoiceSaver
{
    static public string musicChoice;
    static public string modelChoice;
    static public bool isDanceMatching;

    static public string MusicPath()
    {
        return $"music/{musicChoice}";
    }

    static public string ModelPath()
    {
        string path = $"dance/Demo Ready/{musicChoice}/{modelChoice}";
        Debug.Log(path);
        return path;
    }
}
