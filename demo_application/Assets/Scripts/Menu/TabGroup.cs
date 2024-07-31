using System.Collections.Generic;
using UnityEngine;

public class TabGroup : MonoBehaviour
{
    public List<Tab> tabs;

    public Color tabIdleBackgroundColor;
    public Color tabIdleTextColor;

    public Color tabHoverBackgroundColor;
    public Color tabHoverTextColor;

    public Color tabActiveBackgroundColor;
    public Color tabActiveTextColor;

    public int selectedTabIndex = 0;
    public PanelGroup panelGroup;

    public void Subscribe(Tab button)
    {
        tabs ??= new List<Tab>();
        tabs.Add(button);
    }

    public void UpdateCurrentTab(int step)
    {
        selectedTabIndex += step;
        if (selectedTabIndex >= tabs.Count)
        {
            selectedTabIndex = tabs.Count - 1;
        }
        else if (selectedTabIndex < 0)
        {
            selectedTabIndex = 0;
        }
        ResetTabs();
        Debug.Log($"Tab select {selectedTabIndex}");
        tabs[selectedTabIndex].SetButtonColor(tabActiveBackgroundColor, tabActiveTextColor);
        if (panelGroup != null)
        {
            panelGroup.SetPageIndex(selectedTabIndex);
        }
    }

    public void ResetTabs()
    {
        for (int i = 0; i < tabs.Count; ++i)
        {
            if (selectedTabIndex == i)
            {
                continue;
            }
            Debug.Log($"Reset {i}");
            tabs[i].SetButtonColor(tabIdleBackgroundColor, tabIdleTextColor);
        }
    }
}
