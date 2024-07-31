using UnityEngine;
using UnityEngine.UI;

public class FlexibleGridLayout : LayoutGroup
{
    public enum FitType
    {
        Uniform,
        Width,
        Height,
        FixedRows,
        FixedColumns
    }

    public FitType fitType;

    public int nRows;
    public int nColumns;

    public Vector2 cellSize;
    public Vector2 spacing;

    public bool fitX;
    public bool fitY;

    public override void CalculateLayoutInputHorizontal()
    {
        base.CalculateLayoutInputHorizontal();

        if (fitType == FitType.Width || fitType == FitType.Height || fitType == FitType.Uniform)
        {
            fitX = true;
            fitY = true;
            float sqrt = Mathf.Sqrt(transform.childCount);
            nRows = Mathf.CeilToInt(sqrt);
            nColumns = Mathf.CeilToInt(sqrt);
        }

        if (fitType == FitType.Width || fitType == FitType.FixedColumns)
        {
            nRows = Mathf.CeilToInt(transform.childCount / (float)nColumns);
        }

        if (fitType == FitType.Height || fitType == FitType.FixedRows)
        {
            nColumns = Mathf.CeilToInt(transform.childCount / (float)nRows);
        }

        var rect = rectTransform.rect;
        float parentWidth = rect.width;
        float parentHeight = rect.height;

        float cellWidth = parentWidth / (float)nColumns - (spacing.x / (float)nColumns) * 2 -
                          padding.left / (float)nColumns - padding.right / (float)nColumns;
        float cellHeight = parentHeight / (float)nRows - (spacing.y / (float)nRows) * 2 - padding.top / (float)nRows -
                           padding.bottom / (float)nRows;

        cellSize.x = fitX ? cellWidth : cellSize.x;
        cellSize.y = fitY ? cellHeight : cellSize.y;

        for (int i = 0; i < rectChildren.Count; i++)
        {
            var rowCount = i / nColumns;
            var columnCount = i % nColumns;

            var item = rectChildren[i];
            var xPos = cellSize.x * columnCount + spacing.x * columnCount + padding.left;
            var yPos = cellSize.y * rowCount + spacing.y * rowCount + padding.top;

            SetChildAlongAxis(item, 0, xPos, cellSize.x);
            SetChildAlongAxis(item, 1, yPos, cellSize.y);
        }
    }

    public override void CalculateLayoutInputVertical()
    {
    }

    public override void SetLayoutHorizontal()
    {
    }

    public override void SetLayoutVertical()
    {
    }
}