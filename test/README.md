# Extended Model Evaluation Suite

This directory contains a comprehensive test suite for evaluating the performance of the urban predictor models. The goal is to provide a detailed analysis of model performance beyond simple aggregate metrics.

## `evaluate.py`

This is the main script for running the evaluation. It takes a trained model checkpoint and produces a detailed report on its performance.

### Features

- **Per-Class Metrics**: Calculates metrics (MAE, RMSE) for each Dynamic World (DW) land cover type, allowing you to see if the model struggles with specific classes (e.g., "built-up" vs. "trees").
- **Interpretable Errors**: Reports temperature errors in degrees Celsius (°C) by un-normalizing the model's output.
- **Image Quality Analysis**: Assesses potential blurriness in the output images by calculating the variance of the Laplacian, which can indicate a loss of detail.
- **Detailed Logging**: Saves a comprehensive CSV report of all metrics for each sample in the test set to the `reports/tests/` directory.
- **WandB Integration**: Logs summary metrics, per-class breakdowns, and visualizations to Weights & Biases for easy comparison across different models.
- **Visualizations**: Generates and saves detailed plots for a subset of test samples, including:
    - Input data (RGB, NDVI, Temp, DW classes).
    - Ground Truth vs. Prediction for each target.
    - Error maps to visualize where the model is least accurate.
    - A bar chart of the Mean Absolute Error for each land cover type.

### How to Run

To run the evaluation, use the following command:

```bash
python test/evaluate.py --checkpoint-path /path/to/your/model.pth
```

**Arguments:**

*   `--checkpoint-path`: **(Required)** Path to the model checkpoint (`.pth` file) you want to evaluate.
*   `--device`: (Optional) The device to run on, e.g., `cuda:0` or `cpu`. Defaults to the device specified in the main config.
*   `--wandblog`: (Optional) Set to `False` to disable logging to Weights & Biases. Defaults to `True`.
*   `--study-name`: (Optional) A name for the evaluation study, used for grouping runs in WandB. Defaults to "test".
*   `--n-visualize`: (Optional) The number of sample visualizations to generate. Defaults to 10.

### Outputs

1.  **Console Output**: A summary of the average metrics (MAE, RMSE, PSNR, SSIM, Laplacian Variance) will be printed to the console.
2.  **CSV Report**: A detailed CSV file will be saved to `reports/tests/<study_name>_<trial_id>_evaluation.csv`. This file contains metrics for every single sample and every land cover class.
3.  **Visualizations**: PNG images for the visualized samples will be saved in `reports/tests/visualizations/`.
4.  **WandB Run**: If enabled, a new run will be created in your WandB project containing all metrics and visualizations.
