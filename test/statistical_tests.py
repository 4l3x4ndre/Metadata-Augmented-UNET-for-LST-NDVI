# test/statistical_tests.py

import pandas as pd
import typer
from typing import List
from scipy.stats import ttest_rel
from itertools import combinations
import os

app = typer.Typer()

def get_temporal_distance(t1_year):
    if t1_year <= 2021:
        return 'long_distance'
    elif t1_year in [2022, 2023]:
        return 'mid_distance'
    elif t1_year > 2023:
        return 'short_distance'
    else:
        return 'other'


def interpret_metrics(df: pd.DataFrame, model_name: str):
    """Provides interpretation for key metrics from a single model evaluation."""
    print(f"\n--- Interpreting metrics for {model_name} ---")
    print("\nNote: For regression tasks, 'accuracy' and 'precision' are captured by error metrics like MAE and RMSE. Lower is better.")

    if 'is_known_city' not in df.columns:
        df['is_known_city'] = True # Assume all are known if column is missing
    
    if 't1_year' in df.columns:
        df['temporal_distance'] = df['t1_year'].apply(get_temporal_distance)
    else:
        df['temporal_distance'] = 'not_available'


    for (is_known, temp_dist), group_df in df.groupby(['is_known_city', 'temporal_distance']):
        known_str = "Known Cities" if is_known else "Unknown Cities"
        print(f"\n--- Analysis for {known_str} | Temporal Distance: {temp_dist} ---")

        # Filter for overall metrics, as per-class can be noisy for interpretation
        overall_metrics = group_df[group_df['dw_class'] == 'overall'].groupby('channel')[['mae', 'rmse', 'laplacian_var_pred', 'laplacian_var_gt']].mean()

        if overall_metrics.empty:
            print("Could not find 'overall' metrics to interpret for this group.")
            continue

        for channel, row in overall_metrics.iterrows():
            mae = row['mae']
            rmse = row['rmse']
            laplacian_pred = row['laplacian_var_pred']
            laplacian_gt = row['laplacian_var_gt']

            print(f"\nChannel: {channel}")
            print(f"  MAE: {mae:.4f}, RMSE: {rmse:.4f}")

            if 'temp' in channel:
                if mae < 2.0:
                    interpretation = "Excellent. The model\'s temperature predictions are highly accurate (avg. error < 2°C)."
                elif mae < 4.0:
                    interpretation = "Good. The model\'s temperature predictions are reasonably accurate (avg. error < 4°C)."
                else:
                    interpretation = "Needs improvement. The model\'s temperature predictions have a notable deviation (avg. error >= 4°C)."
                print(f"  Interpretation (°C deviation): {interpretation}")
            
            elif 'ndvi' in channel:
                # NDVI range is [-1, 1], so total range is 2.0
                if mae < 0.05:
                    interpretation = "Excellent. The model\'s NDVI predictions are very precise (avg. error < 2.5% of the range)."
                elif mae < 0.1:
                    interpretation = "Good. The model\'s NDVI predictions are reasonably precise (avg. error < 5% of the range)."
                else:
                    interpretation = "Needs improvement. The model\'s NDVI predictions show significant deviation (avg. error >= 5% of the range)."
                print(f"  Interpretation (NDVI deviation): {interpretation}")

            # Smoothness
            if pd.notna(laplacian_pred) and pd.notna(laplacian_gt) and laplacian_gt > 0:
                smoothness_ratio = laplacian_pred / laplacian_gt
                print(f"  Smoothness (Laplacian Var): Pred={laplacian_pred:.4f}, GT={laplacian_gt:.4f}, Ratio={smoothness_ratio:.2f}")
                if smoothness_ratio > 1.5:
                    smoothness_interp = "The model\'s predictions may be overly noisy or contain artifacts."
                elif smoothness_ratio < 0.5:
                    smoothness_interp = "The model\'s predictions may be overly smooth, losing fine details."
                else:
                    smoothness_interp = "The model\'s predictions have a realistic level of detail and texture."
                print(f"  Interpretation (Smoothness): {smoothness_interp}")
            else:
                print("  Smoothness: Not available (GT Laplacian variance is zero or NaN).")


def comparative_analysis(dfs: List[pd.DataFrame], model_names: List[str]):
    """Performs pairwise comparative analysis of models using paired t-tests."""
    
    for i in range(len(dfs)):
        dfs[i]['model'] = model_names[i]
        if 'is_known_city' not in dfs[i].columns:
            dfs[i]['is_known_city'] = True 
        if 't1_year' in dfs[i].columns:
            dfs[i]['temporal_distance'] = dfs[i]['t1_year'].apply(get_temporal_distance)
        else:
            dfs[i]['temporal_distance'] = 'not_available'

    merged_df = pd.concat(dfs, ignore_index=True)

    # Pivot to get metrics side-by-side for each model
    pivot_df = merged_df.pivot_table(
        index=['is_known_city', 'temporal_distance', 'sample_idx', 'channel', 'dw_class'],
        columns='model',
        values=['mae', 'rmse']
    )

    model_pairs = list(combinations(model_names, 2))

    if not model_pairs:
        print("Need at least two models to compare.")
        return

    print("\n--- Comparative Analysis: Paired T-Test Results ---")
    print("A low p-value (< 0.05) suggests a statistically significant difference between models.")
    
    for model1, model2 in model_pairs:
        print(f"\n--- Comparing {model1} vs {model2} ---")
        
        for metric in ['mae', 'rmse']:
            metric_col1 = (metric, model1)
            metric_col2 = (metric, model2)

            # Drop rows where either model has a NaN for this metric
            compare_df = pivot_df[[metric_col1, metric_col2]].dropna()

            if compare_df.empty:
                print(f"No common samples to compare for metric '{metric}'")
                continue

            # Group by is_known_city, channel and dw_class to run tests
            grouped = compare_df.groupby(['is_known_city', 'temporal_distance', 'channel', 'dw_class'])
            
            print(f"\n** Metric: {metric.upper()} **")
            print(f"{'Known/Unknown':<15} {'Temporal Dist':<15} {'Channel':<15} {'DW Class':<20} {'Mean Diff (M1-M2)':<20} {'P-Value':<10} {'Winner'}")
            print("-" * 120)

            for name, group in grouped:
                is_known, temp_dist, channel, dw_class = name
                known_str = "Known" if is_known else "Unknown"
                
                m1_scores = group[metric_col1]
                m2_scores = group[metric_col2]

                if len(m1_scores) < 2 or len(m2_scores) < 2:
                    continue

                # Perform paired t-test
                try:
                    stat, p_value = ttest_rel(m1_scores, m2_scores, nan_policy='omit')
                except ValueError:
                    print(f"{known_str:<15} {temp_dist:<15} {channel:<15} {dw_class:<20} {'N/A':<20} {'N/A':<10} Could not compute test")
                    continue

                mean_diff = m1_scores.mean() - m2_scores.mean()

                winner = "Insignificant"
                if p_value < 0.05:
                    if mean_diff > 0:
                        winner = model2  # Lower score is better
                    else:
                        winner = model1

                print(f"{known_str:<15} {temp_dist:<15} {channel:<15} {dw_class:<20} {mean_diff:<20.4f} {p_value:<10.4f} {winner}")


@app.command()
def main(
    evaluation_csvs: List[str] = typer.Argument(..., help="Paths to one or more evaluation CSV files generated by evaluate.py."),
):
    """
    Performs statistical analysis and interpretation of model evaluation results.

    - If one CSV is provided, it performs an individual analysis, interpreting the metrics.
    - If multiple CSVs are provided, it performs a comparative analysis using paired t-tests
      to determine if the differences between models are statistically significant.
    """
    if not evaluation_csvs:
        print("Error: No evaluation CSV files provided.")
        raise typer.Exit(code=1)

    dfs = []
    model_names = []
    for csv_path in evaluation_csvs:
        if not os.path.exists(csv_path):
            print(f"Error: File not found at {csv_path}")
            raise typer.Exit(code=1)
        dfs.append(pd.read_csv(csv_path))
        model_name = os.path.basename(csv_path).replace('_evaluation.csv', '')
        model_names.append(model_name)

    if len(dfs) == 1:
        print(f"--- Individual Analysis for {model_names[0]} ---")
        interpret_metrics(dfs[0], model_names[0])
    else:
        comparative_analysis(dfs, model_names)

if __name__ == "__main__":
    app()
