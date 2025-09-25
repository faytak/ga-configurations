# ON Q5 Experiment Analysis Scripts

This directory contains scripts to analyze the results from the ON Q5 experiments defined in `on_experiments/on_Q_5runs.sh`.

## Scripts

### 1. `analyze_on_experiments.py`
Main analysis script that reads TensorBoard logs and calculates statistics for each experiment configuration.

**Features:**
- Extracts metrics from TensorBoard event files
- Calculates mean, standard deviation, min, max for all metrics
- Provides both "all values" and "final values" statistics
- Generates a summary table for key metrics (loss, accuracy)

### 2. `analyze_on_experiments_csv.py`
Backup analysis script that reads CSV logs (created by Lightning's CSVLogger).

**Features:**
- Same functionality as the TensorBoard version
- Uses CSV files instead of TensorBoard event files
- More reliable if TensorBoard parsing fails

### 3. `run_analysis.sh`
Convenience script that runs the analysis automatically.

**Usage:**
```bash
./scripts/run_analysis.sh
```

## Experiment Configurations

The scripts analyze 4 experiment configurations, each run 5 times:

1. **Experiment 1**: 30 samples, 1001 steps, batch_size 15
2. **Experiment 2**: 300 samples, 1001 steps, batch_size 15  
3. **Experiment 3**: 3000 samples, 301 steps, batch_size 15
4. **Experiment 4**: 30000 samples, 3001 steps, batch_size 15

## Expected Run Names

Based on the naming convention in `train.py`, the expected run names are:
- `on_regression_on_glg_nsm30_nst1001_bs15`
- `on_regression_on_glg_nsm300_nst1001_bs15`
- `on_regression_on_glg_nsm3000_nst301_bs15`
- `on_regression_on_glg_nsm30000_nst3001_bs15`

## Output

The scripts generate CSV files with detailed statistics:
- `on_q5_tensorboard_results.csv` (from TensorBoard analysis)
- `on_q5_csv_results.csv` (from CSV analysis)

Each row contains:
- Experiment configuration
- Metric name
- Statistics for all values (mean, std, min, max, count)
- Statistics for final values (mean, std, min, max, count)

## Requirements

- Python 3.7+
- pandas
- numpy
- tensorboard (for TensorBoard analysis)

## Usage

1. Run the experiments:
   ```bash
   ./scripts/on_experiments/on_Q_5runs.sh
   ```

2. Analyze the results:
   ```bash
   ./scripts/run_analysis.sh
   ```

3. Or run analysis manually:
   ```bash
   python scripts/analyze_on_experiments.py --log_dir lightning_logs --output results.csv
   ```

## Troubleshooting

- If TensorBoard analysis fails, the script will automatically try CSV analysis
- Make sure you're running from the project root directory
- Check that `lightning_logs` directory exists and contains the experiment results
- Verify that the run names match the expected pattern
