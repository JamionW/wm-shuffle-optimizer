# ============================================================================
# FILE: main.py
# ============================================================================
import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from pathlib import Path

# Import our modules
from wm_model import WhittleMaternModel
from data_collector import ShuffleDataCollector

# Optional: PySpark imports (will fail gracefully if not available)
try:
    from pyspark.sql import SparkSession
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    print("Warning: PySpark not available. Using simulation mode.")


class ShuffleOptimizationExperiment:
    """Main experiment runner"""
    
    def __init__(self, config_path='config.yaml'):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup output directory
        self.output_dir = Path(self.config['experiment']['output_path'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.wm_model = WhittleMaternModel(
            nu=self.config['model']['nu_init'],
            kappa=self.config['model']['kappa_init'],
            tau=self.config['model']['tau_init']
        )
        
        self.collector = ShuffleDataCollector(
            use_simulation=self.config['data']['use_simulation'],
            num_partitions=self.config['data']['num_partitions'],
            num_nodes=self.config['data']['num_nodes']
        )
        
        self.spark = None
        if SPARK_AVAILABLE and not self.config['data']['use_simulation']:
            self.spark = self._init_spark()
    
    def _init_spark(self):
        """Initialize Spark session"""
        return SparkSession.builder \
            .appName(self.config['spark']['app_name']) \
            .master(self.config['spark']['master']) \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "false") \
            .config("spark.eventLog.enabled", "true") \
            .getOrCreate()
    
    def collect_training_data(self):
        """Collect or generate shuffle data"""
        print("Collecting shuffle data...")
        
        if self.config['data']['use_simulation']:
            # Generate synthetic data
            self.collector.simulate_shuffle_metrics(
                num_queries=self.config['data']['synthetic_queries']
            )
        else:
            # Load test queries and execute them
            with open('test_queries.sql', 'r') as f:
                queries = f.read().split(';')
            
            for query in queries:
                if query.strip():
                    # Execute query and collect metrics
                    # USER NOTE: Implement based on your data setup
                    pass
        
        # Save collected data
        self.collector.save_history(self.output_dir / 'shuffle_history.csv')
        print(f"Collected {len(self.collector.shuffle_history)} shuffle records")
        
        return self.collector.shuffle_history
    
    def train_model(self, shuffle_data):
        """Train the Whittle-Matérn model"""
        print("\nTraining Whittle-Matérn model...")
        
        # Split data
        split_idx = int(len(shuffle_data) * self.config['experiment']['train_test_split'])
        train_data = shuffle_data[:split_idx]
        test_data = shuffle_data[split_idx:]
        
        # Extract costs for training
        train_costs = [record['actual_cost'] for record in train_data]
        
        # Fit model
        result = self.wm_model.fit(train_data, train_costs)
        
        print(f"Optimized parameters:")
        print(f"  nu (smoothness): {self.wm_model.nu:.3f}")
        print(f"  kappa (range): {self.wm_model.kappa:.3f}")
        print(f"  tau (precision): {self.wm_model.tau:.3f}")
        
        # Save parameters
        params = {
            'nu': float(self.wm_model.nu),
            'kappa': float(self.wm_model.kappa),
            'tau': float(self.wm_model.tau),
            'optimization_success': result.success,
            'optimization_message': result.message
        }
        
        with open(self.output_dir / 'model_parameters.json', 'w') as f:
            json.dump(params, f, indent=2)
        
        return train_data, test_data
    
    def evaluate_model(self, test_data):
        """Evaluate model performance"""
        print("\nEvaluating model performance...")
        
        predictions = []
        baselines = []
        actuals = []
        
        for record in test_data:
            # WM prediction
            pred = self.wm_model.predict_shuffle_cost(
                record['source_partition'],
                record['target_partition'],
                record['data_volume']
            )
            predictions.append(pred)
            
            # Simple baseline (volume-based)
            baseline = record['data_volume'] / 1e6 * (1 + record['network_distance'])
            baselines.append(baseline)
            
            # Actual cost
            actuals.append(record['actual_cost'])
        
        # Calculate metrics
        predictions = np.array(predictions)
        baselines = np.array(baselines)
        actuals = np.array(actuals)
        
        # Normalize for fair comparison
        predictions = (predictions - predictions.mean()) / predictions.std()
        baselines = (baselines - baselines.mean()) / baselines.std()
        actuals_norm = (actuals - actuals.mean()) / actuals.std()
        
        metrics = {
            'wm_mae': np.mean(np.abs(predictions - actuals_norm)),
            'baseline_mae': np.mean(np.abs(baselines - actuals_norm)),
            'wm_rmse': np.sqrt(np.mean((predictions - actuals_norm)**2)),
            'baseline_rmse': np.sqrt(np.mean((baselines - actuals_norm)**2)),
            'wm_correlation': np.corrcoef(predictions, actuals_norm)[0, 1],
            'baseline_correlation': np.corrcoef(baselines, actuals_norm)[0, 1]
        }
        
        # Calculate improvement
        metrics['mae_improvement'] = (
            (metrics['baseline_mae'] - metrics['wm_mae']) / 
            metrics['baseline_mae'] * 100
        )
        
        metrics['correlation_improvement'] = (
            (metrics['wm_correlation'] - metrics['baseline_correlation']) / 
            metrics['baseline_correlation'] * 100
        )
        
        # Save results
        with open(self.output_dir / 'evaluation_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame({
            'actual': actuals,
            'wm_prediction': predictions * actuals.std() + actuals.mean(),
            'baseline': baselines * actuals.std() + actuals.mean(),
            'pattern': [r['pattern'] for r in test_data]
        })
        comparison_df.to_csv(self.output_dir / 'predictions_comparison.csv', index=False)
        
        return metrics
    
    def visualize_results(self, metrics):
        """Create visualizations"""
        print("\nGenerating visualizations...")
        
        # 1. Covariance matrix heatmap
        plt.figure(figsize=(10, 8))
        cov_matrix = self.wm_model.compute_covariance_matrix()
        sns.heatmap(cov_matrix, cmap='coolwarm', center=0, 
                    cbar_kws={'label': 'Covariance'})
        plt.title('Learned Covariance Structure')
        plt.xlabel('Partition ID')
        plt.ylabel('Partition ID')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'covariance_matrix.png', dpi=150)
        plt.close()
        
        # 2. Prediction comparison scatter plot
        comparison_df = pd.read_csv(self.output_dir / 'predictions_comparison.csv')
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # WM predictions
        axes[0].scatter(comparison_df['actual'], comparison_df['wm_prediction'], 
                       alpha=0.6, c='blue')
        axes[0].plot([0, comparison_df['actual'].max()], 
                    [0, comparison_df['actual'].max()], 'r--', alpha=0.5)
        axes[0].set_xlabel('Actual Cost')
        axes[0].set_ylabel('WM Predicted Cost')
        axes[0].set_title(f'WM Model (Corr: {metrics["wm_correlation"]:.3f})')
        
        # Baseline predictions
        axes[1].scatter(comparison_df['actual'], comparison_df['baseline'], 
                       alpha=0.6, c='green')
        axes[1].plot([0, comparison_df['actual'].max()], 
                    [0, comparison_df['actual'].max()], 'r--', alpha=0.5)
        axes[1].set_xlabel('Actual Cost')
        axes[1].set_ylabel('Baseline Predicted Cost')
        axes[1].set_title(f'Baseline (Corr: {metrics["baseline_correlation"]:.3f})')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'prediction_comparison.png', dpi=150)
        plt.close()
        
        # 3. Error distribution by pattern
        plt.figure(figsize=(10, 6))
        comparison_df['wm_error'] = np.abs(comparison_df['actual'] - comparison_df['wm_prediction'])
        comparison_df['baseline_error'] = np.abs(comparison_df['actual'] - comparison_df['baseline'])
        
        patterns = comparison_df['pattern'].unique()
        x = np.arange(len(patterns))
        width = 0.35
        
        wm_errors = [comparison_df[comparison_df['pattern'] == p]['wm_error'].mean() 
                     for p in patterns]
        baseline_errors = [comparison_df[comparison_df['pattern'] == p]['baseline_error'].mean() 
                          for p in patterns]
        
        plt.bar(x - width/2, wm_errors, width, label='WM Model', color='blue', alpha=0.7)
        plt.bar(x + width/2, baseline_errors, width, label='Baseline', color='green', alpha=0.7)
        
        plt.xlabel('Query Pattern')
        plt.ylabel('Mean Absolute Error')
        plt.title('Prediction Error by Query Pattern')
        plt.xticks(x, patterns)
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'error_by_pattern.png', dpi=150)
        plt.close()
        
        print(f"Visualizations saved to {self.output_dir}")

    def evaluate_model_detailed(self, test_data):
        """Detailed evaluation showing pattern-specific performance"""
        print("\nEvaluating model performance by pattern...")
        
        predictions = []
        baselines = []
        actuals = []
        patterns = []
        
        for record in test_data:
            # WM prediction
            pred = self.wm_model.predict_shuffle_cost_improved(
                record['source_partition'],
                record['target_partition'],
                record['data_volume']
            )
            predictions.append(pred)
            
            # Baseline (what's in the record or recalculate)
            if 'baseline_estimate' in record:
                baseline = record['baseline_estimate']
            else:
                baseline = record['data_volume'] / 1e6 * (1 + record['network_distance'])
            baselines.append(baseline)
            
            # Actual cost
            actuals.append(record['actual_cost'])
            patterns.append(record['pattern'])
        
        # Convert to arrays
        predictions = np.array(predictions)
        baselines = np.array(baselines)
        actuals = np.array(actuals)
        patterns = np.array(patterns)
        
        # Overall metrics
        overall_metrics = {
            'wm_mae': np.mean(np.abs(predictions - actuals)),
            'baseline_mae': np.mean(np.abs(baselines - actuals)),
            'wm_correlation': np.corrcoef(predictions, actuals)[0, 1],
            'baseline_correlation': np.corrcoef(baselines, actuals)[0, 1]
        }
        
        # Pattern-specific analysis
        print("\nPattern-specific performance:")
        print("-" * 50)
        
        for pattern in np.unique(patterns):
            mask = patterns == pattern
            pattern_actuals = actuals[mask]
            pattern_preds = predictions[mask]
            pattern_baselines = baselines[mask]
            
            wm_mae = np.mean(np.abs(pattern_preds - pattern_actuals))
            baseline_mae = np.mean(np.abs(pattern_baselines - pattern_actuals))
            improvement = (baseline_mae - wm_mae) / baseline_mae * 100
            
            print(f"{pattern:10s}: WM MAE={wm_mae:.3f}, Baseline MAE={baseline_mae:.3f}, "
                f"Improvement={improvement:+.1f}%")
        
        print("-" * 50)
        
        # Find where WM wins
        wm_wins = np.abs(predictions - actuals) < np.abs(baselines - actuals)
        win_rate = np.mean(wm_wins) * 100
        
        print(f"\nWM wins on {win_rate:.1f}% of test cases")
        
        # Analyze wins by pattern
        print("\nWin rate by pattern:")
        for pattern in np.unique(patterns):
            mask = patterns == pattern
            pattern_win_rate = np.mean(wm_wins[mask]) * 100
            print(f"  {pattern}: {pattern_win_rate:.1f}%")
        
        return overall_metrics
    
    def generate_report(self, metrics):
        """Generate final report"""
        report = f"""
Whittle-Matérn Shuffle Cost Optimization - Experiment Report
=============================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Configuration:
--------------
- Simulation Mode: {self.config['data']['use_simulation']}
- Number of Partitions: {self.config['data']['num_partitions']}
- Number of Nodes: {self.config['data']['num_nodes']}
- Training Queries: {len(self.collector.shuffle_history)}

Model Parameters (Optimized):
-----------------------------
- nu (smoothness): {self.wm_model.nu:.3f}
- kappa (range): {self.wm_model.kappa:.3f}
- tau (precision): {self.wm_model.tau:.3f}

Performance Metrics:
--------------------
WM Model:
  - MAE: {metrics['wm_mae']:.4f}
  - RMSE: {metrics['wm_rmse']:.4f}
  - Correlation: {metrics['wm_correlation']:.4f}

Baseline (Volume-based):
  - MAE: {metrics['baseline_mae']:.4f}
  - RMSE: {metrics['baseline_rmse']:.4f}
  - Correlation: {metrics['baseline_correlation']:.4f}

Improvements:
  - MAE Reduction: {metrics['mae_improvement']:.1f}%
  - Correlation Improvement: {metrics['correlation_improvement']:.1f}%

Conclusion:
-----------
The Whittle-Matérn model {"OUTPERFORMS" if metrics['mae_improvement'] > 0 else "UNDERPERFORMS"} 
the baseline by {abs(metrics['mae_improvement']):.1f}% in terms of mean absolute error.

The learned covariance structure captures {"STRONG" if metrics['wm_correlation'] > 0.7 else "MODERATE" if metrics['wm_correlation'] > 0.5 else "WEAK"} 
relationships between partitions with a correlation of {metrics['wm_correlation']:.3f}.

Files Generated:
----------------
- shuffle_history.csv: Raw shuffle data
- model_parameters.json: Optimized WM parameters
- evaluation_metrics.json: Performance metrics
- predictions_comparison.csv: Detailed predictions
- covariance_matrix.png: Learned structure visualization
- prediction_comparison.png: Scatter plots
- error_by_pattern.png: Error analysis
"""
        
        with open(self.output_dir / 'experiment_report.txt', 'w') as f:
            f.write(report)
        
        print(report)
        
        return report
    
    def run(self):
        """Execute complete experiment"""
        print("="*60)
        print("Whittle-Matérn Shuffle Cost Optimization Experiment")
        print("="*60)
        
        # Step 1: Collect data
        shuffle_data = self.collect_training_data()
        
        # Step 2: Train model
        train_data, test_data = self.train_model(shuffle_data)
        
        # Step 3: Evaluate
        metrics = self.evaluate_model(test_data)
        
        # Step 4: Visualize
        self.visualize_results(metrics)
        
        # Step 5: Report
        self.generate_report(metrics)
        
        print("\n" + "="*60)
        print("Experiment complete! Check results/ directory for outputs.")
        print("="*60)
        
        return metrics


def main(config_path='config.yaml'):
    """Main entry point - works in both CLI and notebook environments"""
    
    # Check if we're in a notebook/IPython environment
    try:
        get_ipython()
        in_notebook = True
    except NameError:
        in_notebook = False
    
    if in_notebook:
        # Notebook mode - don't use argparse
        print("Running in notebook/Databricks mode")
        # You can override config_path here if needed
        # config_path = '/dbfs/wm-shuffle/config.yaml'  # Example for Databricks
    else:
        # CLI mode - use argparse
        import argparse
        parser = argparse.ArgumentParser(
            description='Whittle-Matérn Shuffle Cost Optimizer'
        )
        parser.add_argument(
            '--config', 
            default='config.yaml',
            help='Path to configuration file'
        )
        
        args = parser.parse_args()
        config_path = args.config
    
    # Run experiment
    experiment = ShuffleOptimizationExperiment(config_path)
    metrics = experiment.run()
    
    # Return metrics for notebook use
    return metrics, experiment

# Update the if __name__ == '__main__': block
if __name__ == '__main__':
    metrics, experiment = main()
    
    # Exit with appropriate code in CLI mode
    try:
        get_ipython()
        # In notebook - don't exit
        print("\nExperiment complete. Check 'metrics' variable for results.")
    except NameError:
        # In CLI - exit with code
        if metrics['mae_improvement'] > 0:
            print("\n✓ SUCCESS: WM model outperforms baseline!")
            sys.exit(0)
        else:
            print("\n✗ FAILURE: WM model underperforms baseline.")
            sys.exit(1)