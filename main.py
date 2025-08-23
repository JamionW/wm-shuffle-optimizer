# ============================================================================
# FILE: main.py (ML VERSION)
# MINIMAL CHANGES: Only imports and model initialization changed
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
from ml_model import MLShuffleModel  # Was: from wm_model import WhittleMaternModel
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
        self.ml_model = MLShuffleModel(
            model_type=self.config['model'].get('type', 'random_forest'),  # Added model type
            n_estimators=self.config['model'].get('n_estimators', 100),     # Added ML params
            max_depth=self.config['model'].get('max_depth', 10)
        )
        # OLD: self.wm_model = WhittleMaternModel(nu=..., kappa=..., tau=...)
        
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
            # Use synthetic data
            shuffle_data = self.collector.simulate_shuffle_metrics(
                num_queries=self.config['data']['synthetic_queries']
            )
        else:
            # Collect real Spark metrics
            if self.spark is None:
                raise RuntimeError("Spark not available for real data collection")
            
            # This would collect real metrics from Spark
            shuffle_data = self.collector.collect_real_metrics(self.spark)
        
        # Save data
        self.collector.save_history(self.output_dir / 'shuffle_history.csv')
        print(f"Collected {len(shuffle_data)} shuffle records")
        
        return shuffle_data
    
    def train_model(self, shuffle_data):
        """Train model """
        print("Training ML model...")  # Was: "Training WM model..."
        
        # Split data
        split_idx = int(len(shuffle_data) * self.config['experiment']['train_test_split'])
        train_data = shuffle_data[:split_idx]
        test_data = shuffle_data[split_idx:]
        
        # Build and train model 
        self.ml_model.build_from_shuffle_history(train_data)  # Was: self.wm_model.build_from_shuffle_history
        
        # Save model parameters 
        self.ml_model.save_model(self.output_dir / 'model_parameters.json')  # Was: self.wm_model.save_model
        
        print(f"Model trained on {len(train_data)} records")
        return train_data, test_data
    
    def evaluate_model(self, test_data):
        """Evaluate model performance """
        print("Evaluating model...")
        
        # Make predictions 
        ml_predictions = []  # Was: wm_predictions
        baseline_predictions = []
        actual_costs = []
        
        for record in test_data:
            # ML model prediction 
            ml_pred = self.ml_model.predict_shuffle_cost(  # Was: self.wm_model.predict_shuffle_cost
                record['source_partition'],
                record['target_partition'], 
                record['data_volume']
            )
            ml_predictions.append(ml_pred)
            
            # Baseline prediction (unchanged)
            baseline_pred = record.get('baseline_estimate', record['data_volume'] * 1e-6)
            baseline_predictions.append(baseline_pred)
            
            actual_costs.append(record['actual_cost'])
        
        # Convert to arrays
        ml_predictions = np.array(ml_predictions)      # Was: wm_predictions
        baseline_predictions = np.array(baseline_predictions)
        actual_costs = np.array(actual_costs)
        
        # Compute metrics 
        ml_mae = np.mean(np.abs(ml_predictions - actual_costs))          # Was: wm_mae
        ml_rmse = np.sqrt(np.mean((ml_predictions - actual_costs)**2))   # Was: wm_rmse
        ml_correlation = np.corrcoef(ml_predictions, actual_costs)[0, 1] # Was: wm_correlation
        
        baseline_mae = np.mean(np.abs(baseline_predictions - actual_costs))
        baseline_rmse = np.sqrt(np.mean((baseline_predictions - actual_costs)**2))
        baseline_correlation = np.corrcoef(baseline_predictions, actual_costs)[0, 1]
        
        # Store metrics 
        metrics = {
            'ml_mae': ml_mae,                    # Was: wm_mae
            'ml_rmse': ml_rmse,                  # Was: wm_rmse  
            'ml_correlation': ml_correlation,    # Was: wm_correlation
            'baseline_mae': baseline_mae,
            'baseline_rmse': baseline_rmse,
            'baseline_correlation': baseline_correlation,
            'mae_improvement': (baseline_mae - ml_mae) / baseline_mae * 100,    # Was: wm_mae
            'correlation_improvement': (ml_correlation - baseline_correlation) / abs(baseline_correlation) * 100  # Was: wm_correlation
        }
        
        # Save detailed results
        results_df = pd.DataFrame({
            'actual_cost': actual_costs,
            'ml_prediction': ml_predictions,        # Was: wm_prediction
            'baseline_prediction': baseline_predictions,
            'ml_error': ml_predictions - actual_costs,      # Was: wm_error
            'baseline_error': baseline_predictions - actual_costs
        })
        
        results_df.to_csv(self.output_dir / 'predictions_comparison.csv', index=False)
        
        # Save metrics
        with open(self.output_dir / 'evaluation_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"ML MAE: {ml_mae:.4f}, Baseline MAE: {baseline_mae:.4f}")      # Was: WM MAE
        print(f"Improvement: {metrics['mae_improvement']:.1f}%")
        
        return metrics
    
    def visualize_results(self, metrics):
        """Create visualizations """
        print("Generating visualizations...")
        
        # Load results
        df = pd.read_csv(self.output_dir / 'predictions_comparison.csv')
        
        # 1. Prediction scatter plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # ML model scatter 
        axes[0].scatter(df['actual_cost'], df['ml_prediction'], alpha=0.6)  # Was: wm_prediction
        axes[0].plot([df['actual_cost'].min(), df['actual_cost'].max()], 
                    [df['actual_cost'].min(), df['actual_cost'].max()], 'r--')
        axes[0].set_xlabel('Actual Cost')
        axes[0].set_ylabel('ML Prediction')                                 # Was: WM Prediction
        axes[0].set_title(f'ML Model (R²={metrics["ml_correlation"]:.3f})')  # Was: WM Model, wm_correlation
        
        # Baseline scatter 
        axes[1].scatter(df['actual_cost'], df['baseline_prediction'], alpha=0.6)
        axes[1].plot([df['actual_cost'].min(), df['actual_cost'].max()], 
                    [df['actual_cost'].min(), df['actual_cost'].max()], 'r--')
        axes[1].set_xlabel('Actual Cost')
        axes[1].set_ylabel('Baseline Prediction')
        axes[1].set_title(f'Baseline Model (R²={metrics["baseline_correlation"]:.3f})')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'prediction_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Covariance/Affinity matrix
        affinity_matrix = self.ml_model.compute_covariance_matrix()  # Was: self.wm_model.compute_covariance_matrix
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(affinity_matrix, cmap='viridis', cbar_kws={'label': 'Partition Affinity'})  # Was: Learned Covariance
        plt.title('Learned Partition Affinity Matrix')  # Was: WM Covariance Structure
        plt.xlabel('Target Partition')
        plt.ylabel('Source Partition')
        plt.savefig(self.output_dir / 'covariance_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 3. Feature importance (NEW - ML specific)
        importance = self.ml_model.get_feature_importance()
        if importance:
            plt.figure(figsize=(10, 6))
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            features, values = zip(*sorted_features[:10])  # Top 10 features
            
            plt.barh(range(len(features)), values)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Feature Importance')
            plt.title('Top 10 Most Important Features')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(self.output_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
            plt.close()
        
        print("Visualizations saved to results/ directory")
        
        return metrics
    
    def generate_report(self, metrics):
        """Generate final report """
        report = f"""
Machine Learning Shuffle Cost Optimization - Experiment Report
===============================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Configuration:
--------------
- Simulation Mode: {self.config['data']['use_simulation']}
- Number of Partitions: {self.config['data']['num_partitions']}
- Number of Nodes: {self.config['data']['num_nodes']}
- Training Queries: {len(self.collector.shuffle_history)}
- Model Type: {self.ml_model.model_type}                   

Performance Metrics:
--------------------
ML Model:                                               
  - MAE: {metrics['ml_mae']:.4f}                           
  - RMSE: {metrics['ml_rmse']:.4f}                        
  - Correlation: {metrics['ml_correlation']:.4f}          

Baseline (Volume-based):
  - MAE: {metrics['baseline_mae']:.4f}
  - RMSE: {metrics['baseline_rmse']:.4f}
  - Correlation: {metrics['baseline_correlation']:.4f}

Improvements:
  - MAE Reduction: {metrics['mae_improvement']:.1f}%
  - Correlation Improvement: {metrics['correlation_improvement']:.1f}%

Conclusion:
-----------
The ML model {"OUTPERFORMS" if metrics['mae_improvement'] > 0 else "UNDERPERFORMS"} 
the baseline by {abs(metrics['mae_improvement']):.1f}% in terms of mean absolute error.

The learned patterns capture {"STRONG" if metrics['ml_correlation'] > 0.7 else "MODERATE" if metrics['ml_correlation'] > 0.5 else "WEAK"} 
relationships with a correlation of {metrics['ml_correlation']:.3f}.

Files Generated:
----------------
- shuffle_history.csv: Raw shuffle data
- model_parameters.json: Model parameters and feature importance
- evaluation_metrics.json: Performance metrics
- predictions_comparison.csv: Detailed predictions
- covariance_matrix.png: Learned affinity structure visualization
- prediction_comparison.png: Scatter plots
- feature_importance.png: Most important features 
"""
        
        with open(self.output_dir / 'experiment_report.txt', 'w') as f:
            f.write(report)
        
        print(report)
        
        return report
    
    def run(self):
        """Execute complete experiment"""
        print("="*60)
        print("Machine Learning Shuffle Cost Optimization Experiment")  # Was: Whittle-Matérn
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
    """Main entry point"""
    
    # Check if we're in a notebook/IPython environment
    try:
        get_ipython()
        in_notebook = True
    except NameError:
        in_notebook = False
    
    if in_notebook:
        print("Running in notebook/Databricks mode")
    else:
        import argparse
        parser = argparse.ArgumentParser(
            description='ML Shuffle Cost Optimizer'  # Was: Whittle-Matérn
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
    
    return metrics, experiment

if __name__ == '__main__':
    metrics, experiment = main()
    
    try:
        get_ipython()
        print("\nExperiment complete. Check 'metrics' variable for results.")
    except NameError:
        pass