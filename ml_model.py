# ============================================================================
# FILE: ml_model.py
# Random Forest/XGBoost replacement for wm_model.py
# ============================================================================

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import networkx as nx
from scipy.spatial.distance import pdist, squareform
import json

# Optional XGBoost import
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available, using RandomForest only")


class MLShuffleModel:
    """
    Machine Learning-based shuffle cost prediction
    Replaces WhittleMaternModel with minimal interface changes
    """
    
    def __init__(self, model_type='random_forest', **kwargs):
        """
        Initialize ML model
        
        Args:
            model_type: 'random_forest' or 'xgboost'
            **kwargs: Model-specific parameters
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.partition_features = {}
        self.is_fitted = False
        
        # Model parameters (similar to WM's nu, kappa, tau)
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                min_samples_split=kwargs.get('min_samples_split', 5),
                random_state=42
            )
        elif model_type == 'xgboost' and XGBOOST_AVAILABLE:
            self.model = xgb.XGBRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.1),
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
    
    def build_from_shuffle_history(self, shuffle_records):
        """
        Build model from shuffle history (replaces WM's graph building)
        Extract partition characteristics and relationships
        """
        df = pd.DataFrame(shuffle_records)
        
        # Extract unique partitions
        partitions = list(set(df['source_partition'].unique().tolist() + 
                            df['target_partition'].unique().tolist()))
        
        # Build partition features
        self._extract_partition_features(df, partitions)
        
        # Build feature matrix
        X, y = self._create_training_data(df)
        
        # Fit model
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        print(f"Trained {self.model_type} on {len(X)} samples with {len(self.feature_names)} features")
        
        return self
    
    def _extract_partition_features(self, df, partitions):
        """Extract characteristics of each partition"""
        self.partition_features = {}
        
        for partition in partitions:
            # Traffic patterns
            outgoing = df[df['source_partition'] == partition]
            incoming = df[df['target_partition'] == partition]
            
            features = {
                'avg_outgoing_volume': outgoing['data_volume'].mean() if len(outgoing) > 0 else 0,
                'avg_incoming_volume': incoming['data_volume'].mean() if len(incoming) > 0 else 0,
                'total_outgoing': outgoing['data_volume'].sum() if len(outgoing) > 0 else 0,
                'total_incoming': incoming['data_volume'].sum() if len(incoming) > 0 else 0,
                'num_outgoing_connections': len(outgoing['target_partition'].unique()) if len(outgoing) > 0 else 0,
                'num_incoming_connections': len(incoming['source_partition'].unique()) if len(incoming) > 0 else 0,
                'avg_outgoing_cost': outgoing['actual_cost'].mean() if len(outgoing) > 0 else 0,
                'avg_incoming_cost': incoming['actual_cost'].mean() if len(incoming) > 0 else 0,
                'partition_id': partition  # Encoded as categorical
            }
            
            self.partition_features[partition] = features
    
    def _create_training_data(self, df):
        """Create feature matrix and target vector"""
        features = []
        targets = []
        
        for _, row in df.iterrows():
            src_partition = row['source_partition']
            tgt_partition = row['target_partition']
            
            # Get partition features
            src_features = self.partition_features.get(src_partition, {})
            tgt_features = self.partition_features.get(tgt_partition, {})
            
            # Combinatorial features
            feature_vector = [
                # Volume and distance features
                row['data_volume'],
                row['network_distance'],
                np.log1p(row['data_volume']),
                
                # Source partition features
                src_features.get('avg_outgoing_volume', 0),
                src_features.get('total_outgoing', 0),
                src_features.get('num_outgoing_connections', 0),
                src_features.get('avg_outgoing_cost', 0),
                
                # Target partition features  
                tgt_features.get('avg_incoming_volume', 0),
                tgt_features.get('total_incoming', 0),
                tgt_features.get('num_incoming_connections', 0),
                tgt_features.get('avg_incoming_cost', 0),
                
                # Relationship features
                int(src_partition == tgt_partition),  # Self-shuffle
                abs(src_partition - tgt_partition),   # Partition distance
                (src_partition + tgt_partition) % 4,  # Cluster affinity
                
                # Volume ratio features
                src_features.get('total_outgoing', 1) / (tgt_features.get('total_incoming', 1) + 1e-6),
                
                # Pattern features (from data_collector simulation)
                1 if row.get('pattern') == 'broadcast' else 0,
                1 if row.get('pattern') == 'join' else 0,
                1 if row.get('pattern') == 'aggregation' else 0,
                1 if row.get('pattern') == 'repartition' else 0,
            ]
            
            features.append(feature_vector)
            targets.append(row['actual_cost'])
        
        # Store feature names for interpretability
        self.feature_names = [
            'data_volume', 'network_distance', 'log_data_volume',
            'src_avg_out_vol', 'src_total_out', 'src_out_connections', 'src_avg_cost',
            'tgt_avg_in_vol', 'tgt_total_in', 'tgt_in_connections', 'tgt_avg_cost',
            'is_self_shuffle', 'partition_distance', 'cluster_affinity',
            'volume_ratio', 'is_broadcast', 'is_join', 'is_aggregation', 'is_repartition'
        ]
        
        return np.array(features), np.array(targets)
    
    def predict_shuffle_cost(self, source_partition, target_partition, data_volume):
        """
        Predict shuffle cost (compatible with WM interface)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call build_from_shuffle_history first.")
        
        # Get partition features
        src_features = self.partition_features.get(source_partition, {})
        tgt_features = self.partition_features.get(target_partition, {})
        
        # Estimate network distance (simplified)
        network_distance = abs(source_partition - target_partition) * 0.1
        
        # Build feature vector
        feature_vector = [
            data_volume,
            network_distance, 
            np.log1p(data_volume),
            src_features.get('avg_outgoing_volume', 0),
            src_features.get('total_outgoing', 0),
            src_features.get('num_outgoing_connections', 0),
            src_features.get('avg_outgoing_cost', 0),
            tgt_features.get('avg_incoming_volume', 0),
            tgt_features.get('total_incoming', 0),
            tgt_features.get('num_incoming_connections', 0),
            tgt_features.get('avg_incoming_cost', 0),
            int(source_partition == target_partition),
            abs(source_partition - target_partition),
            (source_partition + target_partition) % 4,
            src_features.get('total_outgoing', 1) / (tgt_features.get('total_incoming', 1) + 1e-6),
            0, 0, 0, 0  # Pattern features (unknown for prediction)
        ]
        
        X = np.array(feature_vector).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        prediction = self.model.predict(X_scaled)[0]
        return max(prediction, 0.001)  # Ensure positive cost
    
    def fit(self, shuffle_records, observed_costs):
        """
        Fit model (replaces WM's MLE fitting)
        """
        # This is already handled by build_from_shuffle_history
        # but kept for interface compatibility
        pass
    
    def get_feature_importance(self):
        """Get feature importance for interpretability"""
        if not self.is_fitted:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            return dict(zip(self.feature_names, importance))
        return None
    
    def save_model(self, filepath):
        """Save model parameters (compatible with WM interface)"""
        model_data = {
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'partition_features': self.partition_features,
            'is_fitted': self.is_fitted,
            'feature_importance': self.get_feature_importance()
        }
        
        with open(filepath, 'w') as f:
            json.dump(model_data, f, indent=2, default=str)
    
    def compute_covariance_matrix(self):
        """
        Compute partition affinity matrix (replaces WM covariance)
        Uses learned feature importance to estimate partition relationships
        """
        if not self.is_fitted:
            return np.eye(len(self.partition_features))
        
        partitions = list(self.partition_features.keys())
        n = len(partitions)
        affinity_matrix = np.zeros((n, n))
        
        # Use model to predict costs for all partition pairs
        for i, src in enumerate(partitions):
            for j, tgt in enumerate(partitions):
                # Use median volume for standardized comparison
                median_volume = 1e6
                cost = self.predict_shuffle_cost(src, tgt, median_volume)
                # Convert cost to affinity (lower cost = higher affinity)
                affinity_matrix[i, j] = 1.0 / (cost + 1e-6)
        
        # Normalize to [0, 1] range
        affinity_matrix = (affinity_matrix - affinity_matrix.min()) / (affinity_matrix.max() - affinity_matrix.min() + 1e-6)
        
        return affinity_matrix


# Compatibility wrapper to maintain interface
class RandomForestShuffleModel(MLShuffleModel):
    def __init__(self, **kwargs):
        super().__init__(model_type='random_forest', **kwargs)


class XGBoostShuffleModel(MLShuffleModel):
    def __init__(self, **kwargs):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available, use RandomForestShuffleModel instead")
        super().__init__(model_type='xgboost', **kwargs)