# ============================================================================
# COMPLETE REPLACEMENT for data_collector.py
# This fixes the method name issue and array comparison bug
# ============================================================================

import numpy as np
import pandas as pd
from datetime import datetime
import random
from typing import List, Dict, Optional
import json


class ShuffleDataCollector:
    """Collects or simulates shuffle metrics for testing"""
    
    def __init__(self, use_simulation=True, num_partitions=20, num_nodes=4):
        self.use_simulation = use_simulation
        self.num_partitions = num_partitions
        self.num_nodes = num_nodes
        self.shuffle_history = []
        
    def simulate_shuffle_metrics(self, num_queries=100):
        """Generate synthetic shuffle data with hidden spatial patterns"""
        np.random.seed(42)
        
        # Create partition topology with CLUSTERS
        # Partitions in same cluster have affinity (lower cost)
        num_clusters = 4
        partition_clusters = {}
        for i in range(self.num_partitions):
            partition_clusters[i] = i % num_clusters
        
        # Create hidden affinity matrix (this is what WM should discover)
        affinity_matrix = np.zeros((self.num_partitions, self.num_partitions))
        for i in range(self.num_partitions):
            for j in range(self.num_partitions):
                if partition_clusters[i] == partition_clusters[j]:
                    affinity_matrix[i, j] = 0.3  # Low cost multiplier
                else:
                    # Cost increases with cluster distance
                    cluster_dist = abs(partition_clusters[i] - partition_clusters[j])
                    affinity_matrix[i, j] = 1.0 + cluster_dist * 0.5
        
        # Add some noise to make it non-trivial
        affinity_matrix += np.random.normal(0, 0.1, affinity_matrix.shape)
        affinity_matrix = np.maximum(affinity_matrix, 0.1)
        
        for query_id in range(num_queries):
            pattern = random.choice(['broadcast', 'shuffle', 'local', 'skewed'])
            
            # Generate shuffles based on pattern
            if pattern == 'broadcast':
                source = random.randint(0, self.num_partitions - 1)
                targets = random.sample(range(self.num_partitions), k=random.randint(5, 10))
                
                for target in targets:
                    self._add_spatial_shuffle_record(
                        query_id, source, target, affinity_matrix, partition_clusters, pattern
                    )
            
            elif pattern == 'shuffle':
                # Prefer shuffles within clusters
                cluster = random.randint(0, num_clusters - 1)
                cluster_partitions = [p for p, c in partition_clusters.items() if c == cluster]
                
                # 70% within-cluster, 30% cross-cluster
                if random.random() < 0.7 and len(cluster_partitions) >= 2:
                    sources = random.sample(cluster_partitions, 
                                          k=min(len(cluster_partitions), random.randint(2, 5)))
                    targets = random.sample(cluster_partitions,
                                          k=min(len(cluster_partitions), random.randint(2, 5)))
                else:
                    sources = random.sample(range(self.num_partitions), k=random.randint(3, 8))
                    targets = random.sample(range(self.num_partitions), k=random.randint(3, 8))
                
                for source in sources:
                    for target in targets:
                        if random.random() > 0.5:
                            self._add_spatial_shuffle_record(
                                query_id, source, target, affinity_matrix, 
                                partition_clusters, pattern
                            )
            
            elif pattern == 'local':
                # Strong preference for same-cluster
                cluster = random.randint(0, num_clusters - 1)
                cluster_partitions = [p for p, c in partition_clusters.items() if c == cluster]
                
                if len(cluster_partitions) >= 2:
                    for _ in range(random.randint(2, 5)):
                        source, target = random.sample(cluster_partitions, 2)
                        self._add_spatial_shuffle_record(
                            query_id, source, target, affinity_matrix,
                            partition_clusters, pattern
                        )
            
            else:  # skewed
                hot_partition = random.randint(0, self.num_partitions - 1)
                sources = random.sample(range(self.num_partitions), k=random.randint(5, 10))
                
                for source in sources:
                    self._add_spatial_shuffle_record(
                        query_id, source, hot_partition, affinity_matrix,
                        partition_clusters, pattern
                    )
        
        return self.shuffle_history
    
    def _add_spatial_shuffle_record(self, query_id, source, target, affinity_matrix, 
                                partition_clusters, pattern):
        """Add shuffle record with NON-LINEAR spatial cost structure"""
        
        # Base volume (what the baseline sees)
        if pattern == 'broadcast':
            base_volume = np.random.lognormal(20, 1)
        elif pattern == 'skewed':
            base_volume = np.random.lognormal(21, 1.5)
        else:
            base_volume = np.random.lognormal(18, 1)
        
        # Network distance (what the baseline sees)
        source_node = source % self.num_nodes
        target_node = target % self.num_nodes
        same_node = (source_node == target_node)
        network_distance = 0 if same_node else 1
        
        # What the baseline would predict
        baseline_cost = (base_volume / 1e6) * (1 + network_distance)
        
        # Hidden spatial structure that BREAKS the linear relationship
        spatial_multiplier = affinity_matrix[source, target]
        
        # Non-linear transformation to hide the pattern
        if spatial_multiplier < 0.5:
            # Same cluster - costs are LOWER than baseline expects
            hidden_factor = 0.3 + 0.2 * np.sin(spatial_multiplier * np.pi)
        else:
            # Different cluster - costs are HIGHER in non-linear way
            hidden_factor = 1.5 + np.exp(spatial_multiplier - 1.0)
        
        # Add pattern-specific non-linearities
        if pattern == 'local' and spatial_multiplier < 0.5:
            # Local patterns within same cluster are super efficient
            hidden_factor *= 0.5
        elif pattern == 'skewed':
            # Skewed patterns have congestion effects
            hidden_factor *= (1.0 + 0.5 * np.log1p(query_id % 10))
        
        # True cost is NON-LINEAR function of inputs
        actual_cost = baseline_cost * hidden_factor
        
        # Add noise that's proportional to the hidden structure
        noise = np.random.normal(0, actual_cost * 0.1)
        actual_cost = max(0.1, actual_cost + noise)
        
        record = {
            'query_id': query_id,
            'source_partition': source,
            'target_partition': target,
            'data_volume': base_volume,
            'network_distance': network_distance,
            'pattern': pattern,
            'actual_cost': actual_cost,
            'baseline_estimate': baseline_cost,  # Add for comparison
            'timestamp': datetime.now().isoformat()
        }
        
        self.shuffle_history.append(record)
    
    def collect_spark_metrics(self, spark_context):
        """Collect real Spark shuffle metrics (for non-simulation mode)"""
        # USER NOTE: This requires a real Spark cluster
        # Implement based on your Spark version and monitoring setup
        
        status = spark_context.statusTracker()
        stages = status.getActiveStageInfo()
        
        metrics = []
        for stage in stages:
            if stage.numTasks > 0:
                # Extract shuffle read/write metrics
                shuffle_read = stage.shuffleReadBytes
                shuffle_write = stage.shuffleWriteBytes
                
                # This is simplified - real implementation would need
                # to parse stage details for partition-level metrics
                metrics.append({
                    'stage_id': stage.stageId,
                    'shuffle_read_bytes': shuffle_read,
                    'shuffle_write_bytes': shuffle_write,
                    'num_tasks': stage.numTasks
                })
        
        return metrics
    
    def save_history(self, filepath):
        """Save shuffle history to file"""
        df = pd.DataFrame(self.shuffle_history)
        df.to_csv(filepath, index=False)
        
    def load_history(self, filepath):
        """Load shuffle history from file"""
        df = pd.read_csv(filepath)
        self.shuffle_history = df.to_dict('records')
        return self.shuffle_history