# ============================================================================
# FILE: data_collector.py
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
        """Generate synthetic shuffle data for testing"""
        np.random.seed(42)
        
        # Create partition topology (some partitions are co-located)
        partition_nodes = {i: i % self.num_nodes for i in range(self.num_partitions)}
        
        for query_id in range(num_queries):
            # Simulate different query patterns
            pattern = random.choice(['broadcast', 'shuffle', 'local', 'skewed'])
            
            if pattern == 'broadcast':
                # One partition to many
                source = random.randint(0, self.num_partitions - 1)
                targets = random.sample(range(self.num_partitions), k=random.randint(5, 10))
                
                for target in targets:
                    self._add_shuffle_record(query_id, source, target, 
                                           partition_nodes, pattern)
                    
            elif pattern == 'shuffle':
                # Many to many
                sources = random.sample(range(self.num_partitions), k=random.randint(3, 8))
                targets = random.sample(range(self.num_partitions), k=random.randint(3, 8))
                
                for source in sources:
                    for target in targets:
                        if random.random() > 0.5:  # Sparse connections
                            self._add_shuffle_record(query_id, source, target,
                                                   partition_nodes, pattern)
                            
            elif pattern == 'local':
                # Prefer same-node transfers
                for _ in range(random.randint(2, 5)):
                    node = random.randint(0, self.num_nodes - 1)
                    node_partitions = [p for p, n in partition_nodes.items() if n == node]
                    if len(node_partitions) >= 2:
                        source, target = random.sample(node_partitions, 2)
                        self._add_shuffle_record(query_id, source, target,
                                               partition_nodes, pattern)
                                               
            else:  # skewed
                # Heavy traffic to specific partitions
                hot_partition = random.randint(0, self.num_partitions - 1)
                sources = random.sample(range(self.num_partitions), k=random.randint(5, 10))
                
                for source in sources:
                    self._add_shuffle_record(query_id, source, hot_partition,
                                           partition_nodes, pattern)
        
        return self.shuffle_history
    
    def _add_shuffle_record(self, query_id, source, target, partition_nodes, pattern):
        """Add a single shuffle record"""
        # Calculate network distance
        same_node = partition_nodes[source] == partition_nodes[target]
        network_distance = 0 if same_node else 1
        
        # Simulate data volume based on pattern
        if pattern == 'broadcast':
            base_volume = np.random.lognormal(20, 1)  # ~1GB
        elif pattern == 'skewed':
            base_volume = np.random.lognormal(21, 1.5)  # Larger, more variable
        else:
            base_volume = np.random.lognormal(18, 1)  # ~256MB
        
        # Simulate cost (correlated with volume and distance)
        base_cost = base_volume / 1e6 * (1 + network_distance * 2)
        noise = np.random.normal(0, base_cost * 0.1)
        actual_cost = max(0.1, base_cost + noise)
        
        record = {
            'query_id': query_id,
            'source_partition': source,
            'target_partition': target,
            'data_volume': base_volume,
            'network_distance': network_distance,
            'pattern': pattern,
            'actual_cost': actual_cost,
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