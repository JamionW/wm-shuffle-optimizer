# ============================================================================
# FILE: wm_model.py
# ============================================================================
import numpy as np
import networkx as nx
from scipy.linalg import expm, logm
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import inv as sparse_inv
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


class WhittleMaternModel:
    """Whittle-Matérn field model for shuffle cost prediction"""
    
    def __init__(self, nu=1.5, kappa=1.0, tau=1.0):
        self.nu = nu
        self.kappa = kappa
        self.tau = tau
        self.graph = None
        self.laplacian = None
        self.covariance = None
        self.node_mapping = {}
        
    def build_from_shuffle_history(self, shuffle_records):
        """Build partition graph from shuffle history"""
        self.graph = nx.Graph()
        
        for record in shuffle_records:
            src = record['source_partition']
            tgt = record['target_partition']
            weight = record['data_volume']
            
            if self.graph.has_edge(src, tgt):
                self.graph[src][tgt]['weight'] += weight
                self.graph[src][tgt]['count'] += 1
            else:
                self.graph.add_edge(src, tgt, weight=weight, count=1)
        
        # Create node mapping for matrix operations
        self.node_mapping = {node: i for i, node in enumerate(self.graph.nodes())}
        self.inverse_mapping = {i: node for node, i in self.node_mapping.items()}
        
        # Compute Laplacian
        self.laplacian = nx.laplacian_matrix(self.graph, weight='weight').astype(float)
        
    def compute_precision_matrix(self):
        """Compute precision matrix Q = tau * (kappa^2 * I + L)^alpha"""
        n = len(self.graph.nodes())
        alpha = self.nu + 0.5
        
        # Build the operator
        K = self.kappa**2 * np.eye(n) + self.laplacian.toarray()
        
        # Compute fractional power
        if abs(alpha - 1.0) < 1e-10:
            Q = self.tau * K
        elif abs(alpha - 2.0) < 1e-10:
            Q = self.tau * (K @ K)
        else:
            # Use eigendecomposition for fractional power
            eigvals, eigvecs = np.linalg.eigh(K)
            eigvals = np.maximum(eigvals, 1e-10)  # Ensure positive
            Q = self.tau * eigvecs @ np.diag(eigvals**alpha) @ eigvecs.T
            
        return Q
    
    def compute_covariance_matrix(self):
        """Compute and cache covariance matrix"""
        if self.covariance is None:
            Q = self.compute_precision_matrix()
            # Add regularization for numerical stability
            Q += 1e-6 * np.eye(Q.shape[0])
            self.covariance = np.linalg.inv(Q)
        return self.covariance
    
    def predict_shuffle_cost(self, source_partition, target_partition, data_volume):
        """Predict shuffle cost using covariance structure"""
        if self.graph is None:
            raise ValueError("Model not fitted. Call build_from_shuffle_history first.")
        
        C = self.compute_covariance_matrix()
        
        # Get indices
        src_idx = self.node_mapping.get(source_partition, 0)
        tgt_idx = self.node_mapping.get(target_partition, 0)
        
        # Covariance-based cost (low covariance = high cost)
        cov_value = C[src_idx, tgt_idx]
        base_cost = 1.0 / (abs(cov_value) + 1e-6)
        
        # Scale by data volume
        volume_factor = np.log1p(data_volume / 1e6)  # Convert to MB and log scale
        
        return base_cost * volume_factor
    
    def fit(self, shuffle_records, observed_costs):
        """Fit model parameters using MLE"""
        self.build_from_shuffle_history(shuffle_records)
        
        def objective(params):
            self.nu = max(0.1, params[0])
            self.kappa = max(0.01, np.exp(params[1]))
            self.tau = max(0.01, np.exp(params[2]))
            
            try:
                Q = self.compute_precision_matrix()
                
                # Log likelihood components
                sign, logdet = np.linalg.slogdet(Q)
                if sign <= 0:
                    return 1e10
                
                # Standardize costs
                y = np.array(observed_costs)
                y = (y - y.mean()) / (y.std() + 1e-6)
                
                # Pad or truncate y to match Q dimension
                n = Q.shape[0]
                if len(y) < n:
                    y = np.pad(y, (0, n - len(y)), mode='mean')
                else:
                    y = y[:n]
                
                # Negative log likelihood
                quad_form = y.T @ Q @ y
                nll = -0.5 * logdet + 0.5 * quad_form
                
                return nll
                
            except Exception as e:
                return 1e10
        
        # Initial parameters
        x0 = [self.nu, np.log(self.kappa), np.log(self.tau)]
        
        # Optimize
        result = minimize(objective, x0, method='L-BFGS-B', 
                         bounds=[(0.1, 3.0), (-3, 3), (-3, 3)])
        
        # Update parameters
        self.nu = max(0.1, result.x[0])
        self.kappa = max(0.01, np.exp(result.x[1]))
        self.tau = max(0.01, np.exp(result.x[2]))
        
        # Clear cached covariance
        self.covariance = None
        
        return result