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
        """Improved shuffle cost prediction using covariance as affinity"""
        if self.graph is None:
            raise ValueError("Model not fitted. Call build_from_shuffle_history first.")
        
        C = self.compute_covariance_matrix()
        
        # Map partitions to indices
        src_idx = self.node_mapping.get(source_partition, -1)
        tgt_idx = self.node_mapping.get(target_partition, -1)
        
        # Handle unseen partitions
        if src_idx == -1 or tgt_idx == -1:
            # Use average covariance for unseen partitions
            avg_cov = np.mean(np.abs(C))
            cov_value = avg_cov
        else:
            cov_value = C[src_idx, tgt_idx]
        
        # Normalize covariance to [0, 1] range
        C_min = np.min(C)
        C_max = np.max(C)
        if C_max - C_min > 1e-6:
            normalized_cov = (cov_value - C_min) / (C_max - C_min)
        else:
            normalized_cov = 0.5
        
        # High covariance = low cost (affinity)
        # Use exponential transformation for smoother mapping
        affinity = np.exp(2 * normalized_cov - 1)  # Range ~[0.37, 2.72]
        cost_multiplier = 1.0 / affinity
        
        # Combine with volume
        volume_factor = np.log1p(data_volume / 1e6)
        
        # Final cost
        predicted_cost = cost_multiplier * volume_factor
        
        return predicted_cost
    
    def fit(self, shuffle_records, observed_costs):
        """Fit model parameters using MLE"""
        self.build_from_shuffle_history(shuffle_records)
        
        def objective_with_regularization(params):
            nu = max(0.5, params[0])  # Higher minimum nu
            kappa = max(0.1, np.exp(params[1]))
            tau = max(0.1, np.exp(params[2]))
            
            # Prevent extreme values
            if nu > 2.5 or kappa > 10 or tau > 10:
                return 1e10
            
            self.nu = nu
            self.kappa = kappa
            self.tau = tau
            
            try:
                Q = self.compute_precision_matrix()
                
                # Log likelihood
                sign, logdet = np.linalg.slogdet(Q)
                if sign <= 0:
                    return 1e10
                
                # Standardize costs
                y = np.array(observed_costs)
                y = (y - y.mean()) / (y.std() + 1e-6)
                
                # Match dimensions
                n = Q.shape[0]
                if len(y) < n:
                    y = np.pad(y, (0, n - len(y)), mode='mean')
                else:
                    y = y[:n]
                
                # NLL with L2 regularization on parameters
                quad_form = y.T @ Q @ y
                nll = -0.5 * logdet + 0.5 * quad_form
                
                # Add regularization to prefer moderate parameters
                reg_term = 0.1 * (
                    (nu - 1.5)**2 +  # Prefer nu around 1.5
                    (np.log(kappa))**2 +  # Prefer kappa around 1
                    (np.log(tau))**2  # Prefer tau around 1
                )
                
                return nll + reg_term
                
            except Exception as e:
                return 1e10
        
        # Try multiple initializations and pick best
        best_result = None
        best_score = float('inf')
        
        for nu_init in [0.5, 1.0, 1.5, 2.0]:
            x0 = [nu_init, 0.0, 0.0]  # log(1) = 0
            
            result = minimize(
                objective_with_regularization, 
                x0, 
                method='L-BFGS-B',
                bounds=[(0.5, 2.5), (-2, 2), (-2, 2)]
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_result = result
        
        # Update with best parameters
        self.nu = max(0.5, best_result.x[0])
        self.kappa = max(0.1, np.exp(best_result.x[1]))
        self.tau = max(0.1, np.exp(best_result.x[2]))
        
        self.covariance = None
        return best_result