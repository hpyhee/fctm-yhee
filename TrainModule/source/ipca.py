# %%
import torch
import math

class IPCA:
    def __init__(self, n_components, n_features) -> None:
        self.n_components   = n_components
        self.n_features     = n_features
        self.n_samples_seen = 0
        self.step           = 0
        self.explained_variance = None
        self.noise_variance     = None
        self.running_mean  = torch.zeros((1, n_features)           , dtype=torch.float32)
        self.components    = torch.zeros((n_components, n_features), dtype=torch.float32)
        self.singular_vals = torch.zeros((n_components)            , dtype=torch.float32)
    
    def to(self, device):
        self.running_mean  = self.running_mean.to(device)
        self.components    = self.components.to(device)
        self.singular_vals = self.singular_vals.to(device)
        return self
    
    def fit(self, inputs):
        n_samples, n_features = inputs.shape
        assert n_features == self.n_features, f"{n_features} != {self.n_features}"
        
        col_batch_mean = torch.mean(inputs, -2, keepdim=True)

        n_total_samples = self.n_samples_seen + n_samples
        col_mean = self.running_mean * self.n_samples_seen
        col_mean += torch.sum(inputs, -2, keepdim=True)
        col_mean /= n_total_samples

        mean_correction = math.sqrt((self.n_samples_seen * n_samples) / (self.n_samples_seen + n_samples)) * (self.running_mean - col_batch_mean)

        x = inputs - col_batch_mean
        if self.n_samples_seen != 0:
            x = torch.concat(
                [
                    torch.reshape(self.singular_vals, [-1, 1]) * self.components,
                    inputs - col_batch_mean,
                    mean_correction,
                ],
                dim=0,
            )
        _, s, v = torch.svd(x)
        v = v.T
        max_abs_rows = torch.argmax(torch.abs(v), dim=1)
        signs = torch.sign(v[range(v.shape[0]), max_abs_rows])
        v *= signs.reshape((-1,1))

        self.explained_variance = torch.square(s) / (n_total_samples - 1)
        self.noise_variance     = torch.mean(self.explained_variance[self.n_components:])
        self.components      = v[:self.n_components]
        self.singular_vals   = s[:self.n_components]
        self.running_mean    = col_mean
        self.n_samples_seen += n_samples
        self.step           += 1 
    
    def transform(self, inputs, n_components=1024):
        return torch.matmul((inputs - self.running_mean), self.components[:n_components].T)
    
    def invtransform(self, inputs, n_components=1024):
        return torch.matmul(inputs, self.components[:n_components]) + self.running_mean

"""
ipca1 = IPCA(16, 256)
ipca2 = IPCA(16, 256)

A = torch.rand(1024, 256)         
   
for i in range(8):
    ipca1.fit(A[i*128:(i+1)*128])

ipca2.fit(A)

B = ipca1.transform(A)
C = ipca1.invtransform(B)
print(((A-C)**2).mean())

B = ipca2.transform(A)
C = ipca2.invtransform(B)
print(((A-C)**2).mean())
"""