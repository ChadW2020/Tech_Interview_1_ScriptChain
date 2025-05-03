

"""
This is an implementation of learnable relative positional encoding method through Pytorch.

It implemented several efficient tricks to accelerate the calculation.

This whole answer is learned from a blog from Ngieng Kianyew:
https://medium.com/@ngiengkianyew/what-is-relative-positional-encoding-7e2fbaa3b510

I also got help from a blog from Jake Tae:
https://jaketae.github.io/study/relative-positional-encoding/
"""

import torch

batch_size = 16# B = 16
time_steps = 100# T = 100
embedding_dim = 6# D = 6

positional_embedding = torch.nn.Parameter(torch.randn(2 * time_steps - 1, embedding_dim))# torch.Size([2 * T - 1, D]); This will be our learnable positional encoding matrix. This matrix is learnable in the training process by setting requires_grad = True.

# X is a batch of input sequences (B, T)
# Set my dummy dataset, filled with Gaussian random numbers
X = torch.randn(batch_size, time_steps)
K = torch.randn(time_steps, time_steps)# initialize the K matrix, (T, T)
XK = torch.matmul(X, K)# (B, T)
relative_positional_encoding = torch.matmul(XK, positional_embedding.transpose(0, 1))
assert relative_positional_encoding.shape == (batch_size, time_steps, 2 * time_steps - 1)# Check input shape (B, T, 2 * T - 1)

"""
The following steps is a trick of selecting the required T * T elements from a matrix of (T, 2 * T -1) without actually doing a backward sliding window.
"""

padding = torch.zeros((batch_size, time_steps, 1))# (B, T, 1)
rpe_padded = torch.cat([relative_positional_encoding, padding], dim=-1)# (B, T, 2T - 1) -> (B, T, 2T)
rpe_padded_flat = rpe_padded.flatten(start_dim=1)# (B, 2 * T * T)
padding_2 = torch.zeros((batch_size, time_steps - 1))# (B, T - 1)
rpe_padded_flat_padded = torch.cat([rpe_padded_flat, padding_2], dim=-1)# (B, 2 * T * T + T - 1)
rpe_padded_flat_padded_reshape = rpe_padded_flat_padded.reshape(batch_size, time_steps + 1, 2 * time_steps - 1)# (B, T + 1, 2 * T - 1)
rpe_key = rpe_padded_flat_padded_reshape[:, : -1, -time_steps :]# (B, T, T)

"""
rpe_key is calculated from key matrix, X * K, and then added to it:

e_ij = (x_i * Q * (x_j * K + rpe_key_ij)) / sqrt(d)

and goes to attention calculation:

alpha_ij = exp(e_ij) / (sum_k(exp(e_ik)))

rpe_value can be similarly calculated from XV matrix X * V, and added in the self-attention layer output:

z_i = sum_j(alpha_ij * (x_j * V + rpe_value_ij))
"""


















