# KV Cache
Key-Value (KV) caching is a technique used to accelerate the inference process in machine learning models, particularly in autoregressive models. In these models, generating tokens one by one is a common practice, but it can be computationally expensive because it repeats certain calculations at each step. To address this, KV caching comes into play. It involves caching the previous keys and values, so we don’t need to recalculate them for each new token. This significantly reduces the size of matrices used in calculations, making matrix multiplications faster. The only trade-off is that KV caching requires more GPU memory (or CPU memory if a GPU isn’t used) to store these states.

```python
class KVCache:
    def __init__(self, max_batch_size, max_seq_len, n_kv_heads, head_dim, device):
        self.cache_k = torch.zeros((max_batch_size, max_seq_len, n_kv_heads, head_dim)).to(device)
        self.cache_v = torch.zeros((max_batch_size, max_seq_len, n_kv_heads, head_dim)).to(device)

    def update(self, batch_size, start_pos, xk, xv):
        self.cache_k[:batch_size, start_pos :start_pos + xk.size(1)] = xk
        self.cache_v[:batch_size, start_pos :start_pos + xv.size(1)] = xv

    def get(self, batch_size, start_pos, seq_len):
        keys = self.cache_k[:batch_size,  :start_pos + seq_len]
        values = self.cache_v[:batch_size, :start_pos + seq_len]
        return keys, values
```