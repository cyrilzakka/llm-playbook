# Multi-Query Attention (MQA)
Multi-Query Attention (MQA) is a refined version of the [Multi-Head Attention (MHA)](./mha.md) algorithm that improves computational efficiency without sacrificing much in terms of model accuracy. In standard MHA, separate linear transformations are applied to the Query (Q), Key (K), and Value (V) for each attention head. MQA diverges from this by using a single shared set of Keys (K) and Values (V) across all heads, while allowing individual transformations for each Query (Q). Although this approach was first introduced in 2019, it has only been recently popularized by models such as [PaLM](https://arxiv.org/pdf/2204.02311.pdf) and [Falcon](https://arxiv.org/abs/2306.01116). This is illustrated below:

```python
class  MultiQueryAttention(Attention):
    r"""
    https://arxiv.org/pdf/1911.02150.pdf
    """
    def __init__(self, word_size: int = 512, embed_dim: int = 64, n_query:int=8) -> None:
        super().__init__(word_size, embed_dim)
        self.n_query = n_query
        self.proj = nn.Parameter(torch.empty(embed_dim * n_query, embed_dim))
        nn.init.xavier_normal_(self.proj)
        delattr(self, 'query')
        self.querys = nn.ModuleList([
            nn.Linear(in_features=word_size, out_features=embed_dim, bias=True)
            for _ in range(n_query)
        ])
        self.key = nn.Linear(in_features=word_size, out_features=embed_dim, bias=True)
        self.value = nn.Linear(in_features=word_size, out_features=embed_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        K = self.key(x)
        V = self.value(x)
        Z_s = torch.cat([
            self.self_attention(query(x), K, V) for query in self.querys
        ], dim=1)
        Z = torch.matmul(Z_s, self.proj)
        return Z

```
with improvements in: 
* **Memory Space:** Sharing K and V across all heads dramatically reduces the memory footprint. This is critical for handling long sequences without choking your hardware.
* **Memory Bandwidth:** With fewer unique transformations, the computational cost in terms of memory bandwidth also drops.