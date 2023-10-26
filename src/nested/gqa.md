# Grouped-Query Attention (GQA)
Grouped Query Attention (GQA) extends the concept of Multi-Head Attention (MHA) and Multi-Query Attention (MQA) by providing a flexible trade-off between computational efficiency and model expressiveness. In GQA, query heads are divided into `G` groups, where each group shares a common key (K) and value (V) projection. This configuration enables three notable variations:

* **GQA-1:** A single group, which equates to [Multi-Query Attention (MQA)](./mqa.md).
* **GQA-H:** Groups equal to the number of heads, essentially the same as [Multi-Head Attention (MHA)](./mha.md).
* **GQA-G:** An intermediate configuration with `G` groups, balancing between efficiency and expressiveness.

The use of `G` groups allows GQA to mitigate the memory overhead associated with storing keys and values for each head, especially in scenarios with large context windows or batch sizes. At the same time, it offers a nuanced control over the model's quality and efficiency.

In its simplest form, GQA can be implemented as follows:
```python
class  GroupedQueryAttention(Attention):
    r"""
    https://arxiv.org/pdf/2305.13245.pdf
    """
    def __init__(self, word_size: int = 512, embed_dim: int = 64,
                 n_grouped: int = 4, n_query_each_group:int=2) -> None:
        super().__init__(word_size, embed_dim)
        delattr(self, 'query')
        delattr(self, 'key')
        delattr(self, 'value')

        self.grouped = nn.ModuleList([
            MultiQueryAttention(word_size, embed_dim, n_query=n_query_each_group)
            for _ in range(n_grouped)
        ])
        # self.proj = nn.Parameter(torch.empty((..., ...), requires_grad=True))
        self.proj = nn.Parameter(torch.empty(embed_dim * n_grouped, embed_dim))
        nn.init.xavier_uniform_(self.proj)

    def forward(self, x: Tensor) -> Tensor:
        Z_s = torch.cat([head(x) for head in self.grouped], dim=1)
        Z = torch.matmul(Z_s, self.proj)
        return Z
```