# Multi-Headed Attention (MHA)
In a single attention mechanism, each token gets a chance to focus on other parts of the sequence. However, there's a limit to what it can capture this way. Multi-headed attention solves this by running not one but multiple attention layers in parallel, essentially allowing the model to pay attention to different parts of the input for different reasons.
This can naively be implemented in the following way:
```python
class MultiheadAttention(nn.Module):
    r"""
    https://arxiv.org/abs/1706.03762
    """
    def __init__(self, word_size: int = 512, embed_dim: int = 64, n_head:int=8) -> None:
        super().__init__()
        self.n_head = n_head
        self.embed_dim = embed_dim
        self.dim_K = torch.tensor(embed_dim)
        self.proj = nn.Parameter(torch.empty(embed_dim * n_head, embed_dim))
        nn.init.xavier_uniform_(self.proj)
        self.multihead = nn.ModuleList([
            Attention(word_size, embed_dim) for _ in range(n_head)
        ])

    def forward(self, x: Tensor) -> Tensor:
        Z_s = torch.cat([head(x) for head in self.multihead], dim=1)
        Z = torch.matmul(Z_s, self.proj)
        return Z
```
Why is this useful?
* **Diverse Representations:** Having multiple heads allows the model to recognize various types of relationships between tokens, which can be critical for understanding complex structures like sentences.
* **Increased Capacity:** Multi-headed attention increases the model's capacity to learn, as each head can potentially learn different aspects of the data. Think of it like having multiple detectives on the case instead of just one.
* **Parallelism:** Multiple heads can be processed in parallel, providing a computational advantage. Imagine splitting the detective work, where each detective specializes in a different type of evidence.

Keep in mind that the number of heads and their dimensions are hyperparameters that you'll have to fine-tune based on your specific application. More heads are not always better; it's about striking the right balance between model complexity and performance.