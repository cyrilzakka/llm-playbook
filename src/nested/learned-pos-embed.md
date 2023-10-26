# Learned Positional Embeddings
In contrast to fixed positional embeddings like [sinusoidal encoding](./fixed-pos-embed.md), another popular approach is learned positional embeddings. Here, instead of hard-coding the logic for computing positional encodings, we make the model learn the best possible representation for sequence position during the training phase. Like other parameters in the model, these learned positional embeddings get fine-tuned through backpropagation.

The learned positional embeddings offer the model flexibility and adaptability. They can be designed to have the same shape as the input sequence, thus making them directly addable to the token embeddings. In it simplest form, learned positional embeddings can be defined as:
```python
pos_emb_shape = (1, seq_len, d_model) # 1
pos_embedding = np.random.randn(*pos_emb_shape) # 2
x += pos_embedding # 3
```
1. **Initialize Embedding Shape:** The first line of code sets up the shape for the positional embedding array. The shape `(1, seq_len, d_model)` indicates that we'll have:
`1` to denote it's a single tensor that will be broadcasted across multiple batches,
`seq_len` as the length of the sequence to which the positional embedding will be added, and
`d_model` as the dimensions of the model, which should match the dimension of the input sequence embeddings. This ensures that we can add the positional embedding directly to the token embeddings.
2. **Random Initialization:** In the second line, we initialize the positional embedding array with random values from a normal distribution. This serves as a starting point for what the model will later refine during training. These embeddings are considered parameters and are fine-tuned during the backpropagation process.
3. **Add to Input Sequence:** Finally, we add the positional embedding array to the input sequence x. This is done element-wise and serves to encode the position information within each token's embedding. This combined representation is then passed through the model for further processing.