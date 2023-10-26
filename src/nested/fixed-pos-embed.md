# Fixed Positional Embeddings
As previously mentioned, in the Transformer architecture positional encodings serve as a critical component for giving the model an understanding of the order of tokens in a sequence. Unlike recurrent networks, which inherently understand sequence order, the multi-head attention mechanism in the Transformer is non-recurrent and processes the entire sequence in parallel. Consequently, it lacks an innate sense of order among the data points.

To remedy this, the concept of positional encoding is employed. Specifically, a tensor that matches the shape of the input sequence is added to the input, and this tensor is designed such that the difference in values between any two positions reflects their distance in the sequence. This allows the model to understand the relative positions of tokens and treat them accordingly.

To this end, several methods for positional encoding have been proposed. Before we dive into more advanced methods for positional encoding, let's first debunk the shortcomings of seemingly intuitive solutions. Up first, one might think about normalizing time-step values between [0, 1] and using them for positional information:

```python
time_step_normalized = np.linspace(0, 1, num_tokens)
```
Though tempting, this approach is inherently flawed: the normalized values are dependent on sequence length, making it problematic for the model to handle sequences of varying lengths - a positional encoding value of 0.4 means something entirely different to a sequence of length 4 than to a sequence of length 80.

Similarly, one might advocate for a linear numbering scheme such as:
```python
time_step_linear = np.arange(1, num_tokens + 1)
```
Simple? Yes. Effective? Not quite. As sequence length inflates, positional values escalate, undermining the model's ability to generalize to sequences longer than those in the training set, while potentially leading to training instabilities (e.g. exploding gradients).

### Sinusoidal Positional Encoding

Among the various approaches proposed over time, the most widely used form of fixed positional embeddings is sinusoidal positional encoding. In this method, each position in the sequence is uniquely represented by a combination of sine and cosine functions at different frequencies. These sinusoidal embeddings are added to the input embeddings to supplement them with positional context.

```python
def sinusoidal_positional_encoding(position, d_model):
    angle_rads = np.arange(d_model) // 2 * np.pi / np.power(10000, 2 * (np.arange(d_model) // 2) / np.float32(d_model)) # 1
    angle_rads = position * angle_rads # 2
    pos_encoding = np.zeros(d_model) # 3
    pos_encoding[0::2] = np.sin(angle_rads[0::2]) # 4
    pos_encoding[1::2] = np.cos(angle_rads[1::2]) # 4
    return pos_encoding
```

Here, the function takes two arguments: `position` representing the position of a token in the sequence, and `d_model` being the dimension of the model's input embeddings.

1. **Initialize Angle Array:** We start by creating an array that will hold angle values for sine and cosine functions. These angles are calculated in such a way that they depend on both the position of a token in the sequence and its position in the embedding space. The calculations involve some scaling to ensure that the model handles different sequence lengths efficiently.
2. **Position-Based Scaling:** The next step is to multiply these pre-calculated angle values by the position of the token in the sequence. This ensures that each token position will have a unique set of angles.
3. **Initialize Encoding Array:** An array of zeros is then initialized. This array will hold the final positional encodings and has the same size as the embedding dimension of the model.
4. **Populate Sine and Cosine Values:** Finally, we populate this zero array with sine and cosine values based on the angle values we've computed. The sine values go into the even-indexed positions, and the cosine values go into the odd-indexed positions.
The end result is that each position in the sequence gets a unique pattern of sine and cosine values, making it distinguishable from other positions.

How exactly does this approach convey positional information? [Imagine a series of pendulums aligned in a straight line](https://www.blopig.com/blog/2023/10/understanding-positional-encoding-in-transformers/). Each pendulum is swinging at a different frequency, starting from the leftmost one, which swings the slowest, to the rightmost one, which swings the fastest. Now, imagine taking a snapshot of the pendulums at a certain time `t` where`t` corresponds to the token's position in the sequence.

In this snapshot, pendulums on the left have moved very little due to their slower frequencies, while those on the right have moved considerably. If you were to calculate the dot product (read: similarity) of their positions at this moment, the slow-swinging pendulums would be aligned closely and contribute positively to the dot product. In contrast, the fast-swinging pendulums would be out of phase and contribute noise around zero to the dot product.

As time (or position) progresses, the snapshot would capture more pendulums being out of phase, causing the dot product value to gradually converge to zero. This mirrors the behavior of the sinusoidal positional encoding: the dot product between the positional encodings of tokens that are close in sequence will be high, while the value will smoothly decrease for tokens that are further apart.

By mapping each token's position in the sequence to a unique combination of sinusoidal values, we effectively capture the relative positions and relationships between tokens. The encoded values at different positions can then be visualized, showing a high value for nearby tokens and a smoothly decreasing value as the distance between tokens increases.