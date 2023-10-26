# Rotary Positional Embeddings
Rotary Positional Embeddings aim to overcome limitations tied to both [fixed](./fixed-pos-embed.md) and [learned](./learned-pos-embed.md) positional embeddings. While fixed sinusoidal embeddings are generalizable to arbitrary sequence lengths in practice, models have been found to underperform when encountering sequences with lengths substantially different from their training data in practice. Enter rotary positional embeddings.

Rotary Positional Embeddings provide a flexible mechanism to include positional context into tokens, without modifying the original embeddings. The core principle revolves around rotating the queries and keys in the attention mechanism, where each position in the sequence receives a unique rotation. This way, the dot product between queries and keys gradually diminishes for tokens that are distant from one another in the sequence, providing an effective way to encode relative positions.

This approach tends to maintain more of the original token information while still providing the model with an effective way to understand sequence positions. Their implementation would look something like: 
```python
def rotary_positional_embedding(position, d_model):
    freqs = np.exp(np.linspace(0., -1., d_model // 2) * np.log(10000.)) # 1
    angles = position * freqs # 2
    rotary_matrix = np.stack([np.sin(angles), np.cos(angles)], axis=-1) # 3
    return rotary_matrix.reshape(-1, d_model) # 4
```
1. **Initialize Frequency Array:** Similar to the sinusoidal approach, we initiate an array of frequencies. The key difference here is the use of exponential scaling to generate frequencies, which will serve as rotation factors.
2. **Position-Based Scaling:** Next, we scale the positions by these frequencies. Unlike in sinusoidal encodings where the scaled positions would be added to the embeddings, here they are used for rotating the embeddings.
3. **Construct Rotary Matrix:** Using the scaled angles, a rotary matrix is created by stacking the sine and cosine of the angles. This matrix will serve to rotate the original embeddings.
4. **Reshape Rotary Matrix:** Finally, the rotary matrix is reshaped to match the model's embedding dimension, ensuring it's appropriately utilized to rotate the token embeddings. This rotation matrix is then embedded into the original vector by matrix multiplication instead of addition.

Simple enough! Let's conceptualize rotary positional embeddings by imagining a clock with multiple hands. Each hand rotates at a different speed, representing different frequencies. Every token in your sequence corresponds to a specific clock hand.

* **Variable Rotation Speed:** Just like in a real clock where the second, minute, and hour hands rotate at distinct speeds, different dimensions in the query/key embeddings are rotated differently. This can be thought of as each dimension having its own "frequency," determining how fast it rotates based on its position in the sequence.
* **Dot Product Significance:** When two clock hands point in the same or similar direction (i.e., their angles are close), they can be considered "similar" or "close" in sequence context. In the same vein, the dot product between rotated queries and keys would be higher for positions that are close in the sequence, and lower for positions that are farther apart.
As time progresses (or as you traverse through the sequence), each clock hand rotates based on its speed (frequency). When you look at the clock at any given "time" (or position in the sequence), the angles of the clock hands with respect to a fixed starting point provide a snapshot of the tokens' positions.
* **Invariance to Sequence Length:** Much like how the hands of a clock keep rotating indefinitely regardless of the 12-hour clock face, Rotary Positional Embeddings aren't restricted by the length of the sequence. This means they can adapt to sequences of varying lengths, offering a level of flexibility.
* **Impact on Attention:** Just as you could determine the elapsed time between different events by observing the relative angles between clock hands, rotary positional embeddings influence the attention mechanism. They help it focus on tokens that are contextually relevant to each other based on their positional relationships in the sequence.
By simply looking at how much each hand has rotated, you can figure out its relative position in the sequence. In this way, the rotational information captures the essence of each token's position within the overall sequence while leaving the actual token embeddings largely untouched.

In rotary positional embeddings, the same principle applies: each token's embedding gets "rotated" based on its position in the sequence. This rotational change encodes the positional information while retaining the original embedding, thus allowing the model to understand the tokens' relative positions effectively.