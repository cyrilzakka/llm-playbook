# Temperature
Temperature is a hyperparameter used to control the randomness in the probabilistic sampling of tokens (words, in most cases) from a distribution. It's applied to the logits (the raw scores or predictions) before the Softmax operation. Intuitively, you can think of the temperature as a knob to adjust how conservatively or liberally you want to sample the next token.

Here's the basic formula to apply temperature:
```python
import numpy as np

def apply_temperature(logits, temperature):
    logits = logits / temperature  # Apply temperature scaling
    probs = np.exp(logits) / np.sum(np.exp(logits))  # Softmax to get probabilities
    return np.random.choice(np.arange(len(logits)), p=probs)  # Sample from the distribution
```
Let's break down what happens:

1. **Scaling:** The logits are divided by the temperature. Lower temperature (< 1) makes the model more confident in its top choices, whereas a higher temperature (> 1) makes the model more uncertain, effectively flattening the distribution.
2. **Softmax:** After scaling, the logits are transformed into probabilities using the Softmax function.
3. **Sampling:** Finally, a word is sampled from this distribution.

### When to Use Temperature Scaling
Temperature is widely applicable across different sampling methods and provides fine-grained control over the randomness of output text. Whether you are using greedy decoding, Top-K, or nucleus sampling, adding a temperature parameter can help you adjust the output to meet specific quality-diversity criteria.

### Limitations and Considerations
* **Hyperparameter Tuning:** The choice of temperature can have a significant impact on your results.
* **Context-Insensitive:** Temperature scaling is not adaptive to the context, which may or may not be a limitation based on your use-case.