# Sampling
In sequence-to-sequence models like GPT or Transformer-based architectures, generating an output sequence (e.g., text) involves making a series of choices for each element in the sequence. The method by which we make these choices is termed as 'sampling.' Various sampling techniques can be employed, each with its own set of advantages and trade-offs. In this post, we'll zero in on greedy sampling and beam search.

### Greedy Sampling
In greedy sampling, the word with the highest conditional probability is selected as the next word in the sequence, given the previous words.
```python
def greedy_sampling(model, input_sequence):
    output_sequence = []
    for i in range(MAX_LENGTH):
        next_word_probabilities = model.predict(input_sequence)
        next_word = argmax(next_word_probabilities)
        output_sequence.append(next_word)
        input_sequence = update_input(input_sequence, next_word)
    return output_sequence
```
* **Advantages:** It's computationally efficient and straightforward to implement.
* **Limitations:** Greedy sampling often results in suboptimal and repetitive sequences. Since it doesn't explore other probable words, it can get stuck in a 'local optimum.'

### Beam Search
Beam search is an extension of greedy search that aims to improve the quality of the generated sequences by maintaining a 'beam' of the most promising partial sequences at each decoding step. The core principle of beam search is to keep track of not just a single best prediction at each time step, but a fixed number, `B` of best predictions. At each time step, the algorithm considers expanding each of these `B` sequences with all possible next elements and retains only the top `B` sequences based on their probabilities up to the current time step.

Here is a basic NumPy-based function to illustrate a simplified version of beam search:

```python
import numpy as np

def beam_search_decoder(probs, beam_size=3):
    sequences = [[[], 1.0]]  # list of [sequence, sequence_probability]
    
    for prob in probs:  # loop through each time step
        all_candidates = []
        
        for seq, seq_prob in sequences:
            for idx, p in enumerate(prob):
                candidate = [seq + [idx], seq_prob * p]
                all_candidates.append(candidate)
        
        # Sort all candidates by probability
        ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
        
        # Select top-k based on beam size
        sequences = ordered[:beam_size]
    
    return sequences
```
Here's the breakdown:

1. **Initialization:** Begin with a single sequence containing just the start token and with a probability of 1.
2. **Sequence Expansion:** At each time step, expand each sequence in the beam by all possible next elements.
3. **Pruning:** Sort all possible sequences by their probabilities and retain only the top `B` sequences.

#### When to Use Beam Search
Beam search strikes a balance between the breadth of exploration and computational expense. It is often used in applications where the quality of the generated sequence is critical and some level of determinism is acceptable.

#### Limitations and Challenges
* **Search Space:** The algorithm still explores a limited space, defined by the beam size. A small `B` size could yield sub-optimal sequences, while a larger one would be computationally expensive.
* **Length Normalization:** Beam search tends to favor shorter sequences over longer ones. Various strategies, like length normalization, have been proposed to mitigate this.