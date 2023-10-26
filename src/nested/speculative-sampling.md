# Speculative Sampling
In **speculative sampling**, we have two models:

1. A smaller, faster **draft model** (e.g. DeepMind's 7B Chinchilla model)
2. A larger, slower **target model** (e.g. DeepMind's 70B Chinchilla model)

The idea is that the draft model *speculates* what the output is  steps into the future, while the target model determines how many of those tokens we should *accept*. Here's an outline of the algorithm:

1. The draft model decodes  tokens in the regular autoregressive fashion.
2. We get the probability outputs of the target and draft model on the new predicted sequence.
3. We compare the target and draft model probabilities to determine how many of the  tokens we want to keep based on some **rejection criteria**. If a token is rejected, we **resample** it using a combination of the two distributions and don't accept any more tokens.
4. If all  tokens are accepted, we can sample an additional final token from the target model probability output.

```python
def max_fn(x):
    x_max = np.where(x > 0, x, 0)
    return x_max / np.sum(x_max)

def speculative_sampling(x, draft_model, target_model, N, K):
    # NOTE: paper indexes arrays starting from 1, python indexes from 0, so
    # we have to add an extra -1 term when indexing using n, T, or t
    n = len(x)
    T = len(x) + N

    while n < T:
        # Step 1: auto-regressive decode K tokens from draft model and get final p
        x_draft = x
        for _ in range(K):
            p = draft_model(x_draft)
            x_draft = np.append(x_draft, sample(p[-1]))

        # Step 2: target model forward passes on x_draft
        q = target_model(x_draft)

        # Step 3: append draft tokens based on rejection criterion and resample
        # a token on rejection
        all_accepted = True
        for _ in range(K):
            i = n - 1
            j = x_draft[i + 1]
            if np.random.random() < min(1, q[i][j] / p[i][j]):  # accepted
                x = np.append(x, j)
                n += 1
            else:  # rejected
                x = np.append(x, sample(max_fn(q[i] - p[i])))  # resample
                n += 1
                all_accepted = False
                break

        # Step 4: if all draft tokens were accepted, sample a final token
        if all_accepted:
            x = np.append(x, sample(q[-1]))
            n += 1

        # just keeping my sanity
        assert n == len(x), f"{n} {len(x)}"

    return x
```