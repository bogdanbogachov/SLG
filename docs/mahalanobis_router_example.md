# Mahalanobis router: a worked example

A toy walk-through of every operation in `hidden_router/router.py` (`_LayerStats` and `MahalanobisRouter`),
with 2 feature dimensions instead of 2048, 2 experts (A, B), 2 training questions per expert and one fixed
layer `l`. All numbers were reproduced with the actual code (`_LayerStats` on these inputs, `alpha = 0.2`).

| Symbol | Meaning | In the code |
|---|---|---|
| `h(q)` | question vector: mean of the question's token states at layer `l`, BOS excluded | `features[:, layer, :]` |
| `m` | global mean of `h` over all training questions | `stats.center` |
| `sigma` | per-dimension pooled **within-expert** standard deviation of `h` | `stats.scale` |
| `z(q)` | standardized feature `(h - m) / sigma` | (implicit) |
| `mu_k` | mean of `z` over expert `k`'s training questions | `stats.means` |
| `C` | pooled within-expert covariance of `z` (unit diagonal, i.e. a correlation matrix) | `stats.cov` |
| `S_alpha` | shrunk covariance `(1 - alpha) C + alpha I` | `shrunk` in `weights()` |
| `W`, `b` | router weights and biases, `sigma` folded into `W` | `weights()` output |

## 1. Token states -> question vector `h(q)`

Token states of each question at layer `l` are averaged. The BOS token is dropped: its state is the same for
every question (say `[100, -50]`), so it carries no routing information and would only add a large constant.

| Expert | Question | Token states (no BOS) | `h(q)` |
|---|---|---|---|
| A | q1 | [1,2], [3,4] | **[2, 3]** |
| A | q2 | [3,2], [5,4] | **[4, 3]** |
| B | q3 | [6,8], [8,8] | **[7, 8]** |
| B | q4 | [9,6], [9,8] | **[9, 7]** |

From here on every question is one vector: `N = 4` questions, `K = 2` experts.

## 2. Global mean `m`

`m = mean(h) = [(2+4+7+9)/4, (3+3+8+7)/4] = [5.5, 5.25]`

## 3. Pooled within-expert standard deviation `sigma`

Expert means of `h`: A = [3, 3], B = [8, 7.5]. Each question is compared with **its own expert's mean**:

- deviations: q1 [-1, 0], q2 [1, 0], q3 [-1, 0.5], q4 [1, -0.5]
- `sigma_1^2 = (1 + 1 + 1 + 1) / (N - K) = 4 / 2 = 2`, so `sigma_1 = 1.414`
- `sigma_2^2 = (0 + 0 + 0.25 + 0.25) / 2 = 0.25`, so `sigma_2 = 0.5`

`sigma = [1.414, 0.5]`. This is the within-expert spread, not the global one; that is why `C` below has a
unit diagonal.

## 4. Standardize: `z = (h - m) / sigma`

| Question | `z` |
|---|---|
| q1 | [-2.475, -4.5] |
| q2 | [-1.061, -4.5] |
| q3 | [1.061, 5.5] |
| q4 | [2.475, 3.5] |

## 5. Class means `mu_k` (z-space, own expert's questions only)

- `mu_A = mean(z(q1), z(q2)) = [-1.768, -4.5]`
- `mu_B = mean(z(q3), z(q4)) = [1.768, 4.5]`

## 6. Shared covariance `C`, then shrinkage `S_alpha`

Deviations from the own class mean: q1 [-0.707, 0], q2 [0.707, 0], q3 [-0.707, 1], q4 [0.707, -1].
Summing their outer products and dividing by `N - K = 2`:

```
C = [[ 1.000, -0.707],
     [-0.707,  1.000]]        (diagonal = 1 by construction)
```

With `alpha = 0.2`:

```
S_alpha = 0.8 * C + 0.2 * I = [[ 1.000, -0.566],
                               [-0.566,  1.000]]
```

`C` has rank at most `N - K`. In the real data (`D = 2048`) this is barely above `D` on CQADupStack, so `C`
is ill-conditioned; any `alpha > 0` makes `S_alpha` positive definite, so the Cholesky factorization exists.

## 7. Weights and biases (closed form)

`w_k = S_alpha^-1 mu_k` is computed with one Cholesky factorization of `S_alpha` and a forward and a backward
triangular solve, with all `K` class means as right-hand sides (`torch.cholesky_solve`).

- `w_A = [-6.343, -8.088]`, `w_B = [6.343, 8.088]`
- `b_k = -0.5 * mu_k . w_k + log(pi_k)`, with prior `pi_k = 2/4`: `b_A = b_B = -24.498`

## 8. Route a new question

Tokens (no BOS) [5,7], [7,7], so:

- `h = [6, 7]`
- `z = [(6 - 5.5)/1.414, (7 - 5.25)/0.5] = [0.354, 3.5]`
- `s_A = w_A . z + b_A = -55.05`
- `s_B = w_B . z + b_B = 6.05`

The posterior `softmax(s)` gives `p(B | q) ~ 1`, so expert B is selected and its LoRA adapter generates the
answer.

**Folding `sigma` into `W`.** The stored router computes `s = W (h - m) + b` with `W_k = w_k / sigma`:
`W_B = [4.485, 16.177]`, and `(h - m) = [0.5, 1.75]` gives `2.243 + 28.310 - 24.498 = 6.05`, the same score.
This is what `weights()` returns and what `MahalanobisRouter.scores()` applies.

## 9. Layer and shrinkage selection

Steps 2-7 run for every layer `l = 0..16` and every `alpha` in `{0.05, 0.2, 0.5}`. Each fit uses the fit
part of the training questions, and is scored on a stratified 20% validation slice; experts with fewer than
3 training questions stay entirely in the fit part. The pair is chosen by validation top-1, then top-5, then
the earlier layer, then the smaller `alpha` (Python's `max` keeps the first maximum in the grid order).
The router is then refit on all training questions with the chosen `(layer, alpha)`.

In the real setting, `D = 2048`, up to `K = 276` experts and about `N = 18,000` training questions.

## Applying the router to an unseen question

Nothing is fitted at test time. `m`, `W` and `b` are computed once from the training questions and frozen.
A new question is only scored:

```
s = W (h - m) + b
```

| Object | Shape | Set by |
|---|---|---|
| `h`, `m` | `D` | hidden size of the base model (`D = 2048` for Llama-3.2-1B) |
| `W` | `K x D` | one row per expert; `K` = number of experts (knowledge units in the training data) |
| `b` | `K` | one bias per expert |
| `s` | `K` | one score per expert; `argmax(s)` is the selected expert, `softmax(s)` the posterior |

`h` is the question's mean-pooled hidden state at the selected layer `l*` (BOS excluded). Row `k` of `W` is
`S_alpha^-1 mu_k` divided element-wise by `sigma`, so `K` comes from the number of class means, while `D`
comes from the base model and never changes. Adding an expert adds one row to `W` and one entry to `b`.
(In the toy example `K = D = 2`, so `W` happens to be square; with 3 experts it would be `3 x 2`.)

## Mahalanobis or LDA?

Both names describe the same router, from two angles:

- **Mahalanobis distance** is the distance it uses: `(z - mu_k)^T S_alpha^-1 (z - mu_k)`, i.e. distance to
  an expert's mean measured in units of the within-expert spread and correlation, not plain Euclidean
  distance.
- **LDA (linear discriminant analysis)** is the classifier this distance comes from: every expert is a
  Gaussian with its own mean and one covariance shared by all experts. Choosing the most probable expert
  under that model means choosing the expert with the smallest Mahalanobis distance, adjusted by the log
  prior. Because the covariance is shared, the quadratic terms cancel and the rule becomes linear, which is
  why it reduces to `s = W (h - m) + b`.
- **Shrinkage** (`alpha`) regularizes the shared covariance, so the method is *shrinkage LDA*.

LDA is therefore part of the method, not a baseline. The baselines change either the features (the same
classifier on TF-IDF or on token embeddings) or the classifier (nearest centroid, which is LDA with the
covariance replaced by the identity; k-NN; a gradient-trained logistic probe; the fine-tuned LM classifier).

## Adding an expert

Keep, per expert, the question count and the feature sum, plus the pooled within-expert scatter matrix of
the unstandardized `h`. A new expert adds its own count, sum and scatter. Then `sigma` (from the scatter's
diagonal), `C`, `S_alpha` and `W, b` are recomputed in closed form. The global mean `m` changes too, but
centering does not change which expert wins. The layer may need reselecting (step 9).
