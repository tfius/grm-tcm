# GRM-TCM Mathematical Formulation

This document states the project idea in mathematical terms:

Can a Graph Resonance Model (GRM) learn predictive latent resonance structure
from longitudinal whole-body observations, and can coarse TCM-like descriptors be
mapped onto that structure as an interpretable semantic layer?

The central claim should be kept narrow. GRM is not assumed to prove TCM, Qi, or
any biological mechanism. It is a candidate latent-state and propagation model.
TCM enters only as an empirical annotation layer that must be tested against
predictive performance, stability, and non-random alignment.

## 1. Data

Let subject `i = 1, ..., N` be observed at times `t = 1, ..., T_i`.

Each visit has:

```text
x_{i,t} in R^p        observed variables
u_{i,t} in R^m        interventions, events, stressors, exposures
y_{i,t+1}             next-period outcome
c_{i,t}               optional semantic descriptor / TCM-like label
```

Examples of `x_{i,t}`:

```text
sleep, HRV, resting HR, temperature, fatigue, pain, appetite,
bowel quality, mood, energy, heaviness, cold-hot descriptor
```

Examples of prediction targets:

```text
y^{score}_{i,t+1}     next-day global state score
y^{flare}_{i,t+1}     flare / crash probability
y^{response}_{i,t+h}  response after intervention within h days
```

The model should be judged first on `y`, not on whether labels sound meaningful.

## 2. Latent-State View

Assume each visit is generated from an unobserved physiological state:

```math
z_{i,t} in R^d
```

with dynamics:

```math
z_{i,t+1} = F_i z_{i,t} + B_i u_{i,t} + eta_{i,t}
```

and observations:

```math
x_{i,t} = H z_{i,t} + epsilon_{i,t}
```

The true `z_{i,t}` is not observed in real data. GRM is used to construct a
data-driven approximation to the geometry and propagation structure of these
latent states.

## 3. Visit Graph

Create one graph node per visit:

```math
v = (i,t)
```

The graph encodes multiple notions of similarity or coupling:

```math
W = W^{obs} + alpha_t W^{time} + alpha_u W^{intervention}
```

where:

```math
W^{obs}_{ab} = exp(-||x_a - x_b||^2 / (2 sigma^2))
```

for feature-near visits,

```math
W^{time}_{(i,t),(i,t+1)} > 0
```

for within-subject temporal continuity, and `W^{intervention}` links visits with
similar intervention or event context.

Define the degree matrix:

```math
D_{aa} = sum_b W_{ab}
```

and either the unnormalized Laplacian:

```math
L = D - W
```

or the normalized Laplacian:

```math
L_{norm} = I - D^{-1/2} W D^{-1/2}
```

## 4. GRM Spectral Modes

Compute low-frequency graph modes:

```math
L phi_k = lambda_k phi_k
```

or for the normalized Laplacian:

```math
L_{norm} phi_k = lambda_k phi_k
```

Small `lambda_k` modes are smooth over the graph and represent broad latent
patterns shared across visits.

A GRM embedding for visit `a` is:

```math
e_a = [g_1 phi_1(a), ..., g_K phi_K(a)]
```

with spectral weights:

```math
g_k(rho) = 1 / (1 + rho^2 lambda_k)
```

Here `rho` is a resonance or propagation scale. Larger `rho` suppresses
high-frequency modes more strongly.

## 5. Resonance Operator

The GRM propagation kernel is:

```math
G_rho = Phi diag(g_1(rho), ..., g_K(rho)) Phi^T
```

Equivalently, in the full-rank idealization:

```math
G_rho approx (I + rho^2 L)^{-1}
```

up to the chosen Laplacian convention and omitted constant/null modes.

Interpretation:

```text
G_rho[a,b]      how strongly a perturbation at visit b propagates to visit a
G_rho[a,a]      self-resonance / persistence / attractor-like score
row entropy     transition uncertainty or spread of influence
```

This makes "resonance" an operational quantity, not a metaphysical claim.

## 6. Dynamic GRM

For time-varying dynamics, compute a state vocabulary and a rolling propagator.

Assign each visit to a discrete latent state:

```math
s_{i,t} in {1, ..., K_s}
```

or soft state weights:

```math
pi_{i,t,k} = P(s_{i,t}=k | x_{i,t})
```

For each rolling window ending at time `tau`, build a state-state graph:

```math
W_tau = beta_x W^{state-feature}
      + beta_T W^{transition}
      + beta_u W^{treatment}
```

Compute:

```math
L_tau = I - D_tau^{-1/2} W_tau D_tau^{-1/2}
```

and:

```math
G_tau = Phi_tau diag(1 / (1 + r_tau^2 lambda_{tau,k})) Phi_tau^T
```

Dynamic resonance scores:

```math
self_resonance_{i,t} = G_tau[s_{i,t}, s_{i,t}]
```

Soft version:

```math
soft_self_resonance_{i,t}
  = sum_k pi_{i,t,k} G_tau[k,k]
```

Regime-change score:

```math
Delta_tau = ||G_tau - G_{tau-1}||_F
```

These are the model's candidate "stuckness", "attractor", or "regime shift"
signals.

## 7. Prediction Model

Static prediction:

```math
E[y_{i,t+1} | x_{i,t}] = f(e_{i,t})
```

For continuous outcomes:

```math
hat y^{score}_{i,t+1} = theta_0 + theta^T e_{i,t}
```

For flare risk:

```math
P(y^{flare}_{i,t+1}=1 | e_{i,t})
  = sigmoid(theta_0 + theta^T e_{i,t})
```

Dynamic prediction can include resonance scores:

```math
P(y^{flare}_{i,t+1}=1)
  = sigmoid(theta_0
            + theta_e^T e_{i,t}
            + theta_s self_resonance_{i,t}
            + theta_d Delta_t
            + theta_u^T u_{i,t})
```

A GRM claim is credible only if this improves out-of-sample prediction over
simple baselines.

## 8. Transition Model

Let empirical Markov transitions be:

```math
T_{ab} = P(s_{t+1}=b | s_t=a)
```

GRM gives a propagation matrix over states. Convert it into a row-normalized
influence matrix:

```math
R_{ab} = normalize_row(max(G_{ab} - min(G), 0))
```

Blend empirical transition and GRM propagation:

```math
P_{GRM}(s_{t+1}=b | s_t=a)
  = alpha T_{ab} + (1 - alpha) R_{ab}
```

Evaluate using:

```text
top-1 accuracy
log loss
Brier score
expected calibration error
```

The transition model should be compared to:

```text
empirical Markov
Laplace-smoothed Markov
subject-personalized Markov
raw-feature classifier
state + dwell + load baseline
```

## 9. Mapping to TCM-Like Descriptors

Let `c_{i,t}` be a coarse semantic descriptor, for example:

```text
depleted / energized
calm / agitated
flowing / stuck
cold / hot
dry / damp-heavy
```

Do not force these labels into the model before latent states are learned.
Instead, learn GRM states first, then estimate:

```math
P(c = l | s = k)
```

or, for continuous descriptors:

```math
E[c_j | e_{i,t}]
```

Useful alignment scores:

```math
MI(S; C)       mutual information between GRM states and descriptors
ARI(S, C)      adjusted rand index if descriptors are discrete
AUC(c_j | e)   predictability of each descriptor from GRM coordinates
```

A descriptor maps cleanly to GRM only if:

```text
1. it is predictable from GRM states better than chance,
2. the mapping is stable under retraining / resampling,
3. it adds predictive value for outcomes or intervention response,
4. it does not disappear after controlling for obvious confounders.
```

Thus "stuck" may be mapped to high self-resonance only if:

```math
E[self_resonance | c = stuck] > E[self_resonance | c != stuck]
```

and the difference is stable, predictive, and not reducible to today's symptom
severity.

## 10. Intervention Response

Let `u_{i,t}` indicate an intervention class. Define response over horizon `h`:

```math
response_{i,t,h} = y_{i,t+h} - y_{i,t}
```

or a binary improvement:

```math
r_{i,t,h} = 1[ y_{i,t+h} - y_{i,t} <= -delta ]
```

Estimate whether interventions change state transitions:

```math
P(s_{t+h}=b | s_t=a, u_t=1)
  - P(s_{t+h}=b | s_t=a, u_t=0)
```

and whether the effect depends on resonance:

```math
P(r=1) = sigmoid(theta_0
                 + theta_s self_resonance_t
                 + theta_u u_t
                 + theta_{su} self_resonance_t * u_t
                 + controls)
```

This is where a TCM-style claim could become testable:

```text
For subjects in a high-stuckness/high-resonance state, does intervention class A
increase the probability of moving to a lower-risk state?
```

## 11. Falsifiable Hypotheses

The project should test claims in this order:

```text
H1: GRM modes recover stable latent state structure better than raw clustering.
H2: GRM/resonance features improve next-period outcome prediction over baselines.
H3: Dynamic self-resonance predicts persistence, stuck states, flares, or crashes.
H4: GRM transition kernels improve transition prediction and calibration.
H5: Coarse TCM-like descriptors align with GRM states above chance.
H6: Descriptor-aligned states show different intervention response probabilities.
```

Failures are informative. For example, if H2 fails, the model may still be useful
as a diagnostic visualization, but not as a predictive model.

## 12. Go / No-Go Criteria

Go:

```text
GRM beats simple baselines on at least one real predictive task.
Resonance scores are stable across seeds and resampling.
TCM-like descriptors attach to latent states non-randomly.
Descriptor mapping survives control for sleep, stress, caffeine, and severity.
Intervention effects are reproducible within state strata.
```

No-go:

```text
No predictive lift over yesterday / moving average / raw-feature models.
Modes are unstable under retraining.
Resonance is just a monotone transform of current symptom severity.
TCM-like labels do not align with learned states above chance.
Intervention effects vanish after controlling for obvious confounders.
```

## 13. Important Distinctions

Transductive GRM:

```text
The graph is built over all visits, then modes are analyzed. This is useful for
benchmark diagnostics and exploratory structure discovery.
```

Inductive GRM:

```text
The graph and modes are fit on training subjects only. New visits are projected
into the learned representation. This is required for honest deployment claims.
```

Semantic mapping:

```text
TCM descriptors are labels attached after learning, not assumptions baked into
the geometry.
```

Mechanistic biology:

```text
A predictive resonance model is not automatically a biological mechanism.
Molecular, physiological, or intervention evidence would be needed separately.
```

## 14. One-Sentence Formulation

We model each visit as a node in a multi-relational longitudinal graph, use the
GRM operator `G_rho = Phi diag(1 / (1 + rho^2 lambda)) Phi^T` to define latent
propagation and self-resonance, test whether these quantities improve prediction
of future state and intervention response, and only then map coarse TCM-like
descriptors onto the learned states as an empirical semantic overlay.
