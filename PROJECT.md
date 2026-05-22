Be careful about the scientific boundary: this is framed as synthetic benchmark validation, not TCM validation. The main assumptions to probe are therefore technical: leakage, whether out-of-sample prediction truly matches training evaluation, and whether
dynamic “resonance” metrics are compared against baselines fairly.

# PHASE 1

## Layer A: whole-body observables

Collect repeated measures, ideally daily or at least 3x/week:

- sleep duration/quality
- resting HR / HRV
- body temperature
- fatigue
- pain
- appetite
- bowel quality
- mood / irritability / calmness
- perceived energy / “flow” / heaviness / cold-heat style descriptors

## Layer B: intervention log

acupuncture / massage / breathing / exercise / herbs / supplements / caffeine / alcohol / illness / stress events

## Layer C: sparse biochemical anchors

Weekly or biweekly if possible:

- CRP
- CBC
- ferritin / iron if relevant
- glucose / insulin or CGM if available
- cortisol or salivary cortisol if practical

## What the model should do

Use GRM-like machinery to infer:

- latent modes
- coupling between variables
- state transitions over time
- regime shifts

Then compare against dumb baselines:

- yesterday predicts today
- moving average
- logistic regression / XGBoost on raw features
- simple clustering on symptoms only

Go only if the latent-state model beats those baselines on at least one real task:

- next-day fatigue / pain / sleep quality
- flare prediction
- response to intervention
- regime-change detection

## The first prediction targets

Use targets that are concrete and frequent:

- next-day global state score
  Example composite from fatigue + mood + sleep + appetite + bowel + pain
- flare / crash event
  Binary deterioration window
- response to intervention
  Did state improve within 24–72 hours after an intervention?

These are measurable without knowing anything about TCM.

## Where TCM enters

After the latent states are learned.
Add a very crude semantic overlay:

- energized vs depleted
- calm vs agitated
- flowing vs stuck
- hot vs cold
- dry vs damp/heavy

Do not start with formal TCM ontology. Start with low-resolution descriptors and see whether they attach to stable latent modes. That imore likely to work.

## What would count as a positive result

A positive result is not “we found Qi.”
It is something like:

- latent state 2 predicts fatigue crashes two days ahead
- latent state 3 is associated with poor sleep + irritability + elevated resting HR
- one intervention class shifts people from state 3 to state 1 with measurable probability
- some naive holistic descriptors line up with those states better than chance

# PHASE 2

Add a small temporal **phosphoproteomics** or _proteomics_ substudy around state transitions:

- sample during baseline stable state
- sample during predicted flare / dysregulated state
- sample after recovery / response

Phosphoproteomics is especially relevant because it can infer kinase-state and dynamic pathway activation, which is much closer to regulatory state than static protein abundance.

At that point the question becomes:

**Do latent whole-body states have distinct pathway-level molecular signatures?**

That is a real bridge question.

## Where bioelectric / resonance ideas belong

Not as the first claim.

If the model finds clean latent states, and those states align with physiology and maybe molecular signaling, then bioelectric or oscillatory coordination becomes a candidate mechanism class. Reviews support endogenous electric fields as real biological control signals, including tissue and organ-level effects. But that is still a mechanism hypothesis, not something the first pilot can prove.

## Go / no-go in plain language

### Go

- you can collect repeated measurements for 8–12 weeks
- the latent model beats simple baselines
- modes are stable across retraining
- interventions shift states in reproducible ways
- holistic descriptors attach to modes non-randomly

### No-go

- no predictive gain over simple baselines
- latent modes are unstable
- everything is just noise or self-report variance
- intervention effects disappear after controlling for sleep/stress/caffeine

## Minimal dataset template

Per day:

- timestamp
- sleep hours
- sleep quality 1–10
- HR / HRV
- temperature
- fatigue 1–10
- pain 1–10
- appetite 1–10
- bowel quality 1–5
- mood calm↔agitated 1–10
- energy 1–10
- heaviness/lightness 1–10
- cold/hot 1–10
- interventions
- notes

## What synthetic data can prove

It can prove that your modeling architecture works when the truth is known.
It can test whether GRM can recover:

- hidden whole-body states
- coupling structure
- treatment-induced shifts
- early warning before flare
- overlap between latent states and simplified semantic labels
- What it cannot prove

# It cannot prove:

- that real TCM works by this mechanism
- that Qi exists
- that biology is actually organized the same way

So synthetic data is a pipeline validation step, not a scientific conclusion.

**Best synthetic design: Generate a causal latent-state simulator.**

Yes — that is a very good first move.

A **synthetic dataset with known hidden states and predictable outcomes** is exactly how you test whether the GRM-style pipeline can recover:

- latent modes
- transitions
- regime shifts
- intervention effects
- and coarse “Qi-like” descriptors

before touching real biology.

## What synthetic data can prove

It can prove that your **modeling architecture works** when the truth is known.

It can test whether GRM can recover:

- hidden whole-body states
- coupling structure
- treatment-induced shifts
- early warning before flare
- overlap between latent states and simplified semantic labels

## What it cannot prove

It cannot prove:

- that real TCM works by this mechanism
- that Qi exists
- that biology is actually organized the same way

So synthetic data is a **pipeline validation step**, not a scientific conclusion.

## Best synthetic design

Do not generate random noise with labels.
Generate a **causal latent-state simulator**.

### Core structure

#### 1. Hidden state

Let each subject have a latent vector:

[
z_t \in \mathbb{R}^k
]

Example hidden components:

- vitality / depletion
- stress activation
- inflammatory burden
- recovery capacity
- digestive stability

#### 2. State evolution

Let the hidden state evolve over time:

[
z_{t+1} = A z_t + B u_t + \epsilon_t
]

Where:

- (A) = natural dynamics
- (u_t) = interventions or stressors
- (B) = treatment/stressor effect
- (\epsilon_t) = noise

#### 3. Observations

Generate observable variables from latent state:

- sleep quality
- HRV
- fatigue
- pain
- mood
- appetite
- bowel quality
- hot/cold
- heaviness/lightness

[
x_t = C z_t + \eta_t
]

#### 4. Outcome

Add a predictable target:

- next-day flare
- response to intervention
- recovery score
- symptom worsening

#### 5. Semantic overlay

Map latent states to crude descriptors:

- flowing
- stuck
- depleted
- overheated
- damp/heavy

This gives you a toy “TCM-like” layer without pretending it is real TCM.

## Best contrarian synthetic tests

Build datasets where the doctrine-like labels are **partly wrong**.

For example:

- one semantic label contains two true hidden states
- two different labels map to the same hidden state
- intervention works by latent state, not by label
- regime shift starts before symptoms explode

That is perfect for your use case because it tests whether GRM can extract structure that a naive ontology misses.

## Minimal synthetic experiment plan

### Dataset A — easy

- 3 latent states
- linear dynamics
- low noise
- obvious intervention effect

Goal: verify basic recovery

### Dataset B — medium

- 5 latent dimensions
- overlapping symptoms
- partial label mismatch
- moderate noise
- delayed treatment effect

Goal: test robustness

### Dataset C — hard

- nonlinear transition regions
- hidden subtypes inside one label
- strong individual variability
- missing data
- weak signal

Goal: see where GRM breaks

## Success criteria

GRM should recover, at least approximately:

- the correct number of dominant modes
- clusters or trajectories close to true hidden states
- transitions before outcome events
- intervention-linked state shifts
- better prediction than naive baselines

## Recommended outputs

For each synthetic dataset, evaluate:

- latent state recovery
- next-step prediction
- flare prediction
- intervention response prediction
- semantic-label mismatch detection

## My recommendation

Current recommended synthetic scale:

- **200 subjects**
- **120 time steps each**
- **4 latent dimensions**
- **10–15 observed variables**
- **3 intervention types**
- **1 flare outcome**
- **1 crude TCM-like label layer**

This is the generator default because subset-heavy evaluation (`aliased_pair`,
hard-onset, constitution recovery) splits the data thinner than the original
80 × 60 prototype.

Initial implementation:

1. define the latent variables — done
2. define observation equations — done
3. define interventions and flare logic — done
4. generate CSVs — done
5. run GRM vs baselines — done

The current prototype includes:

- `grm_tcm_synthetic_generator.py`
- `grm_tcm_train.py`
- `grm_tcm_diagnostics.py`
- `grm_tcm_dynamic_grm.py`
- `grm_tcm_experiments.py`

## Strategic Path

1. Diagnostics — done
2. Difficulty modes — done
3. Ablations — done
4. Multi-seed experiments — implemented
5. Dynamic / inductive GRM — first pass implemented
6. Intervention-response analysis
7. Real-world pilot schema
8. Optional molecular/proteomics extension

`grm_tcm_experiments.py` now runs the full pipeline across random seeds, difficulty settings, and ablations.

Difficulty settings:

- easy
- medium
- hard
- chaotic

They adjust latent noise, observation noise, missingness, stress and treatment event rates, hidden subtype strength, delayed treatment effect, label noise, and practitioner bias.

Implemented ablations:

- feature similarity only
- temporal only
- feature + temporal
- feature + temporal + treatment
- random graph control
- permuted label control

## Current Diagnostic Answers (post-v2 generator)

Updated after: per-subject persistence baselines added, lag-augmented GRM head (`grm_plus_lag_*`), flare-onset secondary target with hard-subset deconfounding, v2 holism layer in the generator, and per-subject constitution-recovery evaluation. All numbers below are from the strict **inductive** path unless noted.

1. **Did GRM recover the hidden latent state?**

   Partially, on synthetic. Mean abs aligned latent correlation hovers around `0.45` in-sample and `~0.40` out-of-sample (Procrustes rotation, latent_recovery in inductive eval). Strong on `stress_activation` and `inflammatory_load`, weaker on `vitality_depletion` and `digestive_instability`. Real recovery on a known generator, not a clean "we got the latents" claim.

2. **Did GRM predict outcome better than naive baselines? (Honest deployable comparison)**

   Mixed, but clearly positive on the regression task.
   - **`next_day_score` R²**: GRM+lag `0.47`, persistence `0.35`, raw-RF `0.32`, naive `0.17`. **Δ over best baseline: +0.13 R²**.
   - **`flare_next_day` AUC**: GRM+lag `0.60`, naive `0.56`, raw-RF `0.56`, persistence `0.52`. Δ +0.04 AUC.
   - **`flare_onset` (hard subset, `flare_today=0` only)**: GRM `0.55`, GRM+lag `0.58`, lag-only `0.58`. Δ vs lag ≈ 0 — GRM and lag are substitutes here, both capture the same regime-trajectory signal.

   GRM standalone (no lag features) is a weak head; GRM+lag is the deployment-style competitor and beats the strongest baseline on regression.

3. **Did GRM discover label mismatch / hidden subtypes?**

   Yes, on synthetic. Diagnostics still produce contrarian-pattern and ontology-mismatch tables. `tcm_like_label` mixes `hidden_subtype` and aliased regimes mix `tcm_like_label`. Synthetic mismatch detection only — no claim about TCM validity.

4. **Did GRM recover the v2 constitution layer better than per-subject raw averaging?**

   No. Per-subject mean of raw features beats per-subject mean of GRM embeddings by **−0.25 R²** on average across constitution axes (raw `0.86` vs GRM `0.61` mean R²). GRM's spectral graph compresses by regime/dynamics similarity, which discards constitution-orthogonal information. Stable subject identity is *not* what graph-Laplacian eigenmodes recover from this graph construction.

## v2 holism layer (synthetic enrichment)

The generator was enriched to test whether a "holistic" cross-modal pattern is recoverable. Three layers added on top of the existing 7-regime switching state-space:

- **Constitution** — a 3-dim continuous stable per-subject vector (`thermal/energy/stability`) sampled at subject creation, biased by `hidden_subtype`. Projects onto every continuous observation via `D_MATRIX` with small per-channel entries but **coherent signs within each modality**, so the signal lives in cross-channel coherence.
- **Cross-modal coupling** — each modality (`vital_signs`, `sleep_energy`, `digestive`, `pain_mood`) is influenced by yesterday's other-modality means via a 4×4 coupling matrix. Empirically `corr(yest sleep_energy mean, today HRV) ≈ +0.46`.
- **Qualitative ordinal observations** — three TCM-inspired channels (`pulse_quality_like` 3 levels, `tongue_state_like` 4 levels, `complexion_like` 3 levels) sampled from `z + constitution` with ordinal thresholds. Each has a parallel `_label` string column.
- **Seasonal rhythm** — 45-day sinusoid on inflammatory load, per-subject phase, amplitude dampened by `constitution_stability`.

All four are scaled by difficulty preset (`easy/medium/hard/chaotic`).

## Honest findings (what v2 + the new baselines actually showed)

- **The persistence baseline was the previously-hidden winner.** Adding `score_persistence_today` and `flare_persistence_today` as input features to a comparison set raised the strongest non-GRM baseline by a lot. Once we did that, the old "GRM beats naive current score" claim shrank — but GRM+lag is now a fair deployment-style competitor that still beats persistence by +0.13 R² on regression.
- **Flare-onset inflation was real.** The headline 0.75 AUC on `flare_onset` collapsed to ~0.58 on the `flare_today=0` hard subset; that AUC drop was the `flare_today=1 -> onset=0` definitional certainty filter, not real onset prediction. GRM matches lag on the hard subset.
- **Constitution recovery is a clean negative for GRM.** Mean-of-raw beats mean-of-embedding by −0.25 R². GRM's graph captures *dynamics*, not *identity*. To make GRM useful for stable subject patterns, either the graph needs subject-similarity edges or the aggregator needs to escape regime-similarity compression — that's an architectural change, not a feature-engineering one.
- **Including qualitative channels as trainer inputs was a marginal gain.** `flare_next_day` AUC moved by +0.001 and `next_day_score` was unchanged. The qualitative signal duplicates what's already in continuous channels + lag for these targets.

## Synthetic-generator improvement directions (next iteration)

What we've measured tells us *where the current generator under-tests GRM* and where it over-rewards trivial baselines. The Pang et al. 2023 Nature paper (`article/s41586-023-06098-1.pdf`, "Geometric constraints on human brain function") is a useful pointer here: they show that **geometric eigenmodes** of the cortical surface beat **connectome graph eigenmodes** for explaining brain dynamics. GRM-TCM is in the connectome-camp by construction — visit-similarity + temporal + treatment edges. That motivates several generator changes:

1. **Make constitution affect *dynamics*, not just observations.** Currently `K` biases the obs projection. If `K` also modulated transition probabilities (e.g., high `thermal` raises P(enter inflammatory) and lowers self-transition in `digestive_instable`), the constitution signal would live in graph position, which GRM can in principle capture. This directly addresses the "raw mean beats GRM" finding by making graph-position-aware methods relevant.

2. **Continuous-manifold body-state generator (Pang-style).** Add a generator variant where body-state trajectories live on a low-dim continuous manifold (e.g., 2D-3D), and observations are superpositions of **long-wavelength eigenmodes** of a Laplace-Beltrami-like operator on that manifold. Then evaluate whether GRM's discrete graph-Laplacian eigenmodes approximate the continuous LBO eigenmodes. This is the *direct* analogue of the Pang geometry-vs-connectivity test on a body-state substrate.

3. **Aliased-future targets.** Design pairs of regimes that produce nearly identical day-`t` observations but diverge sharply at `t+1`. This breaks the lag baseline (today's obs aliases) and forces methods to rely on graph-position information that aliased obs can't provide. Concretely: a flag in regime metadata `aliased_pair_id` and a deterministic next-regime divergence rule. Evaluate `next_day_score` on the `aliased_pair_id != none` subset.

4. **Counterfactual intervention pairs.** Same subject, same starting state, two simulated futures with different treatment outcomes. Lets us evaluate **intervention-response prediction** as a target (mentioned in PHASE 1 but not yet measured). GRM's graph-position should help when raw obs at `t` don't predict the divergence.

5. **Multi-scale temporal coupling.** Slow constitutional drift (60-90 day timescale) + fast regime switching (1-7 day) + within-day variation. Current generator is single-scale (daily). This tests multi-timescale modeling, which is part of the next-step list and an obvious place where simple AR(1) baselines should fail and a spectral method should help.

6. **Cross-subject shared environmental drivers.** Currently subjects are independent draws. A shared environmental signal (seasonal weather, regional stress pulse) with subject-specific phase lags would force methods to disentangle individual vs population trajectories. The graph wouldn't see this directly unless we add cross-subject edges.

7. **Bigger and longer.** The generator default is now 200 subjects × 120 days because 80 × 60 was tight for subset evaluation. Use 300 × 180 for final stress runs if the aliased-pair, hard-onset, or constitution-recovery confidence intervals remain wide.

8. **Subject-trajectory diversity constraints.** Currently subjects can drift anywhere; in real biology, individual trajectories have stable basins. Adding a "constitutional basin" to each subject (z stays near `K`-determined attractor) would make subject identity more visible to graph methods.

Priority for the next pass: **(1) constitution-dynamics coupling** is the highest-leverage single change because it directly addresses the GRM-loses-on-constitution finding. **(3) aliased-future targets** is the most rigorous test of whether GRM beats lag. **(2) continuous manifold** is the most ambitious and the most aligned with the Pang reference.

## Dynamic GRM Port

`grm_tcm_dynamic_grm.py` implements the next piece from the markets GRM formulation:

- rolling-window `G^(t)`
- regime-change score `||G^(t) - G^(t-1)||_F`
- self-resonance / stuck-state score `G_ii`
- GRM-blended transition probabilities against Markov-only transitions
- data-derived `r_s`
- energy-based mode selection

Current single-run dynamic metrics are mixed:

- rolling regime-change flare AUC: about `0.48`
- self-resonance flare AUC: about `0.74`
- soft self-resonance flare AUC: about `0.83`
- GRM transition accuracy: about `0.79`
- transition accuracy lift over Markov-only: about `0.002`
- subject-level rolling regime-change flare AUC: about `0.66`
- subject-level soft self-resonance flare AUC: about `0.84`
- subject-level transition accuracy lift over Markov-only: about `0.055`
- subject-level soft self-resonance vs hidden subtype eta-squared: about `0.009`

Interpretation: the first dynamic port makes `G` meaningful as a propagator object, not only an embedding source. The useful signal is currently self-resonance / stuck-state detection. Rolling regime change and transition blending need more work before they can be considered strong flare predictors.

The subject-conditioned extension is the important correction. Pooled `R_t` measures population graph deformation and does not align with subject-level flares. Subject-level `R_{s,t}` is more meaningful for physiology and improves flare signal in the current default run. Soft self-resonance is better than hard state lookup because it removes the discrete stripe artifact and treats each visit as a weighted mixture over attractor states.

The subject-level mean soft self-resonance does **not** recover `hidden_subtype` yet; the current eta-squared is near zero. That is a useful negative result: self-resonance is currently a flare-risk / stuck-state marker, not a hidden-subtype marker.

Sharpening the state graph with KNN similarity can further improve dynamic scores; for example, `--similarity-mode knn --state-similarity-k 3 --max-modes 16` increased soft self-resonance AUC in the current smoke test. Early-fit state assignment with `--state-fit-end-day 40` is more honest as a predictive diagnostic, but weaker, so future reported predictive claims should distinguish transductive diagnostics from early-fit validation.

## Regime-Aware Diagnostics

After the synthetic-world redesign, diagnostics now use the new ground truth:

- `true_regime`
- `true_regime_id`
- `attractor_state`
- `true_attractor_states.csv`

Static diagnostics now report cluster alignment against true regimes. In the current run, GRM embedding clusters align substantially with `true_regime` (`NMI` about `0.56`, `ARI` about `0.49`) but much less with `attractor_state` (`NMI` about `0.19`, `ARI` about `0.16`). That means the static embedding can recover broad regime structure, but it does not cleanly isolate stuck-attractor states.

Dynamic diagnostics now add:

- inferred state vs true regime confusion
- soft self-resonance stuck-state AUC
- GRM/Markov transition accuracy after mapping inferred states to true regimes
- hidden subtype eta-squared for true stuck occupancy
- subject resonance vs true stuck occupancy
- state-source comparison across observation KMeans, dynamic-feature KMeans, and oracle true regimes

Current true-subtype signal is present in the generator ground truth: hidden subtype explains roughly `6%` of stuck-depleted occupancy variance and `11%` of stuck-agitated occupancy variance. The inferred dynamic resonance features do not fully recover that structure yet. This is the next modeling gap.

The state vocabulary is now an explicit experimental lever via `--state-source`. `kmeans_observation` tests whether visible observations are enough, `kmeans_dynamic` adds short trajectory features to reduce aliasing, and `true_regime` is an oracle ceiling for the redesigned synthetic world. In the current comparison, dynamic-feature states improve soft self-resonance flare signal relative to observation-only states, while true-regime states mainly improve next-state transition accuracy. That split is meaningful: part of the remaining gap is state aliasing, and part is the difference between recovering regime transitions and predicting flare timing.

Interpretation guardrail: these are synthetic benchmark results only. They test latent-state recovery, ontology mismatch detection, and ablation behavior; they do not prove TCM, Qi, or a biological mechanism.

About to do:
Multi-scale temporal hyper recursion. Cascading cross-time retrieval catching long -cycle historical log resonace.
Deconstruction techno feudalism and the genetic monopoly of the sole owner logevitiy race via Fractal Republicanism.
