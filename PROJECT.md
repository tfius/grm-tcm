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

Start with:

- **100 subjects**
- **90 time steps each**
- **4 latent dimensions**
- **10–15 observed variables**
- **3 intervention types**
- **1 flare outcome**
- **1 crude TCM-like label layer**

That is enough to tell whether this is worth pursuing.

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

## Current Diagnostic Answers

Current single-run diagnostics answer the four core questions this way:

1. **Did GRM recover the hidden latent state?**

   Partially. The trainer reports mean absolute aligned latent correlation around `0.406`, and diagnostics find a best absolute GRM-mode/latent correlation around `0.682`. That is real synthetic latent recovery, but not strong enough to claim the current spectral embedding fully recovers the hidden state.

2. **Did GRM predict outcome better than naive baselines?**

   No for next-day score regression. Current R2 values are roughly:

   - GRM ridge: `0.170`
   - raw random forest: `0.974`
   - naive current score: `0.973`

   GRM flare prediction is high in this synthetic run, with ROC-AUC around `1.000`, but the raw and naive baselines are also near-ceiling. This means the outcome is currently too easy for short-horizon baselines.

3. **Did GRM discover label mismatch / hidden subtypes?**

   Yes, as a diagnostics target. The current diagnostics produce `8` contrarian findings and ontology-mismatch tables showing TCM-like labels mixing hidden subtypes and hidden subtypes splitting across labels. This is synthetic mismatch detection, not validation of TCM categories.

4. **Did GRM clusters align more with true latent structure than with naive TCM-like labels?**

   No in the current single run. Cluster alignment with `hidden_subtype` is near zero (`ARI` about `0.00`, `NMI` about `0.006` at best), while alignment with `tcm_like_label` is higher (`NMI` about `0.281` at best). That suggests the current embedding clusters track observed semantic/label structure more than the generator's hidden subtype IDs.

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

Interpretation guardrail: these are synthetic benchmark results only. They test latent-state recovery, ontology mismatch detection, and ablation behavior; they do not prove TCM, Qi, or a biological mechanism.
