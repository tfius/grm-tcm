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

Next step should be:

1. define the latent variables
2. define observation equations
3. define interventions and flare logic
4. generate CSVs
5. run GRM vs baselines

I can generate the full **synthetic data spec + Python generator** next.
