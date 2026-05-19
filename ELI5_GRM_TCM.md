# GRM-TCM, Explained Simply

## The Big Idea

The project asks:

Can we use repeated body measurements to find hidden body states, predict when a
person is moving toward a bad state, and then see whether simple TCM-like words
such as "stuck", "depleted", "hot", or "cold" line up with those states?

The important part:

GRM does not prove TCM. GRM gives us a way to test whether a "resonance-like"
model is useful for prediction.

## Think of Each Day as a Dot

Imagine every day for every person is one dot.

Each dot has measurements:

```text
sleep
heart rate
HRV
temperature
fatigue
pain
appetite
bowel quality
mood
energy
heaviness
cold / hot feeling
```

Two dots are connected if they are similar.

Dots are also connected if they belong to the same person on nearby days.

So the model builds a big map of all days.

```text
day dots + similarity links + time links = body-state graph
```

## What GRM Does

GRM looks at this graph and asks:

```text
What are the main patterns in how these body-state dots are connected?
```

These patterns are called modes.

You can think of modes like slow waves across the graph:

```text
mode 1: maybe separates stable days from dysregulated days
mode 2: maybe separates tired/heavy days from agitated/hot days
mode 3: maybe separates recovery-like days from flare-like days
```

The model does not know these meanings in advance. We inspect them afterward.

## What "Resonance" Means Here

In this project, resonance means:

```text
If a body state is perturbed, how strongly does that effect stay in the same
state or spread to related states?
```

It is a mathematical score, not a mystical claim.

High self-resonance means:

```text
This state tends to reinforce itself.
The person may be stuck in this pattern.
It may be harder to leave this state.
```

Low self-resonance means:

```text
This state is less sticky.
The person may move out of it more easily.
```

So "stuckness" becomes something measurable:

```text
stuckness = high self-resonance + persistent state + worse future outcomes
```

## Why This Might Matter

Suppose someone has several days like this:

```text
poor sleep
low HRV
high fatigue
heaviness
bad appetite
low mood
```

A normal model might say:

```text
Today's symptoms are bad, so tomorrow may be bad.
```

GRM tries to say something richer:

```text
This day belongs to a sticky hidden state.
People in this state often remain there or move toward flare.
This state may respond differently to interventions.
```

That is the useful version of the resonance idea.

## Where TCM Comes In

TCM-like words should come after the model learns states.

Bad approach:

```text
Start by assuming this is Qi stagnation or spleen deficiency.
```

Better approach:

```text
First learn the hidden states from data.
Then ask whether simple descriptors line up with those states.
```

Examples:

```text
Does high self-resonance line up with "stuck"?
Does a low-energy state line up with "depleted"?
Does high temperature + agitation line up with "hot"?
Does heaviness + poor digestion line up with "damp/heavy"?
```

The mapping is only useful if it predicts something.

## What Would Count as a Good Result?

A good result would look like:

```text
GRM finds stable hidden states.
Some states predict tomorrow's fatigue, pain, sleep, or flare risk.
High self-resonance predicts staying stuck or getting worse.
Some interventions help people leave specific bad states.
Simple TCM-like descriptors line up with those states better than chance.
```

For example:

```text
State 4 has high self-resonance.
People in state 4 often remain there for several days.
State 4 predicts higher flare risk two days later.
People describe state 4 as heavy, depleted, and stuck.
Intervention A increases movement from state 4 to state 1.
```

That would be interesting.

It would still not prove ancient TCM theory. But it would show that some
whole-body descriptors may map to measurable dynamic states.

## What Would Count as a Bad Result?

A bad result would be:

```text
GRM does not beat simple baselines.
The hidden states change every time we retrain.
Self-resonance is just another name for symptom severity.
TCM-like labels do not line up with states.
Intervention effects disappear after controlling for sleep, stress, or caffeine.
```

Then the model is probably not useful for this purpose.

## The Simple Pipeline

Step 1:

```text
Collect repeated measurements.
```

Step 2:

```text
Build a graph where similar days are connected.
```

Step 3:

```text
Use GRM to find hidden modes and resonance scores.
```

Step 4:

```text
Use those scores to predict future outcomes.
```

Step 5:

```text
Compare against simple models.
```

Step 6:

```text
Only if prediction works, map simple TCM-like descriptors onto the states.
```

## The Most Important Test

The key question is not:

```text
Does this sound like TCM?
```

The key question is:

```text
Does this model predict future state better than simple baselines?
```

Baselines include:

```text
yesterday predicts today
moving average
raw symptom model
simple clustering
logistic regression
random forest / XGBoost
```

If GRM cannot beat these, the resonance idea is probably not adding enough.

## One-Sentence Version

GRM turns repeated body measurements into a graph of body states, measures which
states are sticky or resonant, tests whether those states predict future health
changes, and only then checks whether simple TCM-like words map onto the learned
states in a stable and useful way.

