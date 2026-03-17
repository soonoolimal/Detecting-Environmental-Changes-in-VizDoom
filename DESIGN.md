# Design Notes

## DT Training Objective

### Research Purpose

The DT is not trained as a standalone decision-making agent.
Its primary role is to serve as a **feature extractor for the Task Detector (TD)**.
For TD to accurately detect task shifts, the DT's encoder must learn
**task-discriminative representations** — features that vary systematically
across vanilla, observation-shifted, and reward-shifted task variants.

---

### Loss Design

DT is trained with two losses: `ac_loss` and `rtg_loss`.

**ac_loss (Cross-Entropy)**

```
predict a_t from h(o_t)
```

The encoder is forced to capture visual features that are predictive of
which action was taken in a given observation. When an observation shift
occurs (e.g., wall color change), the same action patterns arise in a
visually different context, giving the encoder signal to distinguish
task variants at the observation level.

**rtg_loss (MSE)**

```
predict R_{t+1} from h(a_t)
```

The model is forced to capture how actions relate to future returns.
When a reward shift occurs (reward structure changes), the mapping from
actions to returns changes, propagating task-relevant signal through the
action token representations and back into the encoder.

Both losses flow gradients through the encoder, driving it toward
task-relevant feature learning.

---

### Why ob_loss Was Removed

A third auxiliary loss was considered:

```
predict enc(o_{t+1}) from h(a_t)   # ob_loss
```

This was removed for two reasons.

**1. Moving target problem (technical)**

The target `enc(o_{t+1})` is produced by the same encoder being trained.
As encoder parameters change, the target changes simultaneously.
pred and target move together, making the loss unstable and its
convergence direction undefined. This is a known failure mode
(cf. representation collapse in self-supervised learning).

**2. Objective misalignment (design)**

Even if the moving target problem were resolved, predicting the next
observation's latent representation is a **world modeling** objective,
not a **task discrimination** objective. Low-level temporal continuity
in pixel space does not necessarily correspond to the features that
distinguish task variants. Including ob_loss would mix an unrelated
training signal into the encoder, potentially working against the
primary research purpose.

---

### TD Input Design

Given the above, TD receives:

- `ob_enc = self.dt.encoder(observations)`: task-discriminative visual
  features learned via ac_loss and rtg_loss.
- `rtg_preds`: predicted returns from the frozen DT, carrying reward
  structure information learned via rtg_loss.

Both inputs are derived from a frozen DT, ensuring that TD training
does not alter the representations learned during DT training.