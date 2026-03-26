# Vision: US Political Covariation Model

## Core Hypothesis

Political covariance follows shared community identity -- religion, class/occupation, neighborhood (who you talk to daily) -- more closely than it follows administrative geography or broad demographic bins. Communities that share these features covary politically even when separated by state borders.

This is a community-structure-first approach to political behavior. It is **not** just another election forecasting model. Election forecasting is one downstream application of a system whose primary purpose is to discover, characterize, and leverage the latent community structure that drives American political behavior.

---

## Three Nested Ambitions

The project pursues three goals in order of increasing difficulty. Each level depends on the one beneath it.

### 1. Descriptive

Detect and characterize politically meaningful community types that cross administrative borders. The question at this level is: *What are the natural groupings of American communities when defined by who people actually live among, worship with, and work alongside -- rather than by county lines or Census regions?*

### 2. Explanatory

Show that community-level covariation explains variance in political outcomes beyond what standard demographics plus geography already capture. The question here is: *Does community structure carry signal that demographics and location leave on the table?*

### 3. Predictive

Produce conditional forecasts from poll data propagated through community covariance structure. The question is: *When a poll moves in one place, what does the learned covariance structure imply for places that share its community profile but have not been polled?*

---

## Intellectual Foundations

### "Who you talk to daily" theory

Community is operationalized not through self-identification but through shared institutional space. The people you encounter at church, at work, and in your neighborhood constitute your political community. This draws on Lazarsfeld, Berelson, and Gaudet (1948), Huckfeldt and Sprague (1987), and the broader Columbia tradition of contextual influence.

### Dual-output model: vote share AND turnout

Most election models focus on vote share alone. This project models vote share and turnout jointly, enabling decomposition of election-to-election swings into three distinct mechanisms:

- **Persuasion effects**: voters changing their candidate preference.
- **Mobilization effects**: changes in who shows up to vote.
- **Composition effects**: changes in the electorate's makeup through migration, aging, and registration shifts.

This decomposition matters because the same net swing in a county can arise from very different community-level dynamics, and the correct interpretation determines what the model should propagate to similar communities.

### Voter stickiness assumption

Voters move less than polls suggest. Much of what looks like persuasion in topline polling is actually phantom swing -- differential nonresponse, likely-voter screen artifacts, and compositional churn in the polling sample. Community-level political behavior has high temporal autocorrelation. This follows Gelman et al. (2016) on phantom swings, Kalla and Broockman (2018) on the minimal persuasive effects of campaign contact, and Green and Palmquist (1990) on partisan stability.

### Communities as modeling output, not input

Community types are latent archetypes discovered from data. They are not pre-defined categories imposed by the analyst. The model should surface community types that are politically coherent -- meaning they covary together across elections -- without requiring the user to specify what those types are in advance.

### Topic model framing

Counties are treated as mixtures of community types, analogous to how documents are mixtures of topics in LDA or how pixels are mixtures of endmembers in spectral unmixing. A county that is 40% exurban evangelical, 35% college-town professional, and 25% rural agricultural behaves as a weighted combination of those archetypes. The mixing proportions are the primary object of estimation.

### Hierarchical structure

Fine-grained community types nest into blocs, which nest into mega-blocs. The number of types at each level is data-determined, not fixed a priori. This mirrors the hierarchical structure found in topic models, nested stochastic block models, and biological community ecology (HMSC). The hierarchy allows the model to operate at the resolution appropriate to the question: fine-grained for county-level analysis, coarse for national narrative.

---

## What the Model Should Reveal

The following examples illustrate the kind of distinctions the model must be capable of making. They are not exhaustive; they are litmus tests.

- **Cuban Americans in FL and Mexican Americans in TX should NOT be collapsed into "Hispanic."** These communities have different institutional contexts (religion, class structure, migration history, media ecosystems) and have diverged politically. The model should discover this separation without being told to look for it.

- **Amish and Plain communities across PA, OH, IN, and WV should covary despite crossing state lines.** Their shared religious-institutional identity produces similar political behavior (low baseline turnout, correlated surges when mobilized). State-level models miss this.

- **"Black voters" may fracture along lines the model reveals.** Urban institutional Black communities (anchored by historically Black churches, HBCUs, civic organizations) may behave differently from suburban Black communities in newer Sun Belt developments and from rural Black communities in the Mississippi Delta or coastal Carolinas. The model should surface these distinctions if they exist in the covariance structure.

- **Military base communities may behave similarly regardless of state.** Fayetteville (NC), Killeen (TX), Clarksville (TN), and Colorado Springs (CO) share demographic and institutional features that may produce political covariation invisible to state-level models.

---

## What Success Looks Like

1. **Community types correspond to recognizable communities.** When the model outputs a latent type, domain experts (political journalists, local organizers, regional scholars) should be able to look at its feature profile and say: "Yes, I know what that is."

2. **Community structure predicts political covariance beyond demographics.** In a held-out test, a model that includes community-type mixing proportions should outperform a model with the same demographic and geographic covariates but without community structure, as measured by the accuracy of predicted cross-county correlation in election swings.

3. **Polling propagated through community structure produces useful estimates.** When a poll is observed in counties dominated by a particular community type, the model should generate informative posterior updates for unpolled counties that share that type -- and these updates should outperform naive geographic interpolation.

4. **Shift narratives at the community level are informative for analysts.** The model should produce community-level swing decompositions (persuasion, mobilization, composition) that help explain *why* an election shifted the way it did, not just *that* it shifted.

---

## The Inference Engine: Types as the Unit of Political Inference

The fundamental object of inference in this model is **θ — the vector of type means**. θ[k] is the answer to: *how is type k voting right now, in this cycle?* State outcomes, county predictions, and district-level forecasts are all downstream products of θ. They are not the thing we are estimating — they are what falls out once we know θ.

This framing has a concrete consequence: **polls are observations of W·θ, not of state-level outcomes.** A poll in Georgia is a noisy measurement of how Georgia's mix of types is voting. The W vector encodes that mix. What the model learns from that poll is information about θ — which then propagates everywhere those types exist, regardless of state lines.

This means a poll in Florida, Massachusetts, and Texas that consistently shows the same type responding the same way to a candidate is *not three separate observations*. It is three measurements of the same underlying quantity — θ for that type. The model triangulates θ directly from all three. Utah then requires no polling at all: its prediction follows from θ × Utah's type scores.

The practical payoff in swing states is large. Florida, Georgia, Michigan, Wisconsin, and Ohio are built from heterogeneous mixtures of types. State-level polls are inherently ambiguous — a D+3 topline in Wisconsin could arise from many different type-level configurations. Multiple polls from diverse geographies that share types with Wisconsin allow the model to decompose the ambiguity and infer what each type within Wisconsin is actually doing.

---

## Candidate Effects as Deviations from Type-Covariance Expectations

The type covariance Σ encodes how types have historically moved together. Combined with a national environment signal (fundamentals), Σ generates an *expected* θ for any given cycle: given this national environment, this is how each type would be expected to move if no candidate-specific forces were operating.

When observed polls update θ and produce a *posterior* θ that deviates from the expected, that deviation is interpretable as a **candidate effect**:

```
candidate_effect[k] = θ_posterior[k] - θ_expected[k]
```

A positive candidate effect means type k is performing more Democratic than the national environment and historical covariance predict. A negative candidate effect means it is performing more Republican.

**Canonical examples:**

- **Trump and Rust Belt working-class types (2016):** The national environment in 2016 did not predict non-college white working-class types moving as hard R as they did. The posterior θ for those types significantly exceeded what Σ and fundamentals implied. That residual is Trump's candidate draw with those communities — a durable realignment signal that the model should have detected from early polling and propagated to every county with high working-class type scores.

- **W and Hispanic types (2004):** Historical covariance predicted Hispanic-adjacent types maintaining strong Democratic lean. The posterior after 2004 results showed those types performing more R than expected. The deviation captures W's candidate effect with socially conservative Hispanic Catholic communities — a signal the model should apply to all counties with similar type composition, not just the polled geographies.

This decomposition serves two purposes:

1. **Interpretive**: It provides a principled framework for explaining *why* a cycle diverged from structural expectations — separating national environment effects from candidate-specific draws.

2. **Predictive**: Candidate effects inferred from early-cycle polling in polled states can be applied prospectively to unpolled states through the type structure. If a candidate is consistently over- or underperforming with a type across multiple states, those effects propagate to every county in America with membership in that type.

This architecture bridges the Forecast function and the Sabermetrics silo. Sabermetrics quantifies candidate effects at the race level (CTOV, candidate drag/lift). The Forecast engine uses those same effects at the type level to improve predictions in unpolled geographies.

### Candidate Effects vs. Structural Shifts

A critical distinction must be maintained between two superficially similar phenomena:

**Candidate effect:** A cycle-specific deviation. θ_posterior[k] exceeds θ_expected[k] because of something idiosyncratic to this candidate — their persona, biography, messaging, or platform resonance with type k. Remove the candidate, and a generic party nominee reverts closer to the baseline. The deviation does not persist across candidates.

**Structural shift:** A durable realignment. θ_posterior[k] has moved and *stays moved* across successive candidates from the same party. The type's baseline has genuinely relocated. The historical prior itself needs updating.

The empirical test is persistence: does the deviation survive candidate turnover? Trump's performance with non-college white working-class types initially appeared as a candidate effect. Its persistence across the 2018 and 2022 midterms — with candidates other than Trump — suggests a structural shift is underway, or that a candidate effect was large enough to trigger genuine realignment. By contrast, W's performance with socially conservative Hispanic Catholic types in 2004 did not persist strongly through subsequent Republican nominees — more consistent with a candidate-specific draw.

**Architectural implication:** The model needs two layers:
- **Cycle-specific θ:** Updated each cycle by polls. Captures candidate effects and current environment.
- **Structural θ baseline:** The prior on θ, updated slowly across cycles as evidence of durable shift accumulates. When consecutive cycles show the same directional deviation for a type, that deviation migrates from "candidate effect" into a revised structural baseline.

This is how the model learns realignments over time rather than perpetually treating them as surprises. A model that does not distinguish these two phenomena will either over-correct (chasing each candidate's idiosyncrasies into the baseline) or under-correct (treating durable realignments as temporary noise).

A further complication: a candidate can *widen the tent* — drawing a type toward their party — and then the party can pivot institutionally to follow that candidate, baking what started as a candidate effect into the party's platform, messaging, and coalition. At that point the candidate effect has become a structural shift not because the type changed but because the party changed to meet it. Trump and the Republican Party's relationship with non-college white working-class voters is the clearest modern example. The distinction between "candidate pulls type" and "party follows candidate to lock in type" is genuine but difficult to operationalize cleanly from electoral data alone. This is one reason the candidate effects layer is deferred — the boundary between these phenomena requires careful thinking before it can be modeled.

---

## The 2026–2028 Development Arc

**2026 (proof of concept):** The model ingests midterm state-level polling and a national fundamentals signal. It infers how types are responding to the 2026 national environment and candidate field, produces county-level predictions for all competitive Senate and governor races, and is validated against actual results. The key question: *can the type structure reliably translate state-level polling into defensible county-level predictions?*

**2028 (the engine):** A mature presidential cycle with high polling volume. National polls and state polls from all competitive states produce a dense, overdetermined system for inferring θ. Each poll — regardless of race or geography — is an observation on W·θ. With rich crosstab ingestion (type-specific W vectors), each quality poll with full demographic crosstabs becomes a near-direct measurement of specific type means. The engine pins θ with high confidence for most types well before election day, generating reliable predictions for unpolled states as a byproduct. Candidate effects are detected early from consistent type-level deviations across diverse polling geographies and propagated nationwide.

---

## What Falsification Looks Like

The model's core claims are empirical. The following findings would constitute falsification:

1. **Demographics plus geography capture all useful signal.** If a model with standard ACS variables and geographic fixed effects (or spatial random effects) explains county-level covariance as well as a model augmented with community-type structure, the community-structure hypothesis adds nothing.

2. **Cross-border community structure adds negligible predictive value.** If communities defined by shared religion, occupation, and neighborhood context do not covary more strongly across state lines than would be expected from demographics alone, the "who you talk to" mechanism is not operative at the scale this model targets.

3. **Community types are too unstable to model.** If the latent types shift radically from election to election (beyond what can be explained by real realignment), the framework lacks the temporal coherence needed for forecasting.

4. **Covariance estimated from past elections fails to predict future differential swings.** If the community-covariance matrix learned from 2012-2020 does not improve prediction of differential county swings in 2024 relative to a covariance-free baseline, the historical covariance structure is not informative for the future.

---

## Tracking Assumptions

This project rests on a set of modeling assumptions that must be explicitly stated, monitored, and revisited as evidence accumulates. These are tracked in [`ASSUMPTIONS_LOG.md`](ASSUMPTIONS_LOG.md).

---

## References

All research references -- papers, datasets, repositories, and methods -- are catalogued in [`SOURCE_LOG.md`](SOURCE_LOG.md).
