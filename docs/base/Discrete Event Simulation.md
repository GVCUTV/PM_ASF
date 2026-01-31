<!--
Discrete-Event Simulation: A First Course
Lawrence Leemis, Steve Park
Chapter 1 — Models
-->

# Chapter 1 — Models

The modeling approach in this book is based on the use of a **general-purpose programming language** for model implementation at the computational level.  
The alternative approach is to use a **special-purpose simulation language**.

## Sections
- 1.1 Introduction  
- 1.2 A Single-Server Queue (program `ssq1`)  
- 1.3 A Simple Inventory System (program `sis1`)  

This chapter introduces discrete-event simulation with an emphasis on **model building**.  
Section 1.1 presents the overall modeling philosophy.  
Sections 1.2 and 1.3 present two fundamental models in detail.

---

## 1.1 Introduction

This book provides an introduction to computational and mathematical techniques for **modeling, simulating, and analyzing discrete-event stochastic systems**.

In discrete-event simulation, one does not experiment with the real system. Instead, one develops and experiments with a **model** of the system.  
The emphasis of this chapter is therefore on **model construction**.

---

### 1.1.1 Model Characterization

A system model may be classified along three axes:

- **Deterministic vs stochastic**
- **Static vs dynamic**
- **Continuous vs discrete**

A **deterministic** model has no random components.  
A **stochastic** model includes randomness.

A **static** model ignores time evolution.  
A **dynamic** model depends on time.

A **continuous** dynamic model evolves continuously in time.  
A **discrete** dynamic model changes state only at discrete instants.

#### Definition 1.1.1 — Discrete-Event Simulation Model

A discrete-event simulation model is:

- **stochastic** — at least some state variables are random  
- **dynamic** — the evolution of state over time matters  
- **discrete-event** — state changes occur only at discrete time instants  

A related but secondary class is **Monte Carlo simulation**, which is stochastic and static.

---

### 1.1.2 Model Development

Model development is inherently iterative, but can be summarized as follows.

#### Algorithm 1.1.1 — Discrete-Event Model Development

1. **Determine objectives**  
   Define the questions the model must answer.

2. **Build a conceptual model**  
   Identify state variables and system structure.

3. **Develop a specification model**  
   Define input models, usually based on data or assumptions.

4. **Construct the computational model**  
   Implement the model as a program.

5. **Verify**  
   Ensure the program correctly implements the specification.

6. **Validate**  
   Ensure the model represents the real system adequately.

---

#### Example 1.1.1 — Machine Shop

- 150 identical machines  
- Machines operate until failure  
- Failures are repaired by technicians  
- Technicians work limited hours  
- Each machine generates revenue when operational  

**Objective:** determine the number of technicians that maximizes profit.

Conceptual state variables:
- Machine status (operational / failed)
- Technician status (busy / idle)

---

### System Diagrams

System diagrams help describe conceptual models.

In the machine shop:
- Machines circulate between operational and failed states
- Failed machines queue for repair
- Technicians serve one machine at a time

---

### General Modeling Principles

- Make models **as simple as possible, but not simpler**
- Avoid skipping conceptual and specification modeling
- Verification and validation are distinct activities

---

### 1.1.3 Simulation Studies

Once a model is built, it is used as follows.

#### Algorithm 1.1.2 — Simulation Study

7. Design experiments  
8. Perform production runs  
9. Analyze output statistically  
10. Make decisions  
11. Document results  

---

### Insight

The principal benefit of simulation is **insight** into system behavior.  
Understanding often emerges during model construction, not only from results.

---

### 1.1.4 Programming Languages

There is debate between:

- **General-purpose languages** (C, C++, Java, etc.)
- **Simulation languages** (GPSS, SIMAN, SLAM, SIMSCRIPT)

This book advocates general-purpose languages to promote understanding.

---

### 1.1.5 Organization and Terminology

A **model** exists at three levels:
- Conceptual
- Specification
- Computational

A **simulation** usually refers to the computational model and its execution.

---

### 1.1.6 Exercises

- Identify examples for each system-model class  
- Discuss verification vs validation  
- Define the concept of system state  
- Survey simulation languages  

---

## 1.2 A Single-Server Queue

A **single-server service node** consists of:
- a server
- a queue

Jobs arrive randomly, receive service, and depart.

---

### 1.2.1 Conceptual Model

#### Definition 1.2.1 — Single-Server Service Node

- Jobs arrive randomly
- One server provides service
- Jobs queue if the server is busy
- Service is non-preemptive and conservative

#### Queue Disciplines
- FIFO (default)
- LIFO
- SIRO
- Priority (e.g. SJF)

Capacity may be finite or infinite.

---

### 1.2.2 Specification Model

For job *i*:

- Arrival time: `a_i`
- Delay in queue: `d_i`
- Service start: `b_i = a_i + d_i`
- Service time: `s_i`
- Wait time: `w_i = d_i + s_i`
- Completion time: `c_i = a_i + w_i`

Interarrival times:
```

r_i = a_i − a_{i−1}

```

---

### FIFO Delay Equation

For FIFO queues:
```

d_i = max(0, d_{i−1} + s_{i−1} − r_i)

````

---

### Algorithm 1.2.1 — FIFO Delay Computation

```c
c0 = 0;
i = 0;
while (more jobs) {
  i++;
  ai = GetArrival();
  if (ai < c[i-1])
    di = c[i-1] - ai;
  else
    di = 0;
  si = GetService();
  ci = ai + di + si;
}
````

---

## 1.3 A Simple Inventory System

An inventory system tracks:

* Inventory level
* Demand arrivals
* Reorder policies
* Lead times

Discrete-event simulation models inventory changes due to:

* demand events
* replenishment events

Inventory models provide the foundation for later chapters.

---

**End of Chapter 1**


<!--
Discrete-Event Simulation: A First Course
Lawrence Leemis, Steve Park
Chapter 2 — Random Number Generation
-->

# Chapter 2 — Random Number Generation

Random number generation is a fundamental component of **stochastic simulation**.  
In discrete-event simulation, randomness is required to model arrivals, services, failures, and other stochastic phenomena.

This chapter focuses on **pseudo-random number generators (PRNGs)** and their use in Monte Carlo and discrete-event simulation.

---

## Sections
- 2.1 Lehmer Random Number Generation: Introduction  
- 2.2 Lehmer Random Number Generation: Implementation (`rng`)  
- 2.3 Monte Carlo Simulation (`galileo`, `buffon`)  
- 2.4 Monte Carlo Simulation Examples *(optional)*  
- 2.5 Finite-State Sequences *(optional)*  

---

## 2.1 Lehmer Random Number Generation: Introduction

Simulation requires sequences of numbers that **behave like independent samples from U(0,1)**.

True randomness is rarely available; instead, simulation relies on **deterministic algorithms** that generate sequences with good statistical properties.

Such generators are called **pseudo-random number generators**.

---

### Linear Congruential Generators (LCG)

A commonly used class of PRNGs is defined by the recurrence:

```

Z_i = (a Z_{i−1} + c) mod m
U_i = Z_i / m

```

where:
- `m` is the modulus
- `a` is the multiplier
- `c` is the increment
- `Z_0` is the seed

If `c = 0`, the generator is called a **Lehmer generator**.

---

### Lehmer Generator

```

Z_i = (a Z_{i−1}) mod m
U_i = Z_i / m

````

Properties:
- Simple and fast
- Fully deterministic
- Period at most `m − 1`

Good parameter choices are critical.

---

### Period

The **period** is the length of the sequence before it repeats.

For a Lehmer generator, the maximum period is `m − 1`.

Achieving full period requires:
- `m` prime
- `a` a primitive root modulo `m`

---

## 2.2 Lehmer Random Number Generation: Implementation

This section introduces a reusable **random number library**.

### Library `rng`

The library encapsulates:
- generator state
- seed initialization
- uniform random number generation

#### Typical Interface

```c
void rng_init(long seed);
double rng_rand();
````

The generator returns values in `(0,1)`.

---

### Seeding

The **seed** determines the entire sequence.

Important properties:

* Same seed → same sequence (reproducibility)
* Different seeds → different sequences

Reproducibility is a **key advantage of simulation**.

---

### Numerical Issues

* Integer overflow must be avoided
* Modulus and multiplier must be chosen carefully
* Floating-point conversion should preserve uniformity

---

## 2.3 Monte Carlo Simulation

Monte Carlo simulation uses random sampling to estimate quantities of interest.

It applies primarily to **static stochastic systems**.

---

### Example: Estimating π (`buffon`)

Random points are generated uniformly in a square.

π is estimated by:

```
π ≈ 4 × (points inside circle) / (total points)
```

Accuracy improves as the number of samples increases.

---

### Example: Galileo’s Experiment (`galileo`)

Simulation reproduces probabilistic experiments described historically.

Monte Carlo simulation is often used to:

* validate intuition
* test analytical results
* explore probabilistic behavior

---

## 2.4 Monte Carlo Simulation Examples *(Optional)*

Examples include:

* dice games
* hat-checking problem
* gambling simulations

These examples emphasize:

* law of large numbers
* sampling variability
* convergence behavior

---

## 2.5 Finite-State Sequences *(Optional)*

PRNGs produce **finite-state deterministic sequences**.

Consequences:

* eventual repetition
* dependence structure
* non-random artifacts if poorly designed

This motivates:

* statistical testing
* use of multiple streams
* careful generator selection

---

## Summary

* Simulation requires high-quality uniform random numbers
* Lehmer generators provide a simple, efficient solution
* Reproducibility is essential
* Monte Carlo simulation relies entirely on PRNG quality
* Poor generators can invalidate simulation results

---

**End of Chapter 2**




<!--
Discrete-Event Simulation: A First Course
Lawrence Leemis, Steve Park
Chapter 3 — Discrete-Event Simulation
-->

# Chapter 3 — Discrete-Event Simulation

This chapter introduces **stochastic discrete-event simulation models**, extending the
trace-driven models of Chapter 1 by incorporating **random number generation**.

The focus is on:
- event-based time advance
- stochastic input modeling
- reusable simulation structure

---

## Sections
- 3.1 Discrete-Event Simulation (`ssq2`, `sis2`)
- 3.2 Multi-Stream Lehmer RNG (`rngs`)
- 3.3 Discrete-Event Simulation Models (`ssms`)

---

## 3.1 Discrete-Event Simulation

A **discrete-event simulation** advances time by jumping directly to the next scheduled
event.

System state changes occur only at **event times**.

Typical events:
- job arrival
- service completion
- inventory replenishment

---

### Simulation Clock

A simulation maintains a **simulation clock** `t`, representing the current simulated time.

Rules:
- `t` never decreases
- `t` advances to the time of the next event

---

### Event Scheduling

Each event is associated with:
- an **event type**
- an **event time**

Future events are stored in an **event list**, ordered by event time.

---

### Single-Server Queue (`ssq2`)

This model extends the FIFO single-server queue by:
- generating interarrival times randomly
- generating service times randomly

Interarrival times and service times are sampled from specified distributions.

---

#### Arrival Event Logic

1. Advance clock to arrival time
2. Schedule next arrival
3. If server idle:
   - start service
   - schedule departure
4. Else:
   - enqueue job

---

#### Departure Event Logic

1. Advance clock to departure time
2. If queue empty:
   - server becomes idle
3. Else:
   - dequeue next job
   - schedule its departure

---

### Inventory System (`sis2`)

The inventory level evolves due to:
- demand events
- replenishment events

State variables:
- inventory level
- outstanding orders

Policies determine when orders are placed and how much is ordered.

---

## 3.2 Multi-Stream Lehmer Random Number Generation

Using a **single random number stream** can introduce unwanted correlations.

To avoid this, simulations use **multiple independent streams**.

---

### Stream-Based Generators

Each stream:
- has its own seed
- evolves independently

Typical usage:
- one stream for arrivals
- one stream for services
- one stream for routing or decisions

---

### Library `rngs`

Provides:
- multiple independent Lehmer generators
- stream selection
- reproducibility

Benefits:
- modular simulation design
- easier debugging
- reduced correlation risk

---

## 3.3 Discrete-Event Simulation Models

This section generalizes discrete-event simulation into a **modeling framework**.

---

### Model Components

A discrete-event simulation model consists of:

- **State variables**
- **Event types**
- **Event list**
- **Statistical counters**
- **Initialization logic**
- **Termination conditions**

---

### Initialization

Typical steps:
- set simulation clock to zero
- initialize state variables
- schedule initial events

---

### Main Simulation Loop

```

while (termination condition not met) {
remove next event from event list
advance simulation clock
execute event logic
update statistics
}

```

---

### Termination Conditions

Common termination rules:
- fixed number of events
- fixed simulated time horizon
- convergence-based stopping

---

### Statistics Collection

Statistics may be:
- **job-based** (e.g., delay per job)
- **time-based** (e.g., average queue length)

Statistics must be updated:
- at event times
- when state variables change

---

## Verification vs Validation

- **Verification**: is the model implemented correctly?
- **Validation**: does the model represent the real system?

Both are essential.

---

## Reproducibility

A defining feature of discrete-event simulation:
- identical seeds → identical results
- experiments can be repeated exactly

---

## Summary

- Discrete-event simulation advances time by events
- Event lists drive execution
- Randomness is introduced via RNG streams
- Models are modular and extensible
- Correct statistics collection is critical

---

**End of Chapter 3**




<!--
Discrete-Event Simulation: A First Course
Lawrence Leemis, Steve Park
Chapter 4 — Statistics
-->

# Chapter 4 — Statistics

Statistical analysis is essential in discrete-event simulation because simulation output
is **random**. This chapter develops basic statistical tools required to summarize and
interpret simulation results.

---

## Sections
- 4.1 Sample Statistics (`uvs`)
- 4.2 Discrete-Data Histograms (`ddh`)
- 4.3 Continuous-Data Histograms (`cdh`)
- 4.4 Correlation (`bvs`, `acs`) *(optional)*

---

## 4.1 Sample Statistics

Simulation output consists of **observations**:
```

x1, x2, ..., xn

```

These observations are treated as realizations of a random variable.

---

### Sample Mean

The **sample mean** estimates the expected value:

```

x̄ = (1/n) Σ xi

```

Properties:
- unbiased estimator of E[X]
- variance decreases as n increases

---

### Sample Variance

The **sample variance** estimates variability:

```

s² = (1/(n−1)) Σ (xi − x̄)²

```

The denominator `(n−1)` makes the estimator unbiased.

---

### Sample Standard Deviation

```

s = √s²

```

Provides a measure of dispersion in the same units as the data.

---

### Interpretation

- Mean → central tendency
- Variance / standard deviation → variability
- Both are required to understand performance

---

## 4.2 Discrete-Data Histograms

A **histogram** approximates the probability mass function of a discrete random variable.

---

### Construction

1. Partition values into discrete categories
2. Count occurrences in each category
3. Normalize by total number of observations

```

p̂(x) = count(x) / n

```

---

### Properties

- Visualizes distribution shape
- Approximates probabilities
- Useful for detecting anomalies

---

### Program `ddh`

- Reads discrete data
- Produces frequency tables
- Outputs normalized histograms

---

## 4.3 Continuous-Data Histograms

Continuous data must be **binned**.

---

### Binning

- Divide range into intervals (bins)
- Count observations per bin
- Normalize by bin width and sample size

```

f̂(x) = count(bin) / (n × bin width)

```

---

### Trade-offs

- Too few bins → loss of detail
- Too many bins → excessive noise

Bin width selection is critical.

---

### Program `cdh`

- Builds histograms for continuous data
- Supports adjustable bin widths
- Outputs density estimates

---

## 4.4 Correlation *(Optional)*

Simulation output observations are often **correlated**.

---

### Sample Covariance

For paired observations `(xi, yi)`:

```

cov(x, y) = (1/(n−1)) Σ (xi − x̄)(yi − ȳ)

```

---

### Correlation Coefficient

```

ρ̂ = cov(x, y) / (sx sy)

```

Range:
```

−1 ≤ ρ̂ ≤ 1

```

---

### Implications

- Correlation violates independence assumptions
- Classical statistical formulas may fail
- Output analysis must account for dependence

---

## Summary

- Simulation output is random and must be analyzed statistically
- Sample mean and variance are fundamental
- Histograms reveal distribution shape
- Correlation is common in time-series output
- Ignoring variability leads to invalid conclusions

---

**End of Chapter 4**




<!--
Discrete-Event Simulation: A First Course
Lawrence Leemis, Steve Park
Chapter 5 — Next-Event Simulation
-->

# Chapter 5 — Next-Event Simulation

This chapter presents the **next-event simulation paradigm**, the most common and
powerful approach for implementing discrete-event simulation models.

Time advances directly to the next scheduled event, avoiding unnecessary computation
between events.

---

## Sections
- 5.1 Next-Event Simulation (`ssq3`)
- 5.2 Next-Event Simulation Examples (`sis3`, `msq`)
- 5.3 Event List Management (`ttr`) *(optional)*

---

## 5.1 Next-Event Simulation

In next-event simulation, the simulation clock is always equal to the **time of the most
recent event**.

Future events are stored explicitly and processed in chronological order.

---

### Event List

An **event list** contains:
- event time
- event type
- event-specific data

The next event is the one with **minimum event time**.

---

### Simulation Clock Advancement

```

t ← time of next event

```

Time never advances continuously — only at event times.

---

### Event Types in a Single-Server Queue

Typical events:
- **arrival**
- **departure**

Each event has a corresponding event-handling routine.

---

### Single-Server Queue (`ssq3`)

State variables:
- server status (idle / busy)
- number of jobs in queue

---

#### Arrival Event

1. Advance clock to arrival time
2. Schedule next arrival
3. If server idle:
   - start service
   - schedule departure
4. Else:
   - enqueue job

---

#### Departure Event

1. Advance clock to departure time
2. If queue empty:
   - server becomes idle
3. Else:
   - remove job from queue
   - schedule its departure

---

### Advantages Over Trace-Driven Simulation

- No need to pre-generate arrival times
- Natural handling of stochastic processes
- Better extensibility

---

## 5.2 Next-Event Simulation Examples

### Inventory System (`sis3`)

State variables:
- inventory level
- outstanding orders

Events:
- **demand arrival**
- **order arrival**

Replenishment decisions are triggered by inventory thresholds.

---

### Multi-Server Queue (`msq`)

Extends the single-server model to:
- multiple parallel servers
- shared queue

State variables:
- number of busy servers
- queue length

---

### Scheduling Logic

- Assign jobs to idle servers immediately
- Queue jobs when all servers are busy
- Departures free servers and may trigger service

---

## 5.3 Event List Management *(Optional)*

Efficient event list management is critical for performance.

---

### Data Structures

Common choices:
- ordered linked lists
- binary heaps
- priority queues

Trade-offs:
- insertion cost
- deletion cost
- memory overhead

---

### Program `ttr`

Implements:
- time-ordered event list
- efficient insert/remove operations
- reusable event list abstraction

---

## Statistics Collection

In next-event simulation, statistics are updated:
- at each event
- when state variables change

Time-weighted statistics require tracking:
- time since last event
- current state value

---

## Verification and Debugging

Next-event simulation facilitates:
- step-by-step tracing
- event-level debugging
- deterministic replay using fixed seeds

---

## Summary

- Next-event simulation advances time by events
- Event lists are central data structures
- Arrival and departure logic defines behavior
- Extensible to complex systems
- Efficient event management improves scalability

---

**End of Chapter 5**




<!--
Discrete-Event Simulation: A First Course
Lawrence Leemis, Steve Park
Chapter 6 — Discrete Random Variables
-->

# Chapter 6 — Discrete Random Variables

Discrete random variables play a central role in simulation when modeling systems with
**finite or countable outcomes**, such as routing decisions, inventory demands, or batch
sizes.

This chapter introduces discrete random variables and methods for **generating them from
uniform random numbers**.

---

## Sections
- 6.1 Discrete Random Variables
- 6.2 Generating Discrete Random Variables
- 6.3 Discrete Random Variable Applications (`sis4`)
- 6.4 Discrete Random Variable Models *(optional)*
- 6.5 Random Sampling and Shuffling *(optional)*

---

## 6.1 Discrete Random Variables

A **discrete random variable** takes values from a finite or countable set.

---

### Probability Mass Function (PMF)

For a discrete random variable `X`:

```

P(X = x) = f(x)

```

Properties:
- `f(x) ≥ 0`
- `Σ f(x) = 1`

---

### Expected Value

```

E[X] = Σ x f(x)

```

---

### Variance

```

Var(X) = E[(X − E[X])²]

```

---

### Cumulative Distribution Function (CDF)

```

F(x) = P(X ≤ x) = Σ_{y ≤ x} f(y)

```

The CDF is non-decreasing and right-continuous.

---

## 6.2 Generating Discrete Random Variables

Simulation requires transforming `U ~ U(0,1)` into a discrete random variable.

---

### Inverse Transform Method

Let `U ~ U(0,1)`.

Choose `X = x_i` if:

```

F(x_{i−1}) < U ≤ F(x_i)

```

This method:
- is simple
- works for any discrete distribution
- requires computation of the CDF

---

### Algorithm — Discrete Inverse Transform

1. Generate `U`
2. Find smallest `x_i` such that `F(x_i) ≥ U`
3. Return `x_i`

---

### Computational Considerations

- Linear search is acceptable for small support
- Binary search improves efficiency for large support
- Precomputed CDF tables are commonly used

---

## 6.3 Discrete Random Variable Applications

### Inventory Demand (`sis4`)

Discrete demand sizes are often modeled using:
- empirical distributions
- geometric distributions
- custom PMFs

Simulation steps:
1. Generate demand size
2. Update inventory level
3. Trigger reorder logic if needed

---

### Routing Decisions

Discrete random variables determine:
- next node in a network
- customer class
- job type

Routing probabilities must sum to 1.

---

## 6.4 Discrete Random Variable Models *(Optional)*

Common discrete distributions:
- Bernoulli
- Binomial
- Geometric
- Poisson

These models arise naturally in:
- arrivals
- failures
- batch processes

---

### Example — Bernoulli

```

P(X = 1) = p
P(X = 0) = 1 − p

```

---

### Example — Geometric

Models number of trials until first success.

---

## 6.5 Random Sampling and Shuffling *(Optional)*

Simulation often requires:
- sampling without replacement
- random permutations

---

### Sampling Without Replacement

Used when:
- selecting items from a finite population
- modeling depletion effects

---

### Random Shuffling

Produces a random permutation of elements.

Applications:
- randomized service order
- randomized experiment design

---

## Summary

- Discrete random variables model finite outcomes
- PMF and CDF define distributions
- Inverse transform enables generation
- Discrete models are essential in simulation
- Efficient algorithms improve scalability

---

**End of Chapter 6**





<!--
Discrete-Event Simulation: A First Course
Lawrence Leemis, Steve Park
Chapter 7 — Continuous Random Variables
-->

# Chapter 7 — Continuous Random Variables

Continuous random variables are essential for modeling **interarrival times, service times,
repair times**, and other quantities that vary over a continuum.

This chapter introduces continuous distributions and practical methods for **generating
continuous random variables from U(0,1)**.

---

## Sections
- 7.1 Continuous Random Variables
- 7.2 Generating Continuous Random Variables
- 7.3 Continuous Random Variable Applications (`ssq4`)
- 7.4 Continuous Random Variable Models *(optional)*
- 7.5 Nonstationary Poisson Processes *(optional)*
- 7.6 Acceptance–Rejection *(optional)*

---

## 7.1 Continuous Random Variables

A **continuous random variable** takes values on a continuous domain.

---

### Probability Density Function (PDF)

For a continuous random variable `X`:

```

f(x) ≥ 0
∫ f(x) dx = 1

```

Probabilities are computed as:

```

P(a ≤ X ≤ b) = ∫_a^b f(x) dx

```

---

### Cumulative Distribution Function (CDF)

```

F(x) = P(X ≤ x) = ∫_{−∞}^x f(t) dt

```

Properties:
- non-decreasing
- continuous
- `F(−∞)=0`, `F(∞)=1`

---

### Expected Value

```

E[X] = ∫ x f(x) dx

```

---

### Variance

```

Var(X) = E[(X − E[X])²]

```

---

## 7.2 Generating Continuous Random Variables

Simulation requires transforming `U ~ U(0,1)` into a continuous random variable.

---

### Inverse Transform Method

If `F(x)` is invertible:

```

X = F⁻¹(U)

```

This method:
- is exact
- is simple to implement
- requires closed-form inverse CDF

---

### Example — Exponential Distribution

PDF:
```

f(x) = λ e^{−λx}, x ≥ 0

```

CDF:
```

F(x) = 1 − e^{−λx}

```

Inverse:
```

X = −(1/λ) ln(1 − U)

```

---

### Numerical Considerations

- `1 − U` is often replaced by `U`
- logarithms require care for very small values

---

## 7.3 Continuous Random Variable Applications

### Single-Server Queue (`ssq4`)

Service and interarrival times are modeled using continuous distributions.

Typical choices:
- exponential (memoryless)
- deterministic
- custom empirical models

---

### Service-Time Modeling

Choice of distribution affects:
- waiting times
- queue lengths
- system variability

Mean alone is insufficient; **variance matters**.

---

## 7.4 Continuous Random Variable Models *(Optional)*

Common distributions:
- Uniform
- Exponential
- Erlang
- Hyperexponential
- Weibull

Each distribution captures different variability characteristics.

---

### Variability Comparison

Distributions with equal means may have vastly different variances,
leading to different system behavior.

---

## 7.5 Nonstationary Poisson Processes *(Optional)*

Arrival rates may vary over time:

```

λ = λ(t)

```

Such processes are **nonstationary**.

Applications:
- time-of-day effects
- seasonal demand
- workload bursts

---

### Thinning Method

Used to generate nonstationary Poisson arrivals by:
- bounding with a maximum rate
- probabilistic rejection

---

## 7.6 Acceptance–Rejection *(Optional)*

Used when inverse CDF is unavailable.

---

### Algorithm — Acceptance–Rejection

1. Generate candidate `Y` from easy distribution `g(y)`
2. Generate `U ~ U(0,1)`
3. Accept `Y` if:
```

U ≤ f(Y) / (c g(Y))

```

where `c` bounds `f(x)/g(x)`.

---

### Trade-offs

- Flexible
- Potentially inefficient
- Requires bounding function

---

## Summary

- Continuous random variables model time and quantity
- PDF and CDF define distributions
- Inverse transform is preferred when available
- Variability strongly affects performance
- Advanced techniques support complex processes

---

**End of Chapter 7**





<!--
Discrete-Event Simulation: A First Course
Lawrence Leemis, Steve Park
Chapter 8 — Output Analysis
-->

# Chapter 8 — Output Analysis

The purpose of output analysis is to extract **reliable performance measures** from
simulation output. Because simulation models are stochastic, output data must be treated
using **statistical methods**.

A central challenge is that simulation observations are often **correlated**, especially
in steady-state simulations.

---

## Sections
- 8.1 Interval Estimation (`estimate`)
- 8.2 Monte Carlo Estimation
- 8.3 Finite-Horizon and Infinite-Horizon Statistics
- 8.4 Batch Means
- 8.5 Steady-State Single-Server Statistics *(optional)*

---

## 8.1 Interval Estimation

Point estimates alone are insufficient; we must quantify **uncertainty**.

---

### Confidence Intervals

Given observations:
```

x1, x2, ..., xn

```

Sample mean:
```

x̄

```

Sample standard deviation:
```

s

```

A `(1 − α)` confidence interval for `E[X]` is:

```

x̄ ± t_{α/2, n−1} · (s / √n)

```

where `t` is the Student’s t quantile.

---

### Interpretation

- Interval width reflects uncertainty
- Increasing `n` reduces interval width
- Valid only if observations are approximately independent

---

### Program `estimate`

- Reads simulation output
- Computes confidence intervals
- Automates statistical reporting

---

## 8.2 Monte Carlo Estimation

Monte Carlo estimation applies when:
- observations are **independent**
- output corresponds to a **static** random variable

---

### Example

Estimating:
```

θ = E[g(X)]

```

Estimator:
```

θ̂ = (1/n) Σ g(X_i)

```

By the Central Limit Theorem:
```

θ̂ ≈ Normal(θ, σ²/n)

```

---

### Accuracy vs Cost

- Variance determines required sample size
- High variance ⇒ many samples needed
- Variance reduction techniques are valuable

---

## 8.3 Finite-Horizon and Infinite-Horizon Statistics

Simulation output differs depending on the **time horizon**.

---

### Finite-Horizon Statistics

Used when:
- system has a natural termination
- interest is in transient behavior

Examples:
- total waiting time over a day
- number of failures in a shift

Observations are typically independent across replications.

---

### Infinite-Horizon (Steady-State) Statistics

Used when:
- system runs indefinitely
- interest is in long-run averages

Examples:
- average queue length
- average response time

Steady-state output is usually **correlated**.

---

### Initialization Bias

Initial conditions distort early observations.

Common remedies:
- discard initial observations (warm-up)
- start from realistic initial state

---

## 8.4 Batch Means

Batch means is a primary technique for **steady-state output analysis**.

---

### Idea

1. Run one long simulation
2. Divide output into `k` contiguous batches
3. Compute batch averages
4. Treat batch means as approximately independent

---

### Procedure

Let `Y(t)` be a time-dependent output.

Divide simulation time into batches of equal length `B`.

Batch mean `j`:
```

Ȳ_j = (1/B) ∫ Y(t) dt  over batch j

```

---

### Confidence Interval Using Batch Means

Treat `Ȳ_1, ..., Ȳ_k` as i.i.d.:

```

Ȳ ± t_{α/2, k−1} · (s_b / √k)

```

where `s_b` is the standard deviation of batch means.

---

### Trade-offs

- Larger batches → less correlation
- Fewer batches → less statistical power
- Balance is required

---

## 8.5 Steady-State Single-Server Statistics *(Optional)*

For certain models, analytical steady-state results are known.

Simulation results can be:
- compared to theory
- used for validation

Examples:
- M/M/1 queue
- utilization
- mean delay

---

## Verification vs Validation in Output Analysis

- **Verification**: statistics computed correctly?
- **Validation**: statistics represent the real system?

Output analysis is central to both.

---

## Summary

- Simulation output is random and often correlated
- Confidence intervals quantify uncertainty
- Finite-horizon and steady-state require different methods
- Batch means enables steady-state inference
- Poor output analysis invalidates simulation conclusions

---

**End of Chapter 8**
