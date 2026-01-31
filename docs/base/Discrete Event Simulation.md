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
