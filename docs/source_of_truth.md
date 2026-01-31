# PMCSN ASF Project — Source of Truth

## 1. Project Overview and Academic Context
- Course: **Performance Modeling of Computer Systems and Networks (AA 2024/2025)**.【F:docs/base/Progetto2425.md†L1-L4】
- This project is a **modeling, simulation, and performance evaluation** study, carried out according to the professor’s requirements and the textbook methodology for discrete-event simulation.【F:docs/base/Progetto2425.md†L6-L23】【F:docs/base/Discrete Event Simulation.md†L66-L105】
- Case study domain: **Apache Software Foundation (ASF), Apache BookKeeper** software development workflow, using Jira and GitHub artifacts as the primary evidence base.【F:docs/base/ASF project info.md†L1-L18】【F:docs/base/ASF project info.md†L89-L138】

## 2. Official Requirements and Constraints (from `progetto2425.md`)
### 2.1 Mandatory Steps
1. Identify a system to study.【F:docs/base/Progetto2425.md†L6-L11】
2. Identify the objectives of the study.【F:docs/base/Progetto2425.md†L6-L11】
3. Submit the case study and objectives to the instructor (docente).【F:docs/base/Progetto2425.md†L6-L11】
4. Build a simulation model following **Algorithm 1.1.1 and Algorithm 1.1.2** from the textbook.【F:docs/base/Progetto2425.md†L10-L13】【F:docs/base/Discrete Event Simulation.md†L78-L119】
5. Perform a **transient analysis** to show the initial system behavior over time, verifying convergence and when it occurs.【F:docs/base/Progetto2425.md†L13-L16】
6. Design experiments appropriately for the case study and objectives.【F:docs/base/Progetto2425.md†L16-L17】
7. Present results **both graphically and numerically**, including summary tables.【F:docs/base/Progetto2425.md†L16-L18】

### 2.2 Simulation Guidelines and References
- The simulation must follow the guidelines of the referenced paper **“MANET Simulation Studies: The Incredibles.”**【F:docs/base/Progetto2425.md†L18-L24】
- The project must be grounded in the discrete-event simulation methodology and terminology of the textbook.【F:docs/base/Progetto2425.md†L10-L13】【F:docs/base/Discrete Event Simulation.md†L66-L119】

### 2.3 Individual vs Group Constraints
- The project can be **individual or a group of up to 3 people**. Group formation must be communicated to the instructor by email, including all participants.【F:docs/base/Progetto2425.md†L26-L31】
- **Group projects must include an improvement algorithm** (an evolution of the model) and repeat the required steps for the improved model.【F:docs/base/Progetto2425.md†L32-L33】

### 2.4 Evaluation Criteria
The final work is evaluated on:【F:docs/base/Progetto2425.md†L35-L40】
- Significance and relevance of the case study.
- Modeling ability and analysis of results.
- Clarity and synthesis in the description.
- Completeness of the study.

### 2.5 Required Deliverables
- A **written report** containing:【F:docs/base/Progetto2425.md†L42-L61】
  - Introduction describing the chosen case, motivations, and expected improvements.
  - Description of the system, objectives, models, and results, following the course methodology and development algorithm.
  - Discussion of implementation choices for essential/critical simulation parts.
- The **source code** must be delivered (email or link), excluding system libraries.【F:docs/base/Progetto2425.md†L55-L60】

### 2.6 Presentation Requirements
- Presentation duration: **≤ 10 minutes per participant**, or **≤ 20 minutes total for individual projects**. The instructor may request live execution demos.【F:docs/base/Progetto2425.md†L62-L66】

## 3. Reference Methodology (from `Discrete Event Simulation.md`)
### 3.1 Core Terminology (Must Be Used Consistently)
- **Model types**: stochastic, dynamic, discrete-event (required for this project).【F:docs/base/Discrete Event Simulation.md†L66-L92】
- **Model levels**: conceptual, specification, computational.【F:docs/base/Discrete Event Simulation.md†L128-L137】
- **Verification vs Validation**: verification checks correct implementation; validation checks real-system fidelity.【F:docs/base/Discrete Event Simulation.md†L121-L125】【F:docs/base/Discrete Event Simulation.md†L559-L563】

### 3.2 Algorithm 1.1.1 — Discrete-Event Model Development (Project Mapping)
1. **Determine objectives** → define what questions the ASF workflow model must answer.【F:docs/base/Discrete Event Simulation.md†L78-L105】
2. **Build a conceptual model** → define system entities, states, flows, and queues.【F:docs/base/Discrete Event Simulation.md†L78-L105】
3. **Develop a specification model** → define input models, distributions, and data sources (Jira/GitHub).【F:docs/base/Discrete Event Simulation.md†L78-L105】
4. **Construct the computational model** → implement discrete-event simulation logic (event list, clock, state).【F:docs/base/Discrete Event Simulation.md†L78-L105】【F:docs/base/Discrete Event Simulation.md†L511-L556】
5. **Verify** → ensure implementation matches the specification model.【F:docs/base/Discrete Event Simulation.md†L78-L105】【F:docs/base/Discrete Event Simulation.md†L559-L563】
6. **Validate** → ensure the model represents the ASF workflow adequately.【F:docs/base/Discrete Event Simulation.md†L78-L105】【F:docs/base/Discrete Event Simulation.md†L559-L563】

### 3.3 Algorithm 1.1.2 — Simulation Study (Project Mapping)
7. **Design experiments** → plan scenarios, parameters, horizons, replications/batches.【F:docs/base/Discrete Event Simulation.md†L111-L119】
8. **Perform production runs** → execute the simulation for each experiment design.【F:docs/base/Discrete Event Simulation.md†L111-L119】
9. **Analyze output statistically** → compute estimates, confidence intervals, and summaries.【F:docs/base/Discrete Event Simulation.md†L111-L119】【F:docs/base/Discrete Event Simulation.md†L1647-L1735】
10. **Make decisions** → infer bottlenecks and improvement implications for ASF workflow.【F:docs/base/Discrete Event Simulation.md†L111-L119】
11. **Document results** → include graphs, tables, and interpretation in the report.【F:docs/base/Discrete Event Simulation.md†L111-L119】

### 3.4 Verification vs Validation
- **Verification**: prove the computational model correctly implements the specification model (logic, event handling, statistics).【F:docs/base/Discrete Event Simulation.md†L559-L563】
- **Validation**: show the model sufficiently represents the real ASF development process (compare to Jira/GitHub data).【F:docs/base/Discrete Event Simulation.md†L559-L563】【F:docs/base/ASF project info.md†L89-L138】

### 3.5 Transient vs Steady-State Analysis
- **Transient (finite-horizon) analysis**: required to show the initial behavior over time and whether/when convergence occurs.【F:docs/base/Progetto2425.md†L13-L16】【F:docs/base/Discrete Event Simulation.md†L1701-L1715】
- **Steady-state (infinite-horizon) analysis**: appropriate for long-run averages, with attention to correlation and warm-up bias.【F:docs/base/Discrete Event Simulation.md†L1701-L1735】

## 4. Case Study Definition: ASF / Apache BookKeeper
- The system under study is the **software development workflow** of Apache BookKeeper, an ASF project with volunteer-driven, asynchronous collaboration, and peer review norms (“Apache Way”).【F:docs/base/ASF project info.md†L1-L33】
- The model focuses on the end-to-end life of a feature or bugfix: **from Jira ticket creation to final release**.【F:docs/base/ASF project info.md†L3-L18】【F:docs/base/ASF project info.md†L53-L68】
- The study uses **Jira** (issue states and transitions) and **GitHub** (PRs, commits, CI results) as data sources for input modeling and validation.【F:docs/base/ASF project info.md†L89-L138】

## 5. Conceptual Model (Domain-Driven, No Implementation Details)
### 5.1 System Entities
- **Issues/Tickets**: feature requests or bug reports tracked in Jira, approved for development.【F:docs/base/ASF project info.md†L155-L163】
- **Contributors/Developers**: volunteer-driven capacity, can pick up and implement issues asynchronously.【F:docs/base/ASF project info.md†L19-L33】【F:docs/base/ASF project info.md†L155-L163】
- **Reviewers**: peers who approve or request changes on PRs.【F:docs/base/ASF project info.md†L165-L183】
- **CI/Test Infrastructure**: executes automated checks during PR and additional testing after merge.【F:docs/base/ASF project info.md†L165-L193】

### 5.2 States and Queues
- **Jira states**: Open → In Progress → Review → Testing → Resolved/Released, with possible feedback loops back to development when tests fail or revisions are requested.【F:docs/base/ASF project info.md†L155-L193】
- **Queues**: waiting-to-be-taken (Open), waiting-for-review, waiting-for-testing, waiting-for-release, with rework cycles (Review → Development → Review; Testing → Development).【F:docs/base/ASF project info.md†L165-L193】

### 5.3 Events and Feedback Loops
- Ticket creation and approval (entry into system).【F:docs/base/ASF project info.md†L155-L163】
- PR opened (start of review).【F:docs/base/ASF project info.md†L165-L173】
- Review outcome (approve or request changes), triggering feedback loops.【F:docs/base/ASF project info.md†L165-L183】
- Merge and testing phase; testing failures trigger rework cycles.【F:docs/base/ASF project info.md†L186-L193】
- Release completion (exit from system).【F:docs/base/ASF project info.md†L186-L193】

### 5.4 Performance Metrics (Conceptual)
- **Resolution/response time** from ticket creation to release/closure.【F:docs/base/ASF project info.md†L69-L83】
- **Waiting time** from ticket opening to developer pick-up (queue time).【F:docs/base/ASF project info.md†L69-L83】
- **Bottleneck identification** across phases (e.g., review delay, testing delay).【F:docs/base/ASF project info.md†L69-L83】

## 6. Specification Model (Stochastic Assumptions, Inputs, Metrics)
### 6.1 Input Data Sources
- **Jira**: issue type, status transition timestamps, resolution/closure times, and linkages to related issues.【F:docs/base/ASF project info.md†L95-L115】
- **GitHub**: PR open/close/merge times, number of iterations, CI outcomes; linked to Jira via issue IDs in PR titles/commits.【F:docs/base/ASF project info.md†L117-L138】

### 6.2 Input Modeling Approach
- Use **trace-driven** or **parametric** models for arrivals and service times, depending on data availability and suitability, following textbook input modeling guidance.【F:docs/base/Discrete Event Simulation.md†L1811-L1881】
- Maintain **reproducibility** via deterministic seeding when using parametric models (consistent with discrete-event simulation practice).【F:docs/base/Discrete Event Simulation.md†L272-L314】【F:docs/base/Discrete Event Simulation.md†L552-L556】

### 6.3 Metrics to Estimate
- Mean/variance of resolution time, waiting time, review time, testing time, and rework cycles (derived from state durations).【F:docs/base/ASF project info.md†L69-L83】【F:docs/base/ASF project info.md†L155-L193】
- Queue lengths or backlog indicators for each workflow stage (time-averaged).【F:docs/base/Discrete Event Simulation.md†L511-L556】

### 6.4 Required Statistical Outputs
- Sample means, variances, and **confidence intervals** for each metric (finite-horizon or steady-state as appropriate).【F:docs/base/Discrete Event Simulation.md†L1647-L1689】
- Output must include **both numerical tables and graphs** per project requirements.【F:docs/base/Progetto2425.md†L16-L18】

## 7. Computational Model Guidelines
- Implement a **discrete-event simulation** with:
  - event list (sorted by event time),
  - simulation clock,
  - state variables,
  - event handlers (arrival, review outcome, testing outcome, release),
  - statistics collection on events/state changes.【F:docs/base/Discrete Event Simulation.md†L498-L556】
- **Termination conditions** must be defined explicitly (e.g., number of issues processed or simulation horizon).【F:docs/base/Discrete Event Simulation.md†L541-L548】
- Use **separate random streams** if multiple stochastic inputs are generated to avoid correlation artifacts.【F:docs/base/Discrete Event Simulation.md†L411-L449】
- Maintain **verification** (logic correctness) and **validation** (data fidelity) as explicit checkpoints.【F:docs/base/Discrete Event Simulation.md†L78-L105】【F:docs/base/Discrete Event Simulation.md†L559-L563】

## 8. Experiment Design and Simulation Plan
1. **Define objectives and questions** to be answered (e.g., average resolution time, bottlenecks).【F:docs/base/Discrete Event Simulation.md†L78-L105】【F:docs/base/ASF project info.md†L69-L83】
2. **Choose input modeling strategy** (trace-driven or parametric) using Jira/GitHub data.【F:docs/base/Discrete Event Simulation.md†L1811-L1881】【F:docs/base/ASF project info.md†L95-L138】
3. **Specify the model** (entities, events, queues, and routing rules).【F:docs/base/Discrete Event Simulation.md†L498-L556】【F:docs/base/ASF project info.md†L155-L193】
4. **Implement and verify** the computational model (logic and statistics).【F:docs/base/Discrete Event Simulation.md†L78-L105】【F:docs/base/Discrete Event Simulation.md†L559-L563】
5. **Conduct transient analysis** to identify convergence behavior and warm-up requirements.【F:docs/base/Progetto2425.md†L13-L16】【F:docs/base/Discrete Event Simulation.md†L1701-L1735】
6. **Design experiments** (replications or batch means depending on horizon).【F:docs/base/Discrete Event Simulation.md†L111-L119】【F:docs/base/Discrete Event Simulation.md†L1701-L1735】
7. **Run production simulations** and collect output metrics.【F:docs/base/Discrete Event Simulation.md†L111-L119】
8. **Analyze output statistically** with confidence intervals and summary tables/graphs.【F:docs/base/Discrete Event Simulation.md†L1647-L1735】【F:docs/base/Progetto2425.md†L16-L18】
9. **Validate** results by comparing key metrics to observed Jira/GitHub data distributions.【F:docs/base/Discrete Event Simulation.md†L559-L563】【F:docs/base/ASF project info.md†L95-L138】
10. **Document results** in the report with clear interpretations and limitations.【F:docs/base/Discrete Event Simulation.md†L111-L119】【F:docs/base/Progetto2425.md†L42-L61】

## 9. Output Analysis and Presentation of Results
- Use **sample statistics** (mean, variance, standard deviation) to summarize outputs.【F:docs/base/Discrete Event Simulation.md†L652-L699】
- Provide **confidence intervals** for estimated performance measures.【F:docs/base/Discrete Event Simulation.md†L1647-L1689】
- Distinguish between **finite-horizon** and **steady-state** statistics and apply appropriate methods (replications vs batch means).【F:docs/base/Discrete Event Simulation.md†L1701-L1735】
- Provide **both graphical and tabular summaries** as mandated by the project requirements.【F:docs/base/Progetto2425.md†L16-L18】

## 10. Improvement / Evolution of the Model (Mandatory for Group Projects)
- If the project is a group project, an **improvement algorithm** must be developed to evolve the model and reapply the study steps (Algorithm 1.1.1 and 1.1.2).【F:docs/base/Progetto2425.md†L32-L33】【F:docs/base/Discrete Event Simulation.md†L78-L119】
- The improvement must be justified with respect to observed bottlenecks or performance issues in the baseline model.【F:docs/base/ASF project info.md†L69-L83】

## 11. Final Deliverables and Evaluation Alignment
- **Report**: introduction, system description, objectives, models, results, and implementation choices, as required by the professor.【F:docs/base/Progetto2425.md†L42-L61】
- **Code**: simulation source code delivered separately (email or link), excluding system libraries.【F:docs/base/Progetto2425.md†L55-L60】
- **Presentation**: ≤ 10 minutes per participant (≤ 20 minutes if individual), with possible live demo requests.【F:docs/base/Progetto2425.md†L62-L66】
- **Evaluation**: ensure clarity, modeling rigor, and completeness, aligned to the evaluation grid.【F:docs/base/Progetto2425.md†L35-L40】

## 12. Rules for Future Development (This Document as Single Source of Truth)
- **No undocumented assumptions** are allowed; all modeling decisions must be traced to this document and the cited sources within it.【F:docs/base/Progetto2425.md†L42-L61】【F:docs/base/Discrete Event Simulation.md†L66-L119】
- **Documentation must reflect the implemented model**: any implementation change requires an update to this document and the project report to remain consistent with the simulation model and output interpretation.【F:docs/base/Progetto2425.md†L42-L61】
- **In-scope**: modeling the ASF BookKeeper workflow using Jira/GitHub data, discrete-event simulation, transient analysis, experiment design, and output analysis per textbook methods.【F:docs/base/ASF project info.md†L89-L138】【F:docs/base/Progetto2425.md†L6-L18】
- **Out-of-scope**: assumptions or requirements not explicitly documented in the provided sources.

### PROMPT FOR THE USER
In `ASF project info.md`, the objectives list contains an incomplete requirement: “Identificare il numero minimo …”. Please specify the full requirement (e.g., “minimum number of developers/reviewers/servers” or another quantity) so it can be documented precisely and mapped to model decisions.【F:docs/base/ASF project info.md†L79-L85】
