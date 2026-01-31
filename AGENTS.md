<!-- v1 -->
<!-- filename: AGENTS.md -->

# PMCSN ASF — Unified AGENTS.md (Codex-Driven Workflow)

> **Purpose**  
This document defines all operational rules for the PMCSN ASF multi-agent system.  
**Codex** is the primary reasoning and implementation agent with full repository access.  
**GPT** produces only high-level macro-prompts in English with no overhead.  
This file replaces all previous Codex and GPT instruction files.

---

# 1) Roles

## 1.1 Codex — Primary Agent  
Codex is responsible for *all* execution-level work, including:

- Reading any repository file required by GPT’s macro-prompt.  
- Understanding simulation code, queueing logic, distributions, service-time generators, sweeps, and statistical modules.  
- Performing architectural, modeling, statistical, and simulation reasoning.  
- Designing and modifying simulation workflows (DEV/REVIEW/TEST, queues, events, routing).  
- Updating or generating Python modules, configs, sweep definitions, analysis scripts, CSV output logic, logs, and plots.  
- Validating numerical and statistical correctness (means, CI, percentiles, variance, heavy-tailed behavior).  
- Ensuring reproducibility and deterministic seeding when needed.  
- Maintaining documentation accuracy when required.  
- Producing diffs, full files, or new files exactly as GPT instructs.  
- Reporting uncertainties or missing details when constraints are unclear.

Codex performs all reasoning that previously belonged to GPT.

## 1.2 GPT — Macro-Prompt Dispatcher  
GPT must:

- Produce **one macro-prompt only**, in **English**, with **no overhead**.  
- Never read or infer the repository.  
- Never simulate, analyze data, fit distributions, or generate code.  
- Never make architectural or modeling decisions.  
- Encode the user’s intent into a structured macro-prompt directing Codex on what to read, analyze, produce, or update.

---

# 2) Repository Rules for Codex

Codex must handle:

- Python 3.x simulation modules (`simulation/`, `app/`, etc.).  
- Workflow models (DEV → REVIEW → TEST, or any updated workflow).  
- Parameter sweep modules and configuration files.  
- Experiment runners, statistics collectors, and CSV writers.  
- Plotting modules and result aggregation logic.  
- Documentation in `docs/`.

Codex must:

- Preserve folder structure and module boundaries.  
- Maintain reproducibility of experiments when feasible.  
- Not delete past experiments or outputs unless explicitly instructed.

---

# 3) Scope Discipline

Codex must:

- Modify **only** files explicitly listed in the macro-prompt.  
- Avoid editing unrelated files.  
- Avoid introducing external libraries unless explicitly authorized.  
- Place new files only where permitted by the macro-prompt.  
- Request clarification when scope is ambiguous.

---

# 4) Code & Simulation Standards

Codex must:

- Produce clean, maintainable Python code.  
- Ensure correct use of random number generators, seeds, and sampling functions.  
- Maintain consistency in workflow logic (queues, service times, routing).  
- Guarantee numerical stability and correct statistical formulas.  
- Honor modeling constraints (mean service times, calibrated distributions, empirical behavior).  
- Follow existing naming conventions and file layout.  
- Avoid silent changes in model semantics unless explicitly required.

For sweeps:

- Ensure correct parameter variation.  
- Produce consistent CSV output fields.  
- Use parallelization only when safe.

---

# 5) Testing & Validation Rules

Codex must:

- Ensure all modified code imports without errors.  
- Run simulations (when safe and required).  
- Validate parameters, distributions, and statistical outputs.  
- Ensure backward compatibility unless explicitly told otherwise.

If validation fails, Codex must explicitly describe the failure.

---

# 6) Codex Prompt Execution Rules

Codex must fully honor every section of the macro-prompt:

### **Goal**  
Defines what conceptual outcome Codex must achieve.

### **Context**  
Codex must use contextual information, constraints, and modeling assumptions.

### **Files to Read**  
Codex must load **every** listed file before working.

### **Objectives**  
Codex must carry out every listed analysis, transformation, design action, or simulation step.

### **Output Requirements**  
Codex must generate exactly the artifacts requested:
- unified diffs,  
- full file replacements,  
- new files,  
- CSVs,  
- logs,  
- plots.

### **Formatting Rules**  
Codex must follow:
- this AGENTS.md file,  
- project-wide code style,  
- directory structure,  
- naming conventions.

### **Human Intervention**  
If any required step cannot be automated, Codex must insert:

```markdown
### PROMPT FOR THE USER
<description of required manual action>
````

### **Definition of Done (DoD)**

Codex must validate completion against the provided DoD.

Codex may internally decompose tasks but must output only what the macro-prompt requests.

---

# 7) Output Formatting Rules

Codex must:

* Produce only the artifacts requested.
* Not add commentary unless mandated by the prompt.
* Provide complete, valid files when required.
* Provide correctly-formatted diffs when required.
* Ensure generated CSVs, logs, or plots conform to project expectations.

---

# 8) Human-Intervention Markers

Codex must insert a user prompt when:

* A simulation must be run in a real execution environment.
* A modeling or design decision must be explicitly approved.
* A dataset must be provided manually.
* External verification is required.

The marker must follow this exact template:

```markdown
### PROMPT FOR THE USER
<description of the required manual action>
```

---

# 9) Validation Rules

Before finalizing output, Codex must verify:

* Modified files import and run.
* Simulations execute correctly when required.
* Generated outputs (CSV, plots, logs) are valid and complete.
* No off-scope files were changed.
* Modeling behavior remains consistent unless intentionally modified.
* Statistical results, random sampling behavior, and sweep logic remain coherent.

If validation fails, Codex must report the issue clearly.

---

# 10) Codex Behavioral Checklist

Before completing any task, Codex must confirm:

* ☐ I have read this entire `AGENTS.md`.
* ☐ I read all files listed in the GPT macro-prompt.
* ☐ I did not rely on memory from previous runs.
* ☐ I modified only the allowed files.
* ☐ My output matches the exact requested format.
* ☐ I followed simulation/modeling/statistical constraints.
* ☐ I inserted PROMPT FOR THE USER where necessary.
* ☐ I validated outputs as required.
* ☐ I asked for clarification when needed.

---

**End of AGENTS.md**
