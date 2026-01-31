# Ticket Resolution Phase Distribution Summary

## Data Sources
- GitHub PR raw data: `etl/output/csv/github_prs_raw.csv`
- Jira raw data: `etl/output/csv/jira_issues_raw.csv`

## Generated Plots
- `plots/dev_phase_histogram.png`
- `plots/review_phase_histogram.png`
- `plots/testing_phase_histogram.png`

## Phase Boundary Assumptions
- Dev start: Jira issue creation timestamp.
- Review start: first GitHub PR creation timestamp linked to the Jira key.
- Review end / testing start: latest PR merge (or close) timestamp for the Jira key.
- Testing end: Jira resolution timestamp (fallback to PR merge/close when missing).

## Distribution Fit Summary
### Dev Phase
- Count: 188
- Mean (hours): 4027.9732
- Median (hours): 16.8603
- Variance (hours^2): 131775124.3210
- Best-fit distribution (log-likelihood comparison): lognormal

### Review Phase
- Count: 190
- Mean (hours): 918.1168
- Median (hours): 189.7725
- Variance (hours^2): 9397190.1846
- Best-fit distribution (log-likelihood comparison): weibull

### Testing Phase
- Count: 105
- Mean (hours): 278.5250
- Median (hours): 0.0131
- Variance (hours^2): 1266977.5448
- Best-fit distribution (log-likelihood comparison): lognormal

## Missing or Inferred Boundaries
- Tickets with missing boundaries: 975
- Example ticket keys: BOOKKEEPER-1, BOOKKEEPER-10, BOOKKEEPER-100, BOOKKEEPER-1000, BOOKKEEPER-1001, BOOKKEEPER-1002, BOOKKEEPER-1003, BOOKKEEPER-1004, BOOKKEEPER-1005, BOOKKEEPER-1006

### PROMPT FOR THE USER
The Jira dataset `jira_tickets_raw.csv` was not found in the repository, and no status-transition history is available in the fallback Jira export. Please provide the expected Jira ticket export with transition timestamps (or confirm that the fallback file should be used) so phase boundaries can be derived reliably for all tickets.
