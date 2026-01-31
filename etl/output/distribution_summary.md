# Phase Duration Distribution Summary

Source Jira file: `/home/cantarell/PycharmProjects/PM_ASF/etl/output/csv/jira_issues_raw.csv`.
Source PR file: `/home/cantarell/PycharmProjects/PM_ASF/etl/output/csv/github_prs_raw.csv`.

## Phase boundary assumptions
- Dev start: Jira issue creation timestamp.
- Review start: earliest PR creation timestamp linked to the Jira key.
- Review end: latest PR merge timestamp (fallback to PR close if merge missing).
- Testing end: Jira resolution timestamp (fallback to review end if missing).

## Distribution summary
- **dev**: 190 samples, best fit `lognormal`.
- **review**: 190 samples, best fit `weibull`.
- **testing**: 122 samples, best fit `lognormal`.

## Missing boundary diagnostics
Tickets with missing/invalid boundaries: 973.
Sample keys: BOOKKEEPER-7, BOOKKEEPER-680, BOOKKEEPER-14, BOOKKEEPER-10, BOOKKEEPER-691, BOOKKEEPER-694, BOOKKEEPER-13, BOOKKEEPER-2, BOOKKEEPER-8, BOOKKEEPER-3.

### PROMPT FOR THE USER
Jira transition data was not available in the raw export. Please provide Jira transition timestamps or confirm that the fallback boundary inference rules should remain authoritative.
