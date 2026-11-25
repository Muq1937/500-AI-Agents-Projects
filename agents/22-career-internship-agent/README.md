# AI Career & Internship Agent

A LangGraph multi-agent pipeline that compares a resume with a job description,
identifies the missing skills, and produces a personalized plan to close the gap
and prepare for the interview. Built with students and internship applicants in
mind: it treats projects, coursework, and activities as first-class evidence, not
just work history.

**Framework**: LangGraph  
**LLM**: Gemini 3.6 Flash (Google)

## What it does

1. **Resume Analyzer**: extracts a structured profile (skills, projects, education, activities)
2. **JD Analyzer**: splits the job description into must-have and nice-to-have requirements
3. **Matching Agent**: judges every requirement as met / partial / missing, with evidence from the resume
4. **Gap Analysis Agent**: ranks unmet requirements as blocking / important / nice-to-have, citing the JD
5. **Recommendation Agent**: a week-by-week learning plan, small project ideas that close key gaps,
   specific resume improvements, and interview questions targeted at weak spots

It can also **search a catalog of job postings** for the ones that best fit your skills
(see [Find jobs that match your skills](#find-jobs-that-match-your-skills)).

### How the score works

The LLM only judges evidence; the score is computed in code, so it can be traced
back to the per-requirement assessments:

| | met | partial | missing |
|---|---|---|---|
| must-have (weight 2) | 2 | 1 | 0 |
| nice-to-have (weight 1) | 1 | 0.5 | 0 |

`score = 100 × earned / maximum`. The code also guarantees that every JD requirement
is assessed exactly once, and enforces the severity rules: a missing must-have is always
**blocking**, a partial must-have always **important**, and a nice-to-have is never blocking.
The gap list is reconciled with the assessments: every unmet requirement appears once,
and gaps the LLM invents are dropped.

Eligibility criteria (enrollment, graduation date, availability, location) are checked
separately as met / not-met / unclear and are not part of the score. A criterion the resume
contradicts is always a **blocking** gap; one the resume doesn't mention is only marked unclear.

## Architecture

```
Resume ──> [Resume Analyzer] ─┐
                              ├─> [Matching Agent] ─> [Gap Analysis Agent] ─> [Recommendation Agent] ─> Plan
JD ─────> [JD Analyzer] ──────┘
```

The resume and JD analyzers run in parallel; every stage returns a typed Pydantic object
via Gemini structured output.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your API key
```

Get a free API key: Google AI Studio, https://aistudio.google.com/apikey

To use a different Gemini model, set `GEMINI_MODEL` in `.env` (default: `gemini-3.6-flash`).

## Run

```bash
# Built-in sample (Year 3 Computer Engineering student, backend internship)
python agent.py

# Your own resume and job posting
python agent.py --resume path/to/resume.pdf --job-desc path/to/job_posting.txt

# 4-week plan, raw JSON output
python agent.py --weeks 4 --json
```

`--resume` and `--job-desc` accept a `.pdf` or text file, or raw text. A file name that doesn't
exist (`resumee.pdf`) is reported as an error instead of being analyzed as text.
Runtime: five LLM calls, typically under a minute.

## Find jobs that match your skills

`--find-jobs` ranks a whole catalog of postings against your skills, best fit first.

```bash
# Type your skills: no LLM call, no API key needed
python agent.py --find-jobs --skills "Python, PostgreSQL, Docker, FastAPI, Git"

# Skills extracted from your resume (1 LLM call), searched against your own postings
python agent.py --find-jobs --resume resume.pdf --jobs jobs.json --top 10

# Re-check the top 3 with the LLM Matching Agent (3 more LLM calls, run in parallel)
python agent.py --find-jobs --resume resume.pdf --rerank 3

# Rank by career fit, not just skill coverage (still no LLM call)
python agent.py --find-jobs --skills "Python, PostgreSQL, Docker" --ranking career
python agent.py --find-jobs --resume resume.pdf --ranking career --weights "technical=0.5,domain=0.3"
```

Without `--jobs`, it searches a built-in catalog of 14 fictional internships in Thailand
(backend, frontend, data, ML, mobile, DevOps, QA, embedded, design, security, games).
`--jobs` takes a JSON list. Each posting has a `title` and `must_have_skills` /
`nice_to_have_skills`, or just a `description`, which the JD Analyzer turns into skills
(one LLM call per posting):

```json
[
  {"id": "acme-be", "title": "Backend Intern", "company": "Acme",
   "must_have_skills": ["Python", "SQL"], "nice_to_have_skills": ["Docker", "AWS or GCP"]},
  {"title": "Data Intern", "description": "We're looking for a student who knows SQL and Python..."}
]
```

**How it works.** Running the Matching Agent on every posting would take one LLM call per job,
so search uses a deterministic matcher on the same met / partial / missing scale and weights as
the main score. It knows these relationships between skills:

| | Example | Result |
|---|---|---|
| Aliases | `Postgres` for *PostgreSQL*, `ReactJS` for *React* | met |
| Versions | `Python 3.11` for *Python*, `Java 17` for *Java* | met |
| Implications | `PostgreSQL` for *SQL*, `Next.js` for *JavaScript*, `scikit-learn` for *Machine Learning* | met |
| Specializations | `Docker Compose` for *Docker*, `Spring Boot` for *Spring* | met |
| | `Docker` for *Docker Compose* (the broader skill only) | partial |
| Families | `MySQL` for *PostgreSQL*, `Flask` for *FastAPI or Django* | partial |
| Compounds | `HTML` alone for *HTML/CSS* | partial |
| Umbrella terms | `Azure` or general cloud for *Cloud (AWS/GCP)* | partial |

Requirements with alternatives (`FastAPI or Django`, `Cloud (AWS/GCP)`) are met by any one of them,
and the same skill listed twice under different names (`Postgres`, `PostgreSQL`) counts once.
Ties are broken by the number of missing must-haves. A skill that isn't in the matcher's tables
(for example `Terraform`) only matches its own name, in any script (Thai names work too). That is
what `--rerank` is for: the LLM judges the top results from your full resume, including how deep
each skill goes.

**Career ranking** (`--ranking career`, opt-in) scores four parts and reports each one:

| Part | Default weight | What it measures |
|---|---|---|
| technical | 40% | skill coverage, with specific tools (`FastAPI`) counting more than generic ones (`Git`) |
| domain | 25% | how well your skills' career path (backend, data, mobile, ...) fits the job's |
| experience | 20% | depth in the job's path; with `--resume`, skills shown in work, projects and coursework |
| transferable | 15% | coverage of the generic requirements |

`--weights` changes the split, either by name (`technical=0.5,domain=0.3`, the rest keep their
defaults) or as four numbers (`40,25,20,15`). The weights and the skill-to-path tables were set by
hand, not tuned on data.

**LLM call budget.** A search is refused before it starts if it would make more than
`--max-llm-calls` calls (default 10), counting the resume, each description-only posting and
each re-check. This keeps a large or untrusted catalog from draining your Gemini quota. If the
quota still runs out during `--rerank`, you get the quick ranking instead of an error.

```
 1. Software Engineer Intern (Backend) — FinTech startup · Bangkok
    📊 91/100
    ✅ Python  ✅ SQL  ✅ REST API design  ✅ Git  ✅ FastAPI or Django  ✅ Docker  ❌ Cloud (AWS/GCP)
```

## Accuracy, performance and tests

None of these call the LLM, so they run without an API key.

```bash
python agent.py --eval        # accuracy against hand-labeled cases
python agent.py --bench       # search speed on synthetic catalogs of 1k / 10k / 50k jobs
python agent.py --self-test   # 22 unit, accuracy and performance tests (also: python -m unittest agent)
python -m unittest test_agent_extended   # 137 more, incl. a held-out label set (SKIP_PERF=1 skips the slow ones)
```

**Accuracy** (`--eval`) is measured at two levels and compared with an exact-name baseline
that uses the same weights but no aliases, implications or families:

| Level | Metric | Skills ranking | Career ranking | Exact-name baseline |
|---|---|---|---|---|
| Requirement (43 cases) | status accuracy (met/partial/missing) | 93.0% | | 16.3% |
| Ranking (9 candidates × 14 jobs) | Top-1 (best result is a strong fit) | 0.89 | 0.89 | 0.78 |
| | MRR (first strong fit) | 0.91 | 0.92 | 0.87 |
| | NDCG@5 | 0.95 | 0.97 | 0.91 |
| | Recall@5 (any fit) | 0.90 | 0.93 | 0.84 |

`test_agent_extended.py` also checks a held-out set written separately from these labels
(69 requirement cases, 9 candidates). There, NDCG@5 is 0.96 for skills ranking and 0.98 for career
ranking. The exact-name baseline already gets Top-1 1.00 on that set, so it is easy: it catches
regressions but can't tell the two rankers apart.

The three requirement misses are out-of-vocabulary on purpose (`Spring Boot` → Java,
`MongoDB` → NoSQL, `React` → React Native). The ranking miss is the sample-resume candidate:
it lists React, so the matcher ranks the frontend internship first, while a human reading the
resume sees "basic React" and ranks the backend role first. Skill names alone can't show depth.

The labels were written by hand for this project, alongside the matcher, so treat these numbers
as a regression check on a small set, not as a benchmark of real-world accuracy. To measure
accuracy on your own data, add candidates and labels to `EVAL_CANDIDATES` and `REQUIREMENT_CASES`
in `agent.py`.

**Performance** (`--bench`, one laptop, Python 3.10):

| Ranking | Jobs | Index build | Search p50 | Search p95 |
|---|---|---|---|---|
| skills | 1,000 | 15 ms | 4.5 ms | 6.9 ms |
| | 10,000 | 145 ms | 30 ms | 34 ms |
| | 50,000 | 438 ms | 111 ms | 134 ms |
| career | 1,000 | 48 ms | 9.1 ms | 10.8 ms |
| | 10,000 | 518 ms | 82 ms | 111 ms |
| | 50,000 | 2.4 s | 362 ms | 459 ms |

Search time grows linearly with the catalog: about 400k jobs/second for skills ranking and
130k for career ranking. Requirements are parsed once when the index is built, in linear time
even on hostile input, each skill verdict is cached per search, and the detailed per-requirement
output is built only for the jobs returned. `--self-test` fails if a 10k-job search takes over
250 ms at p95 or the index build takes over 2 s, for either ranking.

## Sample Output

_To be added from a real run._

## Ethical considerations

- **Not a hiring tool.** The match score is guidance for the applicant, not a
  judgment of their worth. Do not use it to screen or rank candidates.
- **Bias.** LLMs can undervalue non-traditional backgrounds, non-English resumes,
  or unconventional career paths. Treat the gap list as a starting point, not a verdict.
- **Hallucinated resources.** Learning resources are named, not linked, to avoid
  invented URLs; check that a course or book exists before relying on it.
- **Personal data.** Resumes contain PII (names, contact details, education history).
  The text is sent to Google's Gemini API; check its data-use terms (the free tier
  may use inputs to improve Google's products) before using real resumes, and
  do not commit resumes to the repository.
