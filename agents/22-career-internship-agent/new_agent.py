"""
AI Career & Internship Agent using LangGraph + Gemini.

A five-stage multi-agent pipeline that turns a resume and a job description
into a personalized, actionable preparation plan:

    Resume Analyzer ─┐
                     ├─> Matching Agent -> Gap Analysis Agent -> Recommendation Agent
    JD Analyzer ─────┘

The LLM judges evidence (is this requirement met, partially met, or missing?);
the match score and the "missing must-have = blocking" rule are computed in code,
so the score is reproducible from the assessments rather than invented.

Usage:
    python new_agent.py                                   # built-in sample resume + JD
    python new_agent.py --resume resume.pdf --job-desc job_posting.txt
    python new_agent.py --weeks 4 --json                  # 4-week plan, raw JSON output
"""

import argparse
import heapq
import json
import math
import os
import random
import re
import statistics
import sys
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Literal, NamedTuple, Optional, TypedDict
from unittest import mock

from dotenv import load_dotenv
from langchain_core.exceptions import (
    ModelAuthenticationError,
    ModelInvalidRequestError,
    ModelNotFoundError,
    ModelRateLimitError,
)
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, model_validator

load_dotenv()

DEFAULT_MODEL = "gemini-3.6-flash"

# Weights used to turn per-requirement assessments into a 0-100 score.
IMPORTANCE_WEIGHT = {"must-have": 2.0, "nice-to-have": 1.0}
STATUS_CREDIT = {"met": 1.0, "partial": 0.5, "missing": 0.0}


def get_llm() -> ChatGoogleGenerativeAI:
    # No temperature: Gemini 3.x models use fixed sampling defaults and ignore it.
    return ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL", DEFAULT_MODEL))


# Far above any resume or job posting, and well within Gemini's input limit.
MAX_INPUT_CHARS = 100_000


class InputTooLongError(ValueError):
    pass


def ask(schema: type[BaseModel], system: str, user: str) -> BaseModel:
    """Call Gemini and parse the reply straight into a Pydantic model.

    The client already retries transient errors (max_retries defaults to 6).
    """
    if len(user) > MAX_INPUT_CHARS:
        raise InputTooLongError(f"Input is {len(user):,} characters; the limit is {MAX_INPUT_CHARS:,}. "
                                "Pass the resume or job description alone, not a whole document.")
    llm = get_llm().with_structured_output(schema)
    return llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])


def norm(skill: str) -> str:
    """Normalize a skill name for comparison: 'REST APIs' -> 'restapis', 'C#' -> 'c#'."""
    return re.sub(r"[^a-z0-9+#]", "", skill.lower())


# ---------------------------------------------------------------------------
# Structured outputs, one per stage
# ---------------------------------------------------------------------------


class Project(BaseModel):
    name: str
    description: str
    skills_used: list[str] = Field(default_factory=list)


class ResumeProfile(BaseModel):
    """Output of the Resume Analyzer."""

    name: str
    education: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list, description="Short canonical skill names")
    projects: list[Project] = Field(default_factory=list)
    experience: list[str] = Field(default_factory=list)
    activities: list[str] = Field(default_factory=list, description="Clubs, hackathons, competitions")


class JobRequirements(BaseModel):
    """Output of the JD Analyzer."""

    title: str
    company: Optional[str] = None
    eligibility: list[str] = Field(default_factory=list, description="Non-skill criteria, e.g. enrollment, dates")
    must_have_skills: list[str] = Field(default_factory=list)
    nice_to_have_skills: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)


class RequirementAssessment(BaseModel):
    skill: str = Field(description="Exactly as given in the requirement list")
    importance: Literal["must-have", "nice-to-have"]
    status: Literal["met", "partial", "missing"]
    evidence: str = Field(description="Quote or paraphrase from the resume, or what is absent")


class EligibilityCheck(BaseModel):
    criterion: str = Field(description="Exactly as given in the eligibility list")
    status: Literal["met", "not-met", "unclear"]
    evidence: str = Field(description="What the resume says, or that it does not say")


class MatchAssessment(BaseModel):
    """What the Matching Agent asks the LLM for."""

    assessments: list[RequirementAssessment]
    eligibility: list[EligibilityCheck] = Field(default_factory=list)
    summary: str = Field(description="2-3 sentences on overall fit, without a numeric score")


class MatchResult(BaseModel):
    """Output of the Matching Agent (score computed in code, from skills only)."""

    score: int
    assessments: list[RequirementAssessment]
    eligibility: list[EligibilityCheck] = Field(default_factory=list)
    summary: str


class SkillGap(BaseModel):
    skill: str
    severity: Literal["blocking", "important", "nice-to-have"]
    evidence: str = Field(description="Where in the JD this requirement comes from")
    why_it_matters: str


class GapAnalysis(BaseModel):
    """Output of the Gap Analysis Agent."""

    gaps: list[SkillGap] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list, description="Strengths to lean on in the application")


class WeekPlan(BaseModel):
    week: int
    focus: str
    tasks: list[str]
    resources: list[str] = Field(default_factory=list, description="Named resources, no URLs")


class ProjectIdea(BaseModel):
    title: str
    description: str
    skills_covered: list[str]


class InterviewQuestion(BaseModel):
    question: str
    why_asked: str
    how_to_prepare: str


class Recommendation(BaseModel):
    """Output of the Recommendation Agent."""

    learning_plan: list[WeekPlan]
    project_ideas: list[ProjectIdea]
    resume_improvements: list[str]
    interview_questions: list[InterviewQuestion]


class CareerState(TypedDict, total=False):
    resume_text: str
    job_description: str
    weeks: int
    resume_profile: ResumeProfile
    job_requirements: JobRequirements
    match: MatchResult
    gap_analysis: GapAnalysis
    recommendation: Recommendation


# ---------------------------------------------------------------------------
# Agents (graph nodes)
# ---------------------------------------------------------------------------

RESUME_PROMPT = """You are a career advisor analyzing the resume of a student or early-career applicant.
Extract a structured profile.
- List skills as short canonical names (e.g. "Python", "PostgreSQL", "REST APIs", "Docker").
- Include skills demonstrated in projects, coursework, or activities, not only the skills section.
- Never add anything the resume does not support."""

JD_PROMPT = """You are a technical recruiter analyzing a job description.
Extract its requirements.
- must_have_skills: explicitly required. nice_to_have_skills: preferred, bonus, or "plus".
- Use short canonical skill names (e.g. "Python", "SQL", "REST API design", "Docker").
- Keep alternatives as ONE requirement: "FastAPI or Django", "Cloud (AWS or GCP)". Never split them.
- Put non-skill criteria (degree enrollment, availability, location) in eligibility, not in skills."""

MATCH_PROMPT = """You compare a candidate profile with a list of job requirements.
Assess EVERY requirement in the list, keeping its skill name and importance exactly as given.
- A requirement with alternatives ("X or Y") is met if the candidate has any one of them.
- met: clear evidence in the profile (skills, projects, experience).
- partial: related or shallow evidence (coursework only, an adjacent tool, a basic level).
- missing: no evidence.
Be strict: do not assume skills without evidence. Do not output a numeric score.
Also check EVERY eligibility criterion (enrollment, graduation date, availability, location), keeping its
text exactly as given: met (the resume confirms it), not-met (the resume contradicts it), or unclear
(the resume does not say). Never mark a criterion not-met just because the resume is silent."""

GAP_PROMPT = """You are a career coach. Given the requirements a candidate has not fully met, classify each gap:
- blocking: a must-have that is missing; likely to fail screening.
- important: a must-have that is only partial, or a nice-to-have that matters for this role's core work.
- nice-to-have: everything else.
Explain why each gap matters for THIS role, citing the job description.
Also list 3-5 genuine strengths the candidate should lean on."""

RECOMMEND_PROMPT = """You are a mentor helping a student prepare for a specific internship or job.
Create a personalized, realistic plan to close the gaps before applying and interviewing.
Honesty is non-negotiable: the candidate must never claim a skill, tool, or result they do not have.
- learning_plan: exactly {weeks} weeks, blocking gaps first, roughly 8-10 hours per week.
  Name concrete resources (official docs, well-known free courses), no URLs.
- project_ideas: 2-3 small projects, each finishable in a weekend or two, that close key gaps;
  extend the candidate's existing projects where possible.
- resume_improvements: specific edits that reframe what the candidate ALREADY did to be more relevant
  (quantify results, surface skills they used but undersell, reorder for this role). Anything that depends
  on finishing a project idea must start with "After completing <project title>:".
- interview_questions: 6-8 questions this employer is likely to ask, mixing technical questions on
  gap areas, deep-dives on the candidate's own projects, and behavioral questions. Questions about the
  candidate's projects must describe them as they are on the resume, not as they could be extended."""


def analyze_resume(state: CareerState) -> CareerState:
    """Resume Analyzer: extract a structured, skill-normalized profile."""
    profile = ask(ResumeProfile, RESUME_PROMPT, state["resume_text"])
    return {"resume_profile": profile}


def analyze_job(state: CareerState) -> CareerState:
    """JD Analyzer: split requirements into must-have vs nice-to-have."""
    requirements = ask(JobRequirements, JD_PROMPT, state["job_description"])
    return {"job_requirements": requirements}


def score_assessments(assessments: list[RequirementAssessment]) -> int:
    total = sum(IMPORTANCE_WEIGHT[a.importance] for a in assessments)
    if total == 0:
        return 0
    earned = sum(IMPORTANCE_WEIGHT[a.importance] * STATUS_CREDIT[a.status] for a in assessments)
    return round(100 * earned / total)


def dedupe_requirements(must_have: list[str], nice_to_have: list[str]) -> list[tuple[str, str]]:
    """One (skill, importance) per requirement, so no skill is scored twice; must-have wins."""
    requirements, seen = [], set()
    for skill, importance in [(s, "must-have") for s in must_have] + [(s, "nice-to-have") for s in nice_to_have]:
        key = canon(skill)  # so "Postgres" and "PostgreSQL" count once
        if key and key not in seen:
            seen.add(key)
            requirements.append((skill, importance))
    return requirements


def match_skills(state: CareerState) -> CareerState:
    """Matching Agent: the LLM judges evidence per requirement; the score is computed here."""
    jd = state["job_requirements"]
    requirements = dedupe_requirements(jd.must_have_skills, jd.nice_to_have_skills)
    requirement_list = "\n".join(f"- {skill} ({importance})" for skill, importance in requirements)

    eligibility_list = "\n".join(f"- {c}" for c in jd.eligibility) or "(none)"
    result = ask(
        MatchAssessment,
        MATCH_PROMPT,
        f"Candidate profile:\n{state['resume_profile'].model_dump_json(indent=2)}\n\n"
        f"Requirements:\n{requirement_list}\n\nEligibility criteria:\n{eligibility_list}",
    )

    # Keep exactly one assessment per JD requirement: drop extras the LLM invented,
    # and count any requirement it skipped as missing.
    by_skill = {canon(a.skill): a for a in result.assessments}
    assessments = []
    for skill, importance in requirements:
        found = by_skill.get(canon(skill))
        if found:
            assessments.append(found.model_copy(update={"skill": skill, "importance": importance}))
        else:
            assessments.append(RequirementAssessment(
                skill=skill, importance=importance, status="missing", evidence="Not assessed; treated as missing."
            ))

    # Same for eligibility; a criterion the LLM skipped is unclear, not failed.
    by_criterion = {norm(c.criterion): c for c in result.eligibility}
    eligibility = [
        by_criterion[norm(c)].model_copy(update={"criterion": c}) if norm(c) in by_criterion
        else EligibilityCheck(criterion=c, status="unclear", evidence="Not assessed.")
        for c in jd.eligibility
    ]

    match = MatchResult(score=score_assessments(assessments), assessments=assessments,
                        eligibility=eligibility, summary=result.summary)
    return {"match": match}


def default_severity(item: RequirementAssessment) -> str:
    if item.importance == "must-have":
        return "blocking" if item.status == "missing" else "important"
    return "nice-to-have"


def reconcile_gaps(open_items: list[RequirementAssessment], returned: list[SkillGap],
                   eligibility: list[EligibilityCheck]) -> list[SkillGap]:
    """Exactly one gap per unmet requirement, ordered by severity.

    Gaps the LLM invented or repeated are dropped, requirements it skipped are added back,
    a missing must-have is always blocking, and so is an eligibility criterion that is not met.
    """
    open_by_key = {canon(item.skill): item for item in open_items}
    gaps: dict[str, SkillGap] = {}
    for gap in returned:
        key = canon(gap.skill)
        item = open_by_key.get(key)
        if item is None or key in gaps:
            continue
        severity = "blocking" if default_severity(item) == "blocking" else gap.severity
        gaps[key] = gap.model_copy(update={"skill": item.skill, "severity": severity})
    for key, item in open_by_key.items():
        if key not in gaps:
            gaps[key] = SkillGap(skill=item.skill, severity=default_severity(item),
                                 evidence=f"Listed as {item.importance} in the job description",
                                 why_it_matters=f"The candidate's evidence for it is {item.status}.")
    for check in eligibility:
        if check.status == "not-met":
            gaps[f"eligibility:{norm(check.criterion)}"] = SkillGap(
                skill=check.criterion, severity="blocking", evidence="Eligibility criterion in the job description",
                why_it_matters=f"Applications that miss eligibility are usually screened out. {check.evidence}")
    order = {"blocking": 0, "important": 1, "nice-to-have": 2}
    return sorted(gaps.values(), key=lambda g: order[g.severity])


def analyze_gaps(state: CareerState) -> CareerState:
    """Gap Analysis Agent: rank missing/partial skills by severity with JD evidence."""
    match = state["match"]
    open_items = [a for a in match.assessments if a.status != "met"]
    if not open_items:
        gaps = reconcile_gaps([], [], match.eligibility)
        return {"gap_analysis": GapAnalysis(gaps=gaps, strengths=["Meets every listed skill requirement."])}

    jd = state["job_requirements"]
    analysis = ask(
        GapAnalysis,
        GAP_PROMPT,
        f"Role: {jd.title}\nResponsibilities:\n" + "\n".join(f"- {r}" for r in jd.responsibilities)
        + "\n\nUnmet requirements:\n"
        + json.dumps([a.model_dump() for a in open_items], indent=2, ensure_ascii=False)
        + f"\n\nCandidate profile:\n{state['resume_profile'].model_dump_json(indent=2)}",
    )
    gaps = reconcile_gaps(open_items, analysis.gaps, match.eligibility)
    return {"gap_analysis": analysis.model_copy(update={"gaps": gaps})}


def fit_plan(plan: list[WeekPlan], weeks: int) -> list[WeekPlan]:
    """Hold the learning plan to the requested length, numbered 1..weeks."""
    plan = sorted(plan, key=lambda w: w.week)
    if len(plan) != weeks:
        print(f"⚠️  The model returned a {len(plan)}-week plan for a {weeks}-week request; "
              + ("keeping the first weeks." if len(plan) > weeks else "showing what it returned."), file=sys.stderr)
    return [w.model_copy(update={"week": i}) for i, w in enumerate(plan[:weeks], 1)]


def recommend(state: CareerState) -> CareerState:
    """Recommendation Agent: learning plan, project ideas, resume edits, interview prep."""
    weeks = state.get("weeks", 3)
    jd = state["job_requirements"]
    recommendation = ask(
        Recommendation,
        RECOMMEND_PROMPT.format(weeks=weeks),
        f"Target role:\n{jd.model_dump_json(indent=2)}\n\n"
        f"Candidate profile:\n{state['resume_profile'].model_dump_json(indent=2)}\n\n"
        f"Match score: {state['match'].score}/100\n{state['match'].summary}\n\n"
        f"Gap analysis:\n{state['gap_analysis'].model_dump_json(indent=2)}",
    )
    plan = fit_plan(recommendation.learning_plan, weeks)
    return {"recommendation": recommendation.model_copy(update={"learning_plan": plan})}


def build_graph():
    graph = StateGraph(CareerState)
    graph.add_node("resume_analyzer", analyze_resume)
    graph.add_node("jd_analyzer", analyze_job)
    graph.add_node("matcher", match_skills)
    graph.add_node("gap_analyzer", analyze_gaps)
    graph.add_node("recommender", recommend)

    # Resume and JD are analyzed in parallel, then joined at the matcher.
    graph.add_edge(START, "resume_analyzer")
    graph.add_edge(START, "jd_analyzer")
    graph.add_edge(["resume_analyzer", "jd_analyzer"], "matcher")
    graph.add_edge("matcher", "gap_analyzer")
    graph.add_edge("gap_analyzer", "recommender")
    graph.add_edge("recommender", END)
    return graph.compile()


# ---------------------------------------------------------------------------
# Job search: rank a whole catalog of postings against one set of skills
# ---------------------------------------------------------------------------
#
# Running the Matching Agent on every posting would cost one LLM call per job, so
# job search ranks the catalog with a deterministic matcher on the same
# met / partial / missing scale and weights, then can re-check the top few with
# the Matching Agent (--rerank). The matcher knows three kinds of relationship:
#   ALIASES  different spellings of one skill       Postgres == PostgreSQL
#   IMPLIES  having A demonstrates B                PostgreSQL -> SQL (met)
#   FAMILIES interchangeable-ish tools              MySQL ~ PostgreSQL (partial)

# Keys and values are norm()-ed names.
ALIASES = {
    "postgres": "postgresql", "psql": "postgresql", "mssql": "sqlserver", "mongo": "mongodb",
    "js": "javascript", "es6": "javascript", "ts": "typescript", "reactjs": "react", "vuejs": "vue",
    "angularjs": "angular", "node": "nodejs", "expressjs": "express", "html5": "html", "css3": "css",
    "golang": "go", "csharp": "c#", "cpp": "c++", "cloudcomputing": "cloud",
    "sklearn": "scikitlearn", "ml": "machinelearning", "dl": "deeplearning", "naturallanguageprocessing": "nlp",
    "rest": "restapi", "restapis": "restapi", "restapidesign": "restapi", "restful": "restapi",
    "restfulapi": "restapi", "restfulapis": "restapi",
    "amazonwebservices": "aws", "googlecloud": "gcp", "googlecloudplatform": "gcp", "microsoftazure": "azure",
    "k8s": "kubernetes", "ci": "cicd", "continuousintegration": "cicd", "shell": "bash", "shellscripting": "bash",
    "unix": "linux", "msexcel": "excel", "microsoftexcel": "excel", "stats": "statistics",
    "testing": "softwaretesting", "unittesting": "softwaretesting", "testautomation": "softwaretesting",
    "automatedtesting": "softwaretesting", "qa": "softwaretesting",
    "android": "androidsdk", "ios": "iossdk", "xcode": "iossdk", "compose": "jetpackcompose",
    "arduino": "microcontrollers", "stm32": "microcontrollers", "esp32": "microcontrollers",
    "microcontroller": "microcontrollers", "freertos": "rtos", "pcb": "pcbdesign",
    "uiux": "uidesign", "uiuxdesign": "uidesign", "uxuidesign": "uidesign", "uxresearch": "userresearch",
    "networks": "networking", "computernetworking": "networking", "tcpip": "networking",
    "pentest": "penetrationtesting", "pentesting": "penetrationtesting",
    "apachespark": "spark", "pyspark": "spark", "apacheairflow": "airflow", "unity3d": "unity",
}

IMPLIES = {
    **{db: ["sql"] for db in ("postgresql", "mysql", "sqlite", "sqlserver", "oracle", "mariadb")},
    "django": ["python"], "flask": ["python"], "fastapi": ["python", "restapi"],
    "pandas": ["python"], "numpy": ["python"], "scikitlearn": ["python", "machinelearning"],
    **{dl: ["python", "deeplearning"] for dl in ("pytorch", "tensorflow", "keras")},
    "deeplearning": ["machinelearning"], "nlp": ["machinelearning"],
    "react": ["javascript"], "vue": ["javascript"], "svelte": ["javascript"], "angular": ["typescript"],
    "nextjs": ["react"], "typescript": ["javascript"], "nodejs": ["javascript"], "express": ["nodejs"],
    "aws": ["cloud"], "gcp": ["cloud"], "azure": ["cloud"],
    "jetpackcompose": ["androidsdk"], "swiftui": ["swift", "iossdk"],
    "selenium": ["softwaretesting"], "playwright": ["softwaretesting"], "pytest": ["softwaretesting", "python"],
    "junit": ["softwaretesting", "java"], "cypress": ["softwaretesting", "javascript"],
    "githubactions": ["cicd"], "jenkins": ["cicd"], "gitlabci": ["cicd"],
    "airflow": ["etl"], "spark": ["etl"], "dbt": ["etl", "sql"], "rtos": ["microcontrollers"],
    "github": ["git"], "gitlab": ["git"], "bitbucket": ["git"],
    "htmlcss": ["html", "css"],
}

# Requirements that need every part: HTML alone is only partial for "HTML/CSS".
COMPOUNDS = {"htmlcss": ("html", "css")}

# Specializations are directional. Having the specific skill shows the broader one (met), but having only
# the broader skill is partial credit for the specific one: Docker alone is not Docker Compose, whereas
# Docker Compose is Docker. Each entry is merged into IMPLIES below.
SPECIALIZES = {
    "dockercompose": ["docker"],
    "springboot": ["spring"],
    "tsql": ["sql"],
    "plsql": ["sql"],
}
for _specific, _broader in SPECIALIZES.items():
    IMPLIES[_specific] = list(dict.fromkeys([*IMPLIES.get(_specific, []), *_broader]))

# Marks an umbrella term from "Cloud (AWS/GCP)": general cloud experience is only partial
# credit when the posting names specific providers.
UMBRELLA = "~"

FAMILIES = [
    {"postgresql", "mysql", "sqlite", "sqlserver", "oracle", "mariadb", "tsql", "plsql"},
    {"mongodb", "firebase", "dynamodb"},
    {"django", "fastapi", "flask"},
    {"react", "vue", "angular", "svelte"},
    {"aws", "gcp", "azure"},
    {"pytorch", "tensorflow", "keras", "jax"},
    {"java", "kotlin", "scala"},
    {"c", "c++"},
    {"docker", "kubernetes"},
    {"tableau", "powerbi", "looker"},
    {"selenium", "cypress", "playwright"},
    {"unity", "unreal", "godot"},
]
SIBLINGS: dict[str, set[str]] = {}
for _family in FAMILIES:
    for _skill in _family:
        SIBLINGS.setdefault(_skill, set()).update(_family - {_skill})

KNOWN_SKILLS = (set(ALIASES.values()) | set(IMPLIES) | set(SIBLINGS)
                | {s for implied in IMPLIES.values() for s in implied})

# Filler words recruiters wrap around skill names: "Strong Python skills" -> "python".
FILLER = {"strong", "solid", "basic", "good", "experience", "experienced", "with", "in", "of", "familiarity",
          "familiar", "knowledge", "proficiency", "proficient", "understanding", "skills", "skill",
          "development", "programming", "language", "databases", "database", "framework", "scripting"}

STATUS_RANK = {"met": 2, "partial": 1, "missing": 0}


# Safeguards for untrusted text (user-supplied --jobs files, LLM output). Real skill names and requirements
# are far shorter than these limits, so ordinary input is never affected.
MAX_SKILL_CHARS = 300   # a longer skill or requirement string is cut off here
MAX_ALTERNATIVES = 32   # at most this many "X or Y or ..." options are kept per requirement
CACHE_SIZE = 50_000     # bounded, so a hostile catalog cannot grow the caches without limit

# A standalone version number after a skill name: "3.11", "17", "v2", "3.x", "8+".
VERSION_TOKEN = re.compile(r"v?\d+(?:\.\d+)*(?:\.x)?\+?")


@lru_cache(maxsize=CACHE_SIZE)
def canon(skill: str) -> str:
    """Canonical skill key: 'Strong Python skills' -> 'python', 'Postgres' -> 'postgresql'.

    Versions are ignored ('Python 3.11', 'Java17' -> 'python', 'java'): the matcher judges which skill
    a person has, not which release, and a version never makes a different skill.
    """
    skill = skill[:MAX_SKILL_CHARS]
    words = [w for w in skill.lower().split() if w not in FILLER]
    unversioned = [w for w in words if not VERSION_TOKEN.fullmatch(w)]  # a lone number is kept as it is
    key = norm(" ".join(unversioned or words)) or norm(skill)
    key = ALIASES.get(key, key)
    if key not in KNOWN_SKILLS:
        # A version glued to a name the tables know: 'python3', 'java17', 'vue3'. Names that merely end in
        # a digit ('s3', 'ec2') stay as they are, because what is left of them is not a known skill.
        base = key.rstrip("0123456789")
        if base and base != key:
            base = ALIASES.get(base, base)
            if base in KNOWN_SKILLS:
                return base
    return key


def _split_parenthetical(text: str) -> Optional[tuple[str, str]]:
    """('Cloud ', 'AWS/GCP') for 'Cloud (AWS/GCP)', or None when text has no such shape.

    Accepts exactly what the regex r"(.*?)\\((.+)\\)\\s*" accepted, in one pass. That regex backtracked:
    text such as "((((((x" took quadratic time (20k characters: 1.1 s).
    """
    core = text.rstrip()
    if "\n" in core or not core.endswith(")"):
        return None
    opening = core.find("(")
    if opening < 0 or opening >= len(core) - 2:  # no "(", or nothing between it and the final ")"
        return None
    return core[:opening], core[opening + 1:-1]


def _split_options(body: str) -> list[str]:
    """Split 'FastAPI or Django', 'AWS/GCP', 'A, B' and 'A | B' into their options."""
    options = []
    for chunk in re.split(r"[/,|]", body):
        # Collapse whitespace first: matching r"\s+or\s+" against a long run of spaces is quadratic.
        options.extend(re.split(r" or ", re.sub(r"\s+", " ", chunk), flags=re.IGNORECASE))
    return options


@lru_cache(maxsize=CACHE_SIZE)
def alternatives(requirement: str) -> tuple[str, ...]:
    """Canonical skills that each satisfy a requirement.

    'FastAPI or Django' -> (fastapi, django); 'Cloud (AWS/GCP)' -> (aws, gcp, ~cloud), where
    ~cloud gives partial credit only; 'CI/CD' and 'HTML/CSS' stay one skill because they are known names.
    Runs in linear time; text beyond MAX_SKILL_CHARS and options beyond MAX_ALTERNATIVES are ignored.
    """
    requirement = requirement[:MAX_SKILL_CHARS]
    whole = canon(requirement)
    if whole in KNOWN_SKILLS:
        return (whole,)
    umbrella = []
    split = _split_parenthetical(requirement)
    if split:
        outer, body = split
        outer = canon(outer)
        if outer in KNOWN_SKILLS:
            umbrella.append(UMBRELLA + outer)
    else:
        body = requirement
    options = list(dict.fromkeys(p for p in map(canon, _split_options(body)) if p))
    return tuple(options[:MAX_ALTERNATIVES] + umbrella) or (whole,)


class Verdict(NamedTuple):
    """A requirement judged against one candidate: the status, how it was matched, and the evidence.

    kind says WHY the status holds, so equivalent and merely related skills are never confused:
      exact       the candidate lists the skill under its canonical name ("Python" for Python)
      equivalent  listed under another spelling or version ("Postgres", "Python 3.11", "ReactJS")
      implied     shown by a more specific skill that is listed ("PostgreSQL" shows SQL; "Docker Compose" shows Docker)
      related     only a related skill: a sibling tool, a broader skill, an umbrella term or half of a compound
      missing     nothing relevant is listed
    exact, equivalent and implied are "met"; related is "partial".
    """

    status: str
    kind: str
    evidence: str


class SkillMatcher:
    """Judges requirements against one candidate's skills, caching each verdict."""

    def __init__(self, skills: list[str]):
        # Map every skill the candidate demonstrates to the listed skill that shows it.
        # Listed skills go in first, so an inferred skill never hides a listed one.
        self.has: dict[str, str] = {}
        self._spelling: dict[str, str] = {}  # listed skill -> "exact" (canonical name) or "equivalent"
        for s in skills:
            key = canon(s)
            if key:
                self.has.setdefault(key, s.strip())
                if norm(s) == key:
                    self._spelling[key] = "exact"
                else:
                    self._spelling.setdefault(key, "equivalent")
        self.listed = set(self.has)
        stack = list(self.has.items())
        while stack:
            skill, source = stack.pop()
            for implied in IMPLIES.get(skill, ()):
                if implied not in self.has:
                    self.has[implied] = source
                    stack.append((implied, source))
        for compound, parts in COMPOUNDS.items():
            if compound not in self.has and all(p in self.has for p in parts):
                self.has[compound] = " + ".join(dict.fromkeys(self.has[p] for p in parts))
        self._cache: dict[tuple[str, ...], tuple[str, str]] = {}
        self._kinds: dict[tuple[str, ...], str] = {}

    def _judge_one(self, skill: str) -> Verdict:
        if skill.startswith(UMBRELLA):
            general = skill[len(UMBRELLA):]
            if general in self.has:
                return Verdict("partial", "related", f"Lists {self.has[general]}, but not one of the named options")
            return Verdict("missing", "missing", "No matching or related skill listed")
        if skill in COMPOUNDS and skill not in self.has:
            have = [self.has[p] for p in COMPOUNDS[skill] if p in self.has]
            if have:
                return Verdict("partial", "related", f"Lists only {', '.join(have)}")
        if skill in self.has:
            if skill in self.listed:
                return Verdict("met", self._spelling[skill], f"Lists {self.has[skill]}")
            return Verdict("met", "implied", f"Lists {self.has[skill]}, which implies it")
        # Directional: a broader skill is only partial credit for the specific one it does not show.
        for broader in SPECIALIZES.get(skill, ()):
            if broader in self.has:
                return Verdict("partial", "related", f"Lists {self.has[broader]}, a broader skill than the one required")
        for sibling in sorted(SIBLINGS.get(skill, ())):
            if sibling in self.has:
                return Verdict("partial", "related", f"Related skill: {self.has[sibling]}")
        return Verdict("missing", "missing", "No matching or related skill listed")

    def judge(self, alts: tuple[str, ...]) -> tuple[str, str]:
        """(status, evidence) for a requirement; any one alternative is enough."""
        cached = self._cache.get(alts)
        if cached is None:
            best = max((self._judge_one(a) for a in alts), key=lambda v: STATUS_RANK[v.status])
            self._kinds[alts] = best.kind
            cached = self._cache[alts] = (best.status, best.evidence)
        return cached

    def judge_detailed(self, alts: tuple[str, ...]) -> Verdict:
        """Like judge(), plus the kind of match (exact / equivalent / implied / related / missing)."""
        status, evidence = self.judge(alts)
        return Verdict(status, self._kinds[alts], evidence)


class JobPosting(BaseModel):
    id: str
    title: str
    company: Optional[str] = None
    location: Optional[str] = None
    must_have_skills: list[str] = Field(default_factory=list)
    nice_to_have_skills: list[str] = Field(default_factory=list)
    family: Optional[str] = Field(default=None, description="Career path(s), e.g. 'backend' or 'frontend,backend'; "
                                                            "inferred from the skills and title when omitted")


class JobMatch(BaseModel):
    job: JobPosting
    score: int = Field(description="Quick skill-match score from the deterministic matcher")
    missing_must_haves: list[str]
    assessments: list[RequirementAssessment]
    llm_score: Optional[int] = Field(default=None, description="Matching Agent score, when re-ranked")
    llm_summary: Optional[str] = None


class JobIndex:
    """A job catalog with requirements parsed once, so each search is a cheap scan."""

    def __init__(self, jobs: list[JobPosting]):
        self.jobs = jobs
        self._requirements = [
            [(skill, importance, alternatives(skill))
             for skill, importance in dedupe_requirements(job.must_have_skills, job.nice_to_have_skills)]
            for job in jobs
        ]
        # Scoring view of the same data: (total weight, [(alternatives, weight, is_must_have)]).
        self._scoring = [
            (sum(IMPORTANCE_WEIGHT[imp] for _, imp, _ in reqs),
             [(alts, IMPORTANCE_WEIGHT[imp], imp == "must-have") for _, imp, alts in reqs])
            for reqs in self._requirements
        ]

    def rank(self, skills: list[str], top: Optional[int] = 5) -> list[JobMatch]:
        """Best-fitting jobs first; ties go to fewer missing must-haves, then catalog order."""
        matcher = SkillMatcher(skills)
        keys = []
        for i, (total, requirements) in enumerate(self._scoring):
            earned = 0.0
            missing_must = 0
            for alts, weight, is_must in requirements:
                status = matcher.judge(alts)[0]
                earned += weight * STATUS_CREDIT[status]
                missing_must += is_must and status == "missing"
            keys.append((-(round(100 * earned / total) if total else 0), missing_must, i))
        best = heapq.nsmallest(top, keys) if top is not None else sorted(keys)
        # Pydantic objects are built only for the jobs returned: on large catalogs, building
        # them for every requirement of every job would dominate the search time.
        return [self._explain(i, matcher) for _, _, i in best]

    def _explain(self, i: int, matcher: SkillMatcher) -> JobMatch:
        assessments = []
        for skill, importance, alts in self._requirements[i]:
            status, evidence = matcher.judge(alts)
            assessments.append(RequirementAssessment(skill=skill, importance=importance, status=status,
                                                     evidence=evidence))
        return JobMatch(
            job=self.jobs[i],
            score=score_assessments(assessments),
            missing_must_haves=[a.skill for a in assessments if a.importance == "must-have" and a.status == "missing"],
            assessments=assessments,
        )


# ---------------------------------------------------------------------------
# Career-aware ranking (opt-in): find_jobs(..., ranking="career")
# ---------------------------------------------------------------------------
#
# JobIndex.rank() gives every requirement the same standing, so a generic skill (Git, Python, SQL) earns as
# much as a specialised one and unrelated roles climb the list on shared basics alone. The career ranker adds
# the missing structure, as four components in [0, 1] combined with configurable weights:
#   technical     skill coverage, each requirement weighted by how SPECIFIC its skill is to one career path
#   domain        how close the job's career path is to the candidate's (families + DOMAIN_AFFINITY)
#   experience    skills shown IN USE (projects, experience), or depth in the job's path when only a list is known
#   transferable  coverage of the GENERIC part of the requirements, so shared basics still count, but capped
# It is plain data and arithmetic: no LLM, no randomness, and ties break the same way every time.
# The tables and weights below are initial, hand-set values: tune them on your own labelled data.

DOMAINS = ("backend", "frontend", "mobile", "data", "ml", "devops", "qa", "embedded", "design", "security", "game")
_DOMAIN_INDEX = {d: i for i, d in enumerate(DOMAINS)}


def _tag(domains: str, *skills: str) -> dict[str, tuple[str, ...]]:
    return {skill: tuple(domains.split()) for skill in skills}


# Which career paths a skill is evidence for (its weight is split evenly). A skill that serves many paths is
# "generic"; one that serves a single path is "specific". Keys are canon() names.
SKILL_DOMAINS: dict[str, tuple[str, ...]] = {
    **_tag("backend", "restapi", "django", "flask", "fastapi", "nodejs", "express", "mongodb", "dynamodb",
           "spring", "springboot"),
    **_tag("frontend", "html", "css", "htmlcss", "react", "vue", "angular", "svelte", "nextjs"),
    **_tag("frontend backend", "javascript", "typescript"),
    **_tag("backend data", "sql", "postgresql", "mysql", "sqlite", "sqlserver", "oracle", "mariadb", "tsql", "plsql"),
    **_tag("backend mobile", "firebase", "kotlin", "java"),
    **_tag("backend data ml devops qa", "python"),
    **_tag("game backend", "c#"),
    **_tag("embedded", "c", "microcontrollers", "rtos", "pcbdesign"),
    **_tag("embedded game", "c++"),
    **_tag("mobile", "swift", "swiftui", "iossdk", "androidsdk", "jetpackcompose"),
    **_tag("devops backend", "docker", "dockercompose"),
    **_tag("devops", "kubernetes", "terraform"),
    **_tag("devops backend data", "aws", "gcp", "azure", "cloud"),
    **_tag("devops security backend embedded", "linux"),
    **_tag("devops security", "bash"),
    **_tag("devops qa", "cicd", "githubactions", "jenkins", "gitlabci"),
    **_tag("qa", "softwaretesting", "selenium", "cypress", "playwright", "junit", "pytest"),
    **_tag("data", "pandas", "tableau", "powerbi", "looker", "excel", "etl", "spark", "airflow", "dbt"),
    **_tag("data ml", "numpy", "statistics", "jupyter", "scikitlearn"),
    **_tag("ml", "machinelearning", "deeplearning", "nlp", "pytorch", "tensorflow", "keras", "jax"),
    **_tag("design", "figma", "uidesign", "userresearch", "prototyping"),
    **_tag("security devops", "networking"),
    **_tag("security", "wireshark", "penetrationtesting"),
    **_tag("game", "unity", "unreal", "godot", "blender"),
    **{skill: DOMAINS for skill in ("git", "github", "gitlab", "bitbucket")},  # every path uses version control
}
UNKNOWN_SPECIFICITY = 0.7  # a tool the table does not know is more likely specific than generic
SPECIFICITY_FLOOR = 0.1    # so a job made only of generic requirements still has something to score

# Career families: paths in the same family are related by default; DOMAIN_AFFINITY overrides pairs explicitly
# (symmetric). 1.0 would mean the same path and 0.0 unrelated.
CAREER_FAMILIES = {
    "software": ("backend", "frontend", "mobile", "qa"),
    "data-ai": ("data", "ml"),
    "infrastructure": ("devops", "security"),
    "design": ("design", "frontend"),
    "systems": ("embedded", "game"),
}
FAMILY_AFFINITY = 0.35
DOMAIN_AFFINITY = {
    ("backend", "frontend"): 0.5, ("backend", "devops"): 0.45, ("backend", "data"): 0.45, ("backend", "qa"): 0.45,
    ("data", "ml"): 0.6, ("frontend", "design"): 0.5, ("frontend", "mobile"): 0.4, ("devops", "security"): 0.4,
    ("devops", "qa"): 0.35, ("ml", "backend"): 0.25, ("data", "devops"): 0.25, ("embedded", "security"): 0.25,
    ("embedded", "devops"): 0.25,
}

# Words in a job title that name a career path (lowercase tokens).
TITLE_HINTS = {
    "backend": ("backend",), "back": ("backend",), "frontend": ("frontend",), "front": ("frontend",),
    "stack": ("frontend", "backend"), "fullstack": ("frontend", "backend"),
    "android": ("mobile",), "ios": ("mobile",), "mobile": ("mobile",),
    "data": ("data",), "analyst": ("data",), "analytics": ("data",), "scientist": ("data", "ml"),
    "machine": ("ml",), "learning": ("ml",), "ml": ("ml",), "ai": ("ml",),
    "devops": ("devops",), "cloud": ("devops",), "sre": ("devops",), "infrastructure": ("devops",),
    "qa": ("qa",), "test": ("qa",), "testing": ("qa",), "embedded": ("embedded",), "firmware": ("embedded",),
    "ux": ("design",), "ui": ("design",), "design": ("design",), "designer": ("design",),
    "security": ("security",), "cybersecurity": ("security",), "game": ("game",), "games": ("game",),
}
TITLE_WEIGHT = 3.0     # a title naming a path counts like this much requirement weight
DEPTH_SATURATION = 3   # specific skills in one path that count as full depth there
NEUTRAL_DOMAIN = 0.5   # domain score when there is no evidence either way

# Words that name a skill but are also ordinary English: not read as skills inside free resume text.
AMBIGUOUS_IN_TEXT = {"rest", "spring", "express", "swift", "unity", "compose", "shell", "testing", "networks",
                     "stats", "node", "cloud", "go"}
MAX_TEXT_CHARS = 5_000
_WORD = re.compile(r"[A-Za-z0-9+#.]+")


@lru_cache(maxsize=CACHE_SIZE)
def skill_specificity(skill: str) -> float:
    """1.0 for a skill that points at one career path, falling to 0.0 for one every path uses (Git)."""
    skill = skill[len(UMBRELLA):] if skill.startswith(UMBRELLA) else skill
    domains = SKILL_DOMAINS.get(skill)
    if not domains:
        return UNKNOWN_SPECIFICITY
    return 1 - math.log(len(domains)) / math.log(len(DOMAINS))


def domain_affinity_matrix() -> list[list[float]]:
    """DOMAINS x DOMAINS: 1.0 on the diagonal, family and explicit affinities elsewhere."""
    n = len(DOMAINS)
    matrix = [[1.0 if a == b else 0.0 for b in range(n)] for a in range(n)]
    for members in CAREER_FAMILIES.values():
        for a in members:
            for b in members:
                if a != b:
                    i, j = _DOMAIN_INDEX[a], _DOMAIN_INDEX[b]
                    matrix[i][j] = max(matrix[i][j], FAMILY_AFFINITY)
    for (a, b), value in DOMAIN_AFFINITY.items():
        matrix[_DOMAIN_INDEX[a]][_DOMAIN_INDEX[b]] = matrix[_DOMAIN_INDEX[b]][_DOMAIN_INDEX[a]] = value
    return matrix


class RankingWeights(BaseModel):
    """Relative importance of the four career-fit components. They are normalised to sum to 1 when used,
    so 40/25/20/15 and 0.40/0.25/0.20/0.15 are the same. These are initial values, not tuned optima."""

    technical: float = Field(0.40, ge=0, allow_inf_nan=False)
    domain: float = Field(0.25, ge=0, allow_inf_nan=False)
    experience: float = Field(0.20, ge=0, allow_inf_nan=False)
    transferable: float = Field(0.15, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _some_weight_is_positive(self):
        if self.technical + self.domain + self.experience + self.transferable <= 0:
            raise ValueError("at least one weight must be positive")
        return self

    def normalized(self) -> "RankingWeights":
        total = self.technical + self.domain + self.experience + self.transferable
        return RankingWeights(technical=self.technical / total, domain=self.domain / total,
                              experience=self.experience / total, transferable=self.transferable / total)

    @classmethod
    def parse(cls, text: str) -> "RankingWeights":
        """'technical=0.5,domain=0.2' (the rest keep their defaults) or four numbers in the order
        technical, domain, experience, transferable: '40,25,20,15'."""
        parts = [p.strip() for p in text.split(",") if p.strip()]
        try:
            if all("=" not in p for p in parts):
                if len(parts) != 4:
                    raise ValueError("give four numbers (technical, domain, experience, transferable) or name=value pairs")
                return cls(**dict(zip(cls.model_fields, map(float, parts))))
            values = {}
            for part in parts:
                name, _, value = part.partition("=")
                if name.strip() not in cls.model_fields:
                    raise ValueError(f"unknown weight '{name.strip()}'; use technical, domain, experience, transferable")
                values[name.strip()] = float(value)
            return cls(**values)
        except ValueError as e:
            raise ValueError(f"bad weights '{text}': {e}") from None


DEFAULT_RANKING_WEIGHTS = RankingWeights()
RANKING_MODES = ("skills", "career")


class CareerBreakdown(BaseModel):
    """Why a job got its career-fit score. Components are in [0, 1]; total is their weighted sum."""

    technical: float
    domain: float
    experience: float
    transferable: float
    total: float
    skill_score: int = Field(description="The plain skill-coverage score JobIndex.rank() would give")
    candidate_paths: list[str]
    job_paths: list[str]


class CareerMatch(JobMatch):
    """A JobMatch ranked by career fit: `score` is the career-fit total (0-100); see `career` for the parts."""

    career: CareerBreakdown


def _skills_in_text(text: str) -> list[str]:
    """Skills named in free text (project descriptions, experience), found as 1-3 word phrases."""
    words = _WORD.findall(text[:MAX_TEXT_CHARS])
    found = []
    for size in (3, 2, 1):
        for i in range(len(words) - size + 1):
            gram = " ".join(words[i:i + size])
            key = canon(gram)
            if len(key) > 2 and key in KNOWN_SKILLS and not (size == 1 and gram.lower().strip(".") in AMBIGUOUS_IN_TEXT):
                found.append(gram)
    return found


EVIDENCE_WEIGHTS = {
    "work_experience": 1.00,
    "project": 0.85,
    "coursework": 0.65,
    "activity": 0.50,
}


def applied_skill_evidence(profile: ResumeProfile) -> dict[str, tuple[str, float]]:
    """Map canonical skill to its strongest evidence source; repeated mentions do not add credit."""
    evidence: dict[str, tuple[str, float]] = {}

    def add(skills: list[str], source: str) -> None:
        weight = EVIDENCE_WEIGHTS[source]
        for skill in skills:
            key = canon(skill)
            if key and (key not in evidence or weight > evidence[key][1]):
                evidence[key] = (source, weight)

    for project in profile.projects:
        add(project.skills_used, "project")
        add(_skills_in_text(project.description), "project")
    for item in profile.experience:
        add(_skills_in_text(item), "work_experience")
    for item in profile.education:
        add(_skills_in_text(item), "coursework")
    for item in profile.activities:
        add(_skills_in_text(item), "activity")
    return evidence


def applied_skills(profile: ResumeProfile) -> list[str]:
    """Compatibility helper: skills with evidence beyond the skills-section list."""
    return list(applied_skill_evidence(profile))


class _JobFeatures(NamedTuple):
    rows: tuple          # (alternatives, technical weight, transferable weight, is_must_have) per requirement
    technical_total: float
    transferable_total: float
    domains: tuple       # the job's share in each of DOMAINS (sums to 1, or all 0 when unknown)
    expanded: tuple      # domains spread through the affinity matrix, so domain alignment is one dot product


class CareerRanker:
    """Ranks a catalog by career fit. Requirements and job features are prepared once; each search is a scan."""

    def __init__(self, jobs: list[JobPosting], weights: Optional[RankingWeights] = None):
        self.weights = (weights or DEFAULT_RANKING_WEIGHTS).normalized()
        self._index = JobIndex(jobs)
        affinity = domain_affinity_matrix()
        self._jobs: list[_JobFeatures] = []
        for job, requirements in zip(jobs, self._index._requirements):
            rows, technical_total, transferable_total = [], 0.0, 0.0
            mass = [0.0] * len(DOMAINS)
            for _skill, importance, alts in requirements:
                weight = IMPORTANCE_WEIGHT[importance]
                specificity = statistics.fmean(skill_specificity(a) for a in alts)
                technical_weight = weight * max(specificity, SPECIFICITY_FLOOR)
                transferable_weight = weight * (1 - specificity)
                rows.append((alts, technical_weight, transferable_weight, importance == "must-have"))
                technical_total += technical_weight
                transferable_total += transferable_weight
                for alt in alts:
                    domains = SKILL_DOMAINS.get(alt[len(UMBRELLA):] if alt.startswith(UMBRELLA) else alt, ())
                    for d in domains:  # a generic skill adds almost nothing: its specificity is near 0
                        mass[_DOMAIN_INDEX[d]] += weight * specificity / len(alts) / len(domains)
            shares = self._job_shares(job, mass)
            expanded = tuple(sum(shares[b] * affinity[a][b] for b in range(len(DOMAINS))) for a in range(len(DOMAINS)))
            self._jobs.append(_JobFeatures(tuple(rows), technical_total, transferable_total, shares, expanded))

    @staticmethod
    def _job_shares(job: JobPosting, mass: list[float]) -> tuple:
        named = [d for d in re.split(r"[,/\s]+", (job.family or "").lower()) if d in _DOMAIN_INDEX]
        if named:  # an explicit family overrides inference
            mass = [1.0 if d in named else 0.0 for d in DOMAINS]
        else:
            hinted = sorted({d for token in re.findall(r"[a-z]+", job.title.lower()) for d in TITLE_HINTS.get(token, ())})
            for d in hinted:
                mass[_DOMAIN_INDEX[d]] += TITLE_WEIGHT / len(hinted)
        total = sum(mass)
        return tuple(m / total for m in mass) if total else tuple(mass)

    @staticmethod
    def _candidate(matcher: SkillMatcher) -> tuple[list[float], list[float]]:
        """(share of the candidate's skills in each path, depth 0-1 in each path)."""
        mass, specific = [0.0] * len(DOMAINS), [0] * len(DOMAINS)
        for skill in sorted(matcher.listed):  # sorted: float sums must not depend on set order
            domains = SKILL_DOMAINS.get(skill)
            if not domains:
                continue
            specificity = skill_specificity(skill)
            for d in domains:
                mass[_DOMAIN_INDEX[d]] += specificity / len(domains)
                specific[_DOMAIN_INDEX[d]] += specificity >= 0.5
        total = sum(mass)
        shares = [m / total for m in mass] if total else mass
        return shares, [min(1.0, n / DEPTH_SATURATION) for n in specific]

    @staticmethod
    def _paths(shares) -> list[str]:
        order = sorted(range(len(DOMAINS)), key=lambda i: (-shares[i], i))
        return [DOMAINS[i] for i in order if shares[i] >= 0.2][:3]

    def rank(self, skills: list[str], top: Optional[int] = 5, profile: Optional[ResumeProfile] = None) -> list[CareerMatch]:
        """Best career fit first; ties go to technical fit, then fewer missing must-haves, then catalog order.

        Pass the resume profile to let skills shown in projects and experience count toward `experience`."""
        matcher = SkillMatcher(skills)
        evidence = applied_skill_evidence(profile) if profile else {}
        evidence_matchers = {
            source: SkillMatcher([key for key, (kind, _weight) in evidence.items() if kind == source])
            for source in EVIDENCE_WEIGHTS
        } if evidence else {}
        shares, depth = self._candidate(matcher)
        has_candidate_domain = any(shares)
        w = self.weights
        scored = []
        for i, job in enumerate(self._jobs):
            technical_sum = transferable_sum = evidence_sum = 0.0
            missing_must = 0
            for alts, technical_weight, transferable_weight, is_must in job.rows:
                status = matcher.judge(alts)[0]
                credit = STATUS_CREDIT[status]
                technical_sum += technical_weight * credit
                transferable_sum += transferable_weight * credit
                evidence_credit = 0.0
                for source, source_matcher in evidence_matchers.items():
                    evidence_credit = max(
                        evidence_credit,
                        STATUS_CREDIT[source_matcher.judge(alts)[0]] * EVIDENCE_WEIGHTS[source],
                    )
                evidence_sum += technical_weight * evidence_credit
                missing_must += is_must and status == "missing"
            technical = technical_sum / job.technical_total if job.technical_total else 0.0
            # No generic requirement to judge: shared basics say nothing, so fall back to the technical fit.
            transferable = transferable_sum / job.transferable_total if job.transferable_total > 1e-9 else technical
            if has_candidate_domain and any(job.domains):
                domain = sum(c * e for c, e in zip(shares, job.expanded))
            else:
                domain = NEUTRAL_DOMAIN
            depth_experience = sum(d * s for d, s in zip(depth, job.domains))
            if evidence_matchers and job.technical_total:
                evidence_experience = evidence_sum / job.technical_total
                experience = 0.25 * depth_experience + 0.75 * evidence_experience
            else:
                experience = depth_experience
            total = w.technical * technical + w.domain * domain + w.experience * experience + w.transferable * transferable
            key = (-round(total, 9), -round(technical, 9), missing_must, i)  # rounding hides float noise in ties
            scored.append((key, (technical, domain, experience, transferable, total)))
        best = heapq.nsmallest(top, scored) if top is not None else sorted(scored)
        results = []
        for key, (technical, domain, experience, transferable, total) in best:
            base = self._index._explain(key[3], matcher)
            results.append(CareerMatch(
                job=base.job, score=round(100 * total), missing_must_haves=base.missing_must_haves,
                assessments=base.assessments,
                career=CareerBreakdown(
                    technical=round(technical, 3), domain=round(domain, 3), experience=round(experience, 3),
                    transferable=round(transferable, 3), total=round(total, 4), skill_score=base.score,
                    candidate_paths=self._paths(shares), job_paths=self._paths(self._jobs[key[3]].domains))))
        return results


def rerank_with_llm(matches: list[JobMatch], profile: ResumeProfile, workers: int = 4) -> list[JobMatch]:
    """Re-check matches with the Matching Agent (one LLM call each, run in parallel)."""

    def check(m: JobMatch) -> JobMatch:
        jd = JobRequirements(title=m.job.title, company=m.job.company,
                             must_have_skills=m.job.must_have_skills, nice_to_have_skills=m.job.nice_to_have_skills)
        llm = match_skills({"resume_profile": profile, "job_requirements": jd})["match"]
        missing = [a.skill for a in llm.assessments if a.importance == "must-have" and a.status == "missing"]
        return m.model_copy(update={"llm_score": llm.score, "llm_summary": llm.summary,
                                    "assessments": llm.assessments, "missing_must_haves": missing})

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(matches)))) as pool:
        checked = list(pool.map(check, matches))
    return sorted(checked, key=lambda m: (-m.llm_score, -m.score))


def find_jobs(skills: list[str], jobs: list[JobPosting], top: int = 5, rerank: int = 0,
              profile: Optional[ResumeProfile] = None, timings: Optional[dict] = None,
              ranking: str = "skills", weights: Optional[RankingWeights] = None) -> list[JobMatch]:
    """Rank jobs by skill fit; with rerank > 0, the top `rerank` are re-checked by the LLM.

    ranking="skills" (the default) scores skill coverage only. ranking="career" also weighs career path,
    experience and transferable skills (see CareerRanker); `weights` tunes it, and `profile` lets projects
    and experience count as evidence.

    Pass a dict as `timings` to get the time spent in each stage (ms).
    """
    if top < 1 or rerank < 0:
        raise ValueError(f"top must be at least 1 and rerank at least 0 (got top={top}, rerank={rerank})")
    if ranking not in RANKING_MODES:
        raise ValueError(f"ranking must be one of {RANKING_MODES} (got {ranking!r})")
    timings = {} if timings is None else timings
    start = time.perf_counter()
    if ranking == "career":
        matches = CareerRanker(jobs, weights).rank(skills, top=max(top, rerank), profile=profile)
    else:
        matches = JobIndex(jobs).rank(skills, top=max(top, rerank))
    timings["rank_ms"] = (time.perf_counter() - start) * 1000
    if rerank and matches:
        start = time.perf_counter()
        profile = profile or ResumeProfile(name="Candidate", skills=skills)
        try:
            matches = rerank_with_llm(matches[:rerank], profile) + matches[rerank:]
            timings["reranked"] = min(rerank, len(matches))
        except ModelRateLimitError:
            # The quick ranking is still valid; return it rather than nothing.
            print("⚠️  Gemini quota ran out during the LLM re-check; showing the quick ranking.", file=sys.stderr)
        timings["rerank_ms"] = (time.perf_counter() - start) * 1000
    return matches[:top]


# ---------------------------------------------------------------------------
# Inputs and report
# ---------------------------------------------------------------------------

SAMPLE_RESUME = """
Ploy Srisuk
B.Eng. Computer Engineering (Year 3), Chulalongkorn University — GPA 3.42

Skills: Python, Java, SQL, Git, HTML/CSS, basic React, pandas

Projects
- Campus Food Finder: React + Firebase web app to browse canteen menus (team of 3)
- Thai News Classifier: scikit-learn model classifying news headlines into 6 topics, 84% accuracy
- Data Structures coursework: implemented AVL tree and graph algorithms in Java

Activities
- Staff, CU Hackathon 2026
- Member, Google Developer Student Club
"""

SAMPLE_JOB_DESCRIPTION = """
Software Engineer Intern (Backend) — FinTech startup, Bangkok
Duration: 3 months, summer 2027

Responsibilities
- Build and maintain REST APIs for our payments platform
- Write unit and integration tests
- Work with the team on code reviews and deployments

Requirements
- Currently pursuing a degree in Computer Science/Engineering
- Strong Python skills
- Experience with SQL databases
- Familiarity with REST API design
- Git and collaborative development

Nice to have
- FastAPI or Django
- Docker
- Cloud experience (AWS/GCP)
- Interest in finance or payments
"""

# Skills listed on SAMPLE_RESUME, used by --find-jobs when no --skills or --resume is given.
SAMPLE_SKILLS = ["Python", "Java", "SQL", "Git", "HTML/CSS", "React", "pandas", "scikit-learn", "Firebase"]


def _job(id, title, company, location, must, nice):
    return JobPosting(id=id, title=title, company=company, location=location,
                      must_have_skills=must, nice_to_have_skills=nice)


# Built-in catalog of fictional internship postings (Thailand), already in structured form.
SAMPLE_JOBS = [
    _job("backend-fintech", "Software Engineer Intern (Backend)", "FinTech startup", "Bangkok",
         ["Python", "SQL", "REST API design", "Git"], ["FastAPI or Django", "Docker", "Cloud (AWS/GCP)"]),
    _job("frontend-ecommerce", "Frontend Developer Intern", "E-commerce platform", "Bangkok",
         ["JavaScript", "React", "HTML/CSS", "Git"], ["TypeScript", "Next.js", "Figma"]),
    _job("data-analyst", "Data Analyst Intern", "Retail group", "Bangkok",
         ["SQL", "Excel", "Python"], ["Tableau or Power BI", "pandas", "Statistics"]),
    _job("ml-engineer", "Machine Learning Engineer Intern", "AI research lab", "Bangkok",
         ["Python", "Machine Learning", "PyTorch or TensorFlow"], ["NLP", "Docker", "Linux"]),
    _job("android", "Android Developer Intern", "Food delivery app", "Bangkok",
         ["Kotlin or Java", "Android SDK", "Git"], ["Jetpack Compose", "Firebase", "REST APIs"]),
    _job("ios", "iOS Developer Intern", "Travel booking app", "Chiang Mai",
         ["Swift", "iOS SDK", "Git"], ["SwiftUI", "Firebase"]),
    _job("devops", "DevOps / Cloud Intern", "Telecom company", "Bangkok",
         ["Linux", "Docker", "Bash", "Git"], ["Kubernetes", "AWS", "Terraform", "CI/CD"]),
    _job("qa-automation", "QA Automation Intern", "Banking software vendor", "Bangkok",
         ["Python or Java", "Software testing", "Git"], ["Selenium", "CI/CD", "SQL"]),
    _job("embedded", "Embedded Systems Intern", "Electronics manufacturer", "Chonburi",
         ["C", "C++", "Microcontrollers"], ["RTOS", "Python", "PCB design"]),
    _job("ux-ui", "UX/UI Design Intern", "Digital agency", "Remote",
         ["Figma", "UI design", "User research"], ["HTML/CSS", "Prototyping"]),
    _job("fullstack", "Full-stack Developer Intern", "SaaS startup", "Remote",
         ["JavaScript", "Node.js", "React", "SQL"], ["TypeScript", "Docker", "MongoDB"]),
    _job("data-engineer", "Data Engineering Intern", "Logistics company", "Bangkok",
         ["Python", "SQL", "ETL"], ["Apache Spark", "Airflow", "Cloud (AWS/GCP)"]),
    _job("security", "Cybersecurity Intern", "Managed security provider", "Bangkok",
         ["Networking", "Linux", "Python"], ["Penetration testing", "Wireshark"]),
    _job("game-dev", "Game Developer Intern", "Indie game studio", "Bangkok",
         ["C#", "Unity"], ["Blender", "C++"]),
]

STATUS_ICON = {"met": "✅", "partial": "🟡", "missing": "❌"}
SEVERITY_ICON = {"blocking": "🔴", "important": "🟠", "nice-to-have": "🟢"}
ELIGIBILITY_ICON = {"met": "✅", "not-met": "❌", "unclear": "❔"}


def read_text(source: str) -> str:
    """Read a .pdf or text file, or treat the argument itself as raw text."""
    path = Path(source)
    try:
        is_file = path.is_file()
    except OSError:  # e.g. ENAMETOOLONG: raw text longer than a file name can be
        is_file = False
    if not is_file:
        return source
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return path.read_text(encoding="utf-8")


def print_report(result: CareerState) -> None:
    profile, jd = result["resume_profile"], result["job_requirements"]
    match, gaps, rec = result["match"], result["gap_analysis"], result["recommendation"]
    line = "=" * 60

    print(f"\n{line}\n🎯 {profile.name}  →  {jd.title}" + (f" @ {jd.company}" if jd.company else ""))
    print(f"{line}\n\n📊 MATCH SCORE: {match.score}/100\n{match.summary}\n")
    for a in match.assessments:
        print(f"  {STATUS_ICON[a.status]} {a.skill} ({a.importance}): {a.evidence}")
    if match.eligibility:
        print("\n  🎓 Eligibility (not part of the score):")
        for c in match.eligibility:
            print(f"  {ELIGIBILITY_ICON[c.status]} {c.criterion}: {c.evidence}")

    print(f"\n{line}\n🔍 GAP ANALYSIS\n{line}")
    for g in gaps.gaps:
        print(f"  {SEVERITY_ICON[g.severity]} [{g.severity}] {g.skill}: {g.why_it_matters}")
    if gaps.strengths:
        print("\n  💪 Strengths to lean on:")
        for s in gaps.strengths:
            print(f"   - {s}")

    print(f"\n{line}\n🗺️  LEARNING PLAN\n{line}")
    for w in rec.learning_plan:
        print(f"\n  Week {w.week}: {w.focus}")
        for t in w.tasks:
            print(f"   - {t}")
        if w.resources:
            print(f"   📚 {', '.join(w.resources)}")

    print(f"\n{line}\n🛠️  PROJECT IDEAS\n{line}")
    for p in rec.project_ideas:
        print(f"\n  {p.title} [{', '.join(p.skills_covered)}]\n   {p.description}")

    print(f"\n{line}\n📝 RESUME IMPROVEMENTS\n{line}")
    for r in rec.resume_improvements:
        print(f"  - {r}")

    print(f"\n{line}\n🎤 INTERVIEW PREP\n{line}")
    for i, q in enumerate(rec.interview_questions, 1):
        print(f"\n  {i}. {q.question}\n     Why: {q.why_asked}\n     Prep: {q.how_to_prepare}")
    print()


def read_job_file(path: str) -> list[dict]:
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
        raise ValueError(f"{path} must contain a JSON list of job objects")
    return items


def needs_parsing(item: dict) -> bool:
    """True for a description-only posting, which costs one LLM call to turn into skills."""
    return not (item.get("must_have_skills") or item.get("nice_to_have_skills")) and bool(item.get("description"))


def load_jobs(items: list[dict]) -> list[JobPosting]:
    """Turn posting dicts into JobPostings. Each one needs a title and skill lists, or a raw
    "description" that the JD Analyzer turns into skills (one LLM call per such posting)."""
    jobs = []
    for i, item in enumerate(items, 1):
        item = {**item, "id": str(item.get("id", f"job-{i}"))}
        if needs_parsing(item):
            jd = analyze_job({"job_description": item["description"]})["job_requirements"]
            item.update(title=item.get("title") or jd.title, company=item.get("company") or jd.company,
                        must_have_skills=jd.must_have_skills, nice_to_have_skills=jd.nice_to_have_skills)
        jobs.append(JobPosting.model_validate({k: v for k, v in item.items() if k in JobPosting.model_fields}))
    return jobs


def print_job_matches(matches: list[JobMatch], n_jobs: int, skills: list[str], timings: dict) -> None:
    line = "=" * 60
    print(f"\n{line}\n🔎 Top {len(matches)} of {n_jobs} jobs for: {', '.join(skills)}\n{line}")
    for rank, m in enumerate(matches, 1):
        where = " · ".join(x for x in (m.job.company, m.job.location) if x)
        score = f"{m.score}/100" if m.llm_score is None else f"{m.llm_score}/100 (LLM re-check; quick score {m.score})"
        print(f"\n {rank}. {m.job.title}" + (f" — {where}" if where else "") + f"\n    📊 {score}")
        if isinstance(m, CareerMatch):
            c = m.career
            print(f"    🧭 career fit: technical {c.technical:.2f} · domain {c.domain:.2f} · experience "
                  f"{c.experience:.2f} · transferable {c.transferable:.2f}  (skills only: {c.skill_score}/100)")
            print(f"    🛤️  paths: you {', '.join(c.candidate_paths) or '?'} → job {', '.join(c.job_paths) or '?'}")
        print("    " + "  ".join(f"{STATUS_ICON[a.status]} {a.skill}" for a in m.assessments))
        if m.missing_must_haves:
            print(f"    ⚠️  Missing must-haves: {', '.join(m.missing_must_haves)}")
        if m.llm_summary:
            print(f"    💬 {m.llm_summary}")
    timing = f"Ranked {n_jobs} jobs in {timings['rank_ms']:.1f} ms"
    if "reranked" in timings:
        timing += f"; LLM re-check of the top {timings['reranked']} took {timings['rerank_ms'] / 1000:.1f} s"
    print(f"\n⏱️  {timing}.\n")


# ---------------------------------------------------------------------------
# Accuracy evaluation (--eval), benchmark (--bench) and self-tests (--self-test)
# None of these call the LLM, so they run without an API key.
# ---------------------------------------------------------------------------

# Requirement level: (candidate skills, requirement, status a careful human reviewer would give).
# The last three are out-of-vocabulary for the matcher and are kept on purpose, so the score
# shows the limit of a table-driven matcher (that is what --rerank is for).
REQUIREMENT_CASES = [
    (["Python"], "Strong Python skills", "met"),
    (["Postgres"], "PostgreSQL", "met"),
    (["PostgreSQL"], "Experience with SQL databases", "met"),
    (["MySQL"], "PostgreSQL", "partial"),
    (["Django"], "FastAPI or Django", "met"),
    (["Flask"], "FastAPI or Django", "partial"),
    (["AWS"], "Cloud (AWS/GCP)", "met"),
    (["Azure"], "Cloud (AWS/GCP)", "partial"),
    (["Cloud computing"], "Cloud (AWS/GCP)", "partial"),
    (["Azure"], "AWS or GCP", "partial"),
    (["ReactJS"], "React", "met"),
    (["Next.js"], "JavaScript", "met"),
    (["TypeScript"], "JavaScript", "met"),
    (["Java"], "Kotlin or Java", "met"),
    (["Kotlin"], "Java", "partial"),
    (["Python"], "Java", "missing"),
    (["Java"], "JavaScript", "missing"),
    (["scikit-learn"], "Machine Learning", "met"),
    (["PyTorch"], "PyTorch or TensorFlow", "met"),
    (["Keras"], "TensorFlow", "partial"),
    (["C++"], "C", "partial"),
    (["C"], "C#", "missing"),
    (["C++"], "C/C++", "met"),
    (["Docker"], "Kubernetes", "partial"),
    (["GitHub"], "Git", "met"),
    (["Selenium"], "Software testing", "met"),
    (["pytest"], "Unit testing", "met"),
    (["Excel"], "Power BI", "missing"),
    (["Tableau"], "Tableau or Power BI", "met"),
    (["HTML", "CSS"], "HTML/CSS", "met"),
    (["HTML/CSS"], "CSS", "met"),
    (["HTML"], "HTML/CSS", "partial"),
    (["CSS"], "HTML", "missing"),
    (["Node"], "Node.js", "met"),
    (["REST APIs"], "REST API design", "met"),
    (["Jetpack Compose"], "Android SDK", "met"),
    (["GitHub Actions"], "CI/CD", "met"),
    (["Arduino"], "Microcontrollers", "met"),
    (["SQL"], "NoSQL", "missing"),
    (["Go"], "Google Cloud", "missing"),
    (["React"], "React Native", "partial"),
    (["Spring Boot"], "Java", "met"),
    (["MongoDB"], "NoSQL databases", "met"),
]

# Ranking level: candidate skills and how well each SAMPLE_JOBS posting fits them, judged by hand:
# 2 = strong fit (apply now), 1 = stretch (worth applying after some prep), unlisted = 0 (poor fit).
EVAL_CANDIDATES = {
    "backend": (["Python", "Django", "PostgreSQL", "Git", "REST APIs", "Docker", "Linux"],
                {"backend-fintech": 2, "data-engineer": 1, "devops": 1, "qa-automation": 1,
                 "data-analyst": 1, "security": 1, "fullstack": 1}),
    "frontend": (["JavaScript", "TypeScript", "React", "Next.js", "HTML/CSS", "Figma", "Git"],
                 {"frontend-ecommerce": 2, "fullstack": 1, "ux-ui": 1}),
    "data-ml": (["Python", "pandas", "NumPy", "scikit-learn", "PyTorch", "SQL", "Statistics", "Jupyter"],
                {"ml-engineer": 2, "data-analyst": 2, "data-engineer": 1, "backend-fintech": 1}),
    "mobile": (["Kotlin", "Java", "Android SDK", "Jetpack Compose", "Firebase", "Git"],
               {"android": 2, "ios": 1, "qa-automation": 1}),
    "devops": (["Linux", "Bash", "Docker", "Kubernetes", "AWS", "Terraform", "Git", "Python", "GitHub Actions"],
               {"devops": 2, "backend-fintech": 1, "data-engineer": 1, "security": 1, "qa-automation": 1}),
    "embedded": (["C", "C++", "Arduino", "FreeRTOS", "Python", "PCB design"], {"embedded": 2}),
    "designer": (["Figma", "UI/UX design", "User research", "Prototyping", "HTML", "CSS"],
                 {"ux-ui": 2, "frontend-ecommerce": 1}),
    "game": (["C#", "Unity", "Blender", "C++", "Git"], {"game-dev": 2}),
    "sample-resume": (SAMPLE_SKILLS,
                      {"backend-fintech": 2, "data-analyst": 1, "qa-automation": 1, "frontend-ecommerce": 1,
                       "fullstack": 1, "ml-engineer": 1, "android": 1, "data-engineer": 1}),
}


def baseline_rank(jobs: list[JobPosting], skills: list[str]) -> list[str]:
    """Exact-name overlap with the same weights: the matcher minus aliases, implications, families."""
    have = {s.strip().lower() for s in skills}

    def score(job: JobPosting) -> float:
        reqs = dedupe_requirements(job.must_have_skills, job.nice_to_have_skills)
        total = sum(IMPORTANCE_WEIGHT[imp] for _, imp in reqs)
        return sum(IMPORTANCE_WEIGHT[imp] for s, imp in reqs if s.strip().lower() in have) / total if total else 0.0

    return [job.id for job in sorted(jobs, key=score, reverse=True)]


def ranking_metrics(ranked: list[str], labels: dict[str, int], k: int = 5) -> dict[str, float]:
    """Top-1 (best result is a strong fit), MRR of the first strong fit, NDCG@k, Recall@k of any fit."""
    first_strong = next((r for r, job in enumerate(ranked, 1) if labels.get(job) == 2), None)
    dcg = sum((2 ** labels.get(job, 0) - 1) / math.log2(r + 2) for r, job in enumerate(ranked[:k]))
    ideal = sum((2 ** g - 1) / math.log2(r + 2) for r, g in enumerate(sorted(labels.values(), reverse=True)[:k]))
    relevant = {job for job, g in labels.items() if g >= 1}
    return {
        "Top-1": float(labels.get(ranked[0]) == 2),
        "MRR": 1 / first_strong if first_strong else 0.0,
        f"NDCG@{k}": dcg / ideal if ideal else 0.0,
        f"Recall@{k}": len(relevant & set(ranked[:k])) / len(relevant) if relevant else 1.0,
    }


def evaluate(jobs: list[JobPosting] = SAMPLE_JOBS) -> dict:
    """Compare skill-only, career-aware, and exact-name ranking on identical labelled cases."""
    misses, baseline_correct = [], 0
    for skills, requirement, expected in REQUIREMENT_CASES:
        got = SkillMatcher(skills).judge(alternatives(requirement))[0]
        if got != expected:
            misses.append((skills, requirement, expected, got))
        exact = "met" if requirement.strip().lower() in {s.strip().lower() for s in skills} else "missing"
        baseline_correct += exact == expected

    skill_index = JobIndex(jobs)
    career_ranker = CareerRanker(jobs)
    per_candidate, skill_rows, career_rows, baseline_rows = {}, [], [], []
    for name, (skills, labels) in EVAL_CANDIDATES.items():
        skill_ranked = [m.job.id for m in skill_index.rank(skills, top=None)]
        career_ranked = [m.job.id for m in career_ranker.rank(skills, top=None)]
        baseline_ranked = baseline_rank(jobs, skills)
        skill_metrics = ranking_metrics(skill_ranked, labels)
        career_metrics = ranking_metrics(career_ranked, labels)
        baseline_metrics = ranking_metrics(baseline_ranked, labels)
        skill_rows.append(skill_metrics)
        career_rows.append(career_metrics)
        baseline_rows.append(baseline_metrics)
        per_candidate[name] = {
            "skills_top3": skill_ranked[:3], "career_top3": career_ranked[:3],
            "baseline_top3": baseline_ranked[:3],
            "skills": skill_metrics, "career": career_metrics, "baseline": baseline_metrics,
        }

    def mean(rows):
        return {metric: statistics.mean(row[metric] for row in rows) for metric in rows[0]}

    n = len(REQUIREMENT_CASES)
    return {
        "requirement_accuracy": (n - len(misses)) / n,
        "baseline_requirement_accuracy": baseline_correct / n,
        "requirement_cases": n, "misses": misses,
        "ranking": {"skills": mean(skill_rows), "career": mean(career_rows), "baseline": mean(baseline_rows)},
        "per_candidate": per_candidate,
    }


def print_eval(report: dict) -> None:
    line = "=" * 60
    print(f"\\n{line}\\n📏 REQUIREMENT ACCURACY ({report['requirement_cases']} hand-labeled cases)\\n{line}")
    print(f"  matcher {report['requirement_accuracy']:6.1%}    "
          f"baseline (exact name) {report['baseline_requirement_accuracy']:6.1%}")
    for skills, requirement, expected, got in report["misses"]:
        print(f"  ✗ {', '.join(skills)} vs '{requirement}': expected {expected}, got {got}")

    print(f"\\n{line}\\n📏 RANKING QUALITY ({len(report['per_candidate'])} candidates × {len(SAMPLE_JOBS)} jobs)\\n{line}")
    rankings = report["ranking"]
    print(f"  {'metric':<10}{'skills':>9}{'career':>9}{'baseline':>10}")
    for metric in rankings["skills"]:
        print(f"  {metric:<10}{rankings['skills'][metric]:>9.2f}"
              f"{rankings['career'][metric]:>9.2f}{rankings['baseline'][metric]:>10.2f}")
    print()
    for name, row in report["per_candidate"].items():
        print(f"  {name:<14} skills NDCG@5 {row['skills']['NDCG@5']:.2f} "
              f"career NDCG@5 {row['career']['NDCG@5']:.2f}")
        print(f"    skills top 3: {', '.join(row['skills_top3'])}")
        print(f"    career top 3: {', '.join(row['career_top3'])}")
    print()


def synthetic_catalog(n: int, seed: int = 7) -> list[JobPosting]:
    """n random postings drawn from the skills used in SAMPLE_JOBS, ~15% with 'X or Y' requirements."""
    rng = random.Random(seed)
    vocab = sorted({s for job in SAMPLE_JOBS for s in job.must_have_skills + job.nice_to_have_skills})

    def requirement() -> str:
        return f"{rng.choice(vocab)} or {rng.choice(vocab)}" if rng.random() < 0.15 else rng.choice(vocab)

    return [JobPosting(id=f"job-{i}", title=f"Synthetic job {i}",
                       must_have_skills=[requirement() for _ in range(rng.randint(2, 5))],
                       nice_to_have_skills=[requirement() for _ in range(rng.randint(1, 4))])
            for i in range(n)]


def benchmark(sizes=(1_000, 10_000, 50_000), queries: int = 30, top: int = 10, seed: int = 7) -> list[dict]:
    """Index build time and per-search latency on synthetic catalogs; each search uses new skills."""
    rng = random.Random(seed)
    vocab = sorted({s for job in SAMPLE_JOBS for s in job.must_have_skills + job.nice_to_have_skills})
    results = []
    for n in sizes:
        catalog = synthetic_catalog(n, seed)
        canon.cache_clear()
        alternatives.cache_clear()
        start = time.perf_counter()
        index = JobIndex(catalog)
        build_ms = (time.perf_counter() - start) * 1000
        latencies = []
        for _ in range(queries):
            skills = rng.sample(vocab, rng.randint(4, 10))
            start = time.perf_counter()
            index.rank(skills, top=top)
            latencies.append((time.perf_counter() - start) * 1000)
        results.append({
            "jobs": n, "build_ms": build_ms,
            "p50_ms": statistics.median(latencies),
            "p95_ms": statistics.quantiles(latencies, n=20)[18],
            "jobs_per_sec": n / (statistics.mean(latencies) / 1000),
        })
    return results


def print_benchmark(results: list[dict]) -> None:
    line = "=" * 60
    print(f"\n{line}\n⚡ JOB SEARCH BENCHMARK (synthetic catalogs, top 10)\n{line}")
    print(f"  {'jobs':>8}{'index build':>14}{'search p50':>13}{'search p95':>13}{'jobs/sec':>12}")
    for r in results:
        print(f"  {r['jobs']:>8,}{r['build_ms']:>11.0f} ms{r['p50_ms']:>10.1f} ms{r['p95_ms']:>10.1f} ms"
              f"{r['jobs_per_sec']:>12,.0f}")
    print(f"\n  For comparison: the LLM Matching Agent on every job would take "
          f"{results[-1]['jobs']:,} API calls for one search.\n")


class TestSkillMatcher(unittest.TestCase):
    def test_alternatives(self):
        self.assertEqual(alternatives("Cloud (AWS/GCP)"), ("aws", "gcp", "~cloud"))
        self.assertEqual(alternatives("Tableau or Power BI"), ("tableau", "powerbi"))
        self.assertEqual(alternatives("C/C++"), ("c", "c++"))
        self.assertEqual(alternatives("CI/CD"), ("cicd",))
        self.assertEqual(alternatives("HTML/CSS"), ("htmlcss",))
        self.assertEqual(alternatives("Strong Python skills"), ("python",))

    def test_listed_skill_is_cited_before_an_implied_one(self):
        status, evidence = SkillMatcher(["Next.js", "React"]).judge(("react",))
        self.assertEqual((status, evidence), ("met", "Lists React"))
        self.assertEqual(SkillMatcher(["Next.js"]).judge(("javascript",)), ("met", "Lists Next.js, which implies it"))

    def test_requirement_accuracy(self):
        report = evaluate()
        self.assertGreaterEqual(report["requirement_accuracy"], 0.90, report["misses"])


class TestJobSearch(unittest.TestCase):
    def test_full_ranking_covers_every_job_sorted_by_score(self):
        index = JobIndex(SAMPLE_JOBS)
        for skills, _ in EVAL_CANDIDATES.values():
            matches = index.rank(skills, top=None)
            self.assertEqual(len(matches), len(SAMPLE_JOBS))
            scores = [m.score for m in matches]
            self.assertEqual(scores, sorted(scores, reverse=True))

    def test_fewer_missing_must_haves_win_ties(self):
        # Both score 60 for Python + SQL, but only "a" is missing a must-have (Rust).
        jobs = [_job("a", "A", None, None, ["Python", "Rust"], ["SQL"]),
                _job("b", "B", None, None, ["Python"], ["SQL", "Rust", "Go"])]
        matches = JobIndex(jobs).rank(["Python", "SQL"], top=None)
        self.assertEqual([(m.job.id, m.score) for m in matches], [("b", 60), ("a", 60)])

    def test_search_without_rerank_never_calls_the_llm(self):
        with mock.patch.object(sys.modules[__name__], "get_llm", side_effect=AssertionError("LLM called")):
            matches = find_jobs(SAMPLE_SKILLS, SAMPLE_JOBS, top=3)
        self.assertEqual(len(matches), 3)
        self.assertTrue(all(m.llm_score is None for m in matches))

    def test_rerank_orders_by_llm_score_and_keeps_the_rest(self):
        quick = find_jobs(SAMPLE_SKILLS, SAMPLE_JOBS, top=5)
        # The LLM disagrees with the quick matcher about its favourite job.
        demoted = quick[0].job.title

        def fake_match(state):
            score = 40 if state["job_requirements"].title == demoted else 90
            return {"match": MatchResult(score=score, assessments=[], summary="fake")}

        with mock.patch.object(sys.modules[__name__], "match_skills", side_effect=fake_match) as fake:
            reranked = find_jobs(SAMPLE_SKILLS, SAMPLE_JOBS, top=5, rerank=3)
        self.assertEqual(fake.call_count, 3)
        self.assertEqual([m.llm_score for m in reranked[:3]], [90, 90, 40])
        self.assertEqual(reranked[2].job.title, demoted)
        self.assertEqual({m.job.id for m in reranked[:3]}, {m.job.id for m in quick[:3]})
        self.assertEqual([m.job.id for m in reranked[3:]], [m.job.id for m in quick[3:]])
        self.assertTrue(all(m.llm_score is None for m in reranked[3:]))


class TestPipelineGuards(unittest.TestCase):
    """Rules the code enforces on LLM output, and input checks at the CLI boundary."""

    def test_aliases_are_one_requirement(self):
        self.assertEqual(dedupe_requirements(["Postgres"], ["PostgreSQL", "Docker"]),
                         [("Postgres", "must-have"), ("Docker", "nice-to-have")])

    def test_gaps_are_reconciled_with_unmet_requirements(self):
        open_items = [
            RequirementAssessment(skill="PostgreSQL", importance="must-have", status="missing", evidence=""),
            RequirementAssessment(skill="Docker", importance="nice-to-have", status="partial", evidence=""),
        ]
        returned = [
            SkillGap(skill="Postgres", severity="important", evidence="", why_it_matters=""),  # alias, too mild
            SkillGap(skill="postgresql", severity="important", evidence="", why_it_matters=""),  # duplicate
            SkillGap(skill="Rust", severity="blocking", evidence="", why_it_matters=""),  # invented
        ]
        eligibility = [EligibilityCheck(criterion="Graduating in 2027", status="not-met", evidence="Graduates 2029."),
                       EligibilityCheck(criterion="Based in Bangkok", status="unclear", evidence="")]
        gaps = reconcile_gaps(open_items, returned, eligibility)
        self.assertEqual([(g.skill, g.severity) for g in gaps], [
            ("PostgreSQL", "blocking"), ("Graduating in 2027", "blocking"), ("Docker", "nice-to-have")])

    def test_eligibility_is_checked_once_per_criterion_and_kept_out_of_the_score(self):
        reply = MatchAssessment(
            assessments=[RequirementAssessment(skill="Python", importance="must-have", status="met", evidence="")],
            eligibility=[EligibilityCheck(criterion="enrolled in a CS degree", status="met", evidence="Year 3"),
                         EligibilityCheck(criterion="Invented criterion", status="not-met", evidence="")],
            summary="")
        jd = JobRequirements(title="Intern", must_have_skills=["Python"],
                             eligibility=["Enrolled in a CS degree", "Available summer 2027"])
        with mock.patch.object(sys.modules[__name__], "ask", return_value=reply):
            match = match_skills({"resume_profile": ResumeProfile(name="A"), "job_requirements": jd})["match"]
        self.assertEqual(match.score, 100)
        self.assertEqual([(c.criterion, c.status) for c in match.eligibility],
                         [("Enrolled in a CS degree", "met"), ("Available summer 2027", "unclear")])

    def test_learning_plan_is_held_to_the_requested_weeks(self):
        plan = [WeekPlan(week=w, focus=str(w), tasks=[]) for w in (3, 1, 2, 4)]
        with mock.patch("sys.stderr"):
            fitted = fit_plan(plan, 3)
        self.assertEqual([(w.week, w.focus) for w in fitted], [(1, "1"), (2, "2"), (3, "3")])

    def test_invalid_counts_are_rejected(self):
        with self.assertRaises(ValueError):
            find_jobs(SAMPLE_SKILLS, SAMPLE_JOBS, rerank=-1)
        with self.assertRaises(ValueError):
            find_jobs(SAMPLE_SKILLS, SAMPLE_JOBS, top=0)
        with self.assertRaises(argparse.ArgumentTypeError):
            int_at_least(1)("0")

    def test_quota_during_rerank_keeps_the_quick_ranking(self):
        quick = find_jobs(SAMPLE_SKILLS, SAMPLE_JOBS, top=5)
        with mock.patch.object(sys.modules[__name__], "match_skills", side_effect=ModelRateLimitError("quota")), \
                mock.patch("sys.stderr"):
            reranked = find_jobs(SAMPLE_SKILLS, SAMPLE_JOBS, top=5, rerank=3)
        self.assertEqual([m.job.id for m in reranked], [m.job.id for m in quick])

    def test_llm_call_budget_is_counted_before_any_call(self):
        items = [{"title": "A", "description": "..."}, {"title": "B", "must_have_skills": ["Go"]},
                 {"description": "..."}]
        args = argparse.Namespace(resume="cv.pdf", skills=None, rerank=3)
        self.assertEqual(planned_llm_calls(args, items), 1 + 2 + 3)
        args.skills = "Python"  # typed skills: the resume is not sent to the LLM
        self.assertEqual(planned_llm_calls(args, items), 5)

    def test_long_raw_text_is_not_treated_as_a_path(self):
        text = "Python developer. " * 1000
        self.assertEqual(read_text(text), text)
        with self.assertRaises(InputTooLongError):
            ask(ResumeProfile, "", "x" * (MAX_INPUT_CHARS + 1))


class TestAccuracy(unittest.TestCase):
    def test_evaluation_reports_all_rankers_on_same_labels(self):
        report = evaluate()
        self.assertEqual(set(report["ranking"]), {"skills", "career", "baseline"})
        for row in report["per_candidate"].values():
            self.assertEqual(len(row["skills_top3"]), 3)
            self.assertEqual(len(row["career_top3"]), 3)
            self.assertEqual(len(row["baseline_top3"]), 3)

    def test_experience_evidence_uses_strongest_source_once(self):
        profile = ResumeProfile(
            name="Test", skills=["Python"],
            projects=[Project(name="P", description="Built with Python", skills_used=["Python"])],
            experience=["Used Python in production"], education=["Coursework: Python"],
            activities=["Python club"],
        )
        self.assertEqual(applied_skill_evidence(profile)[canon("Python")],
                         ("work_experience", EVIDENCE_WEIGHTS["work_experience"]))

    def test_skills_section_alone_is_not_applied_evidence(self):
        self.assertEqual(applied_skill_evidence(ResumeProfile(name="Test", skills=["Python"])), {})


class TestPerformance(unittest.TestCase):
    def test_search_10k_jobs_under_budget(self):
        # Budgets are ~8x what a laptop measures (p95 ~30 ms, build ~270 ms), to stay stable on slow machines.
        result = benchmark(sizes=(10_000,), queries=20)[0]
        self.assertLess(result["p95_ms"], 250, result)
        self.assertLess(result["build_ms"], 2_000, result)


def run_self_tests() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


def run_with_llm_errors(model: str, fn, *args):
    """Run fn, turning common Gemini errors into one-line exits."""
    try:
        return fn(*args)
    except ModelRateLimitError:
        sys.exit(f"❌ Gemini quota exceeded for {model}. The free tier allows about 20 requests/day per model "
                 "(a full run uses 5). Wait for the reset or set GEMINI_MODEL to another model in .env.")
    except (ModelAuthenticationError, ModelInvalidRequestError) as e:
        # Google reports a bad key as 400 INVALID_ARGUMENT with reason API_KEY_INVALID.
        if isinstance(e, ModelInvalidRequestError) and "API_KEY_INVALID" not in str(e):
            raise
        sys.exit("❌ Gemini rejected the API key. Check GOOGLE_API_KEY in .env.")
    except ModelNotFoundError:
        sys.exit(f"❌ Model '{model}' not found. Set GEMINI_MODEL in .env to a current Gemini model.")
    except InputTooLongError as e:
        sys.exit(f"❌ {e}")


def planned_llm_calls(args, items: Optional[list[dict]]) -> int:
    """LLM calls a search will make: resume extraction, description-only postings, re-checks."""
    return (int(bool(args.resume) and not args.skills) + sum(map(needs_parsing, items or []))
            + args.rerank)


def run_find_jobs(args) -> None:
    items = read_job_file(args.jobs) if args.jobs else None
    # Check the budget before any call, so a big catalog can't drain the quota half-way through.
    calls = planned_llm_calls(args, items)
    if calls > args.max_llm_calls:
        sys.exit(f"❌ This search needs {calls} LLM calls, over --max-llm-calls {args.max_llm_calls}. "
                 "Add skill lists to description-only postings, lower --rerank, or raise the limit.")
    jobs = load_jobs(items) if items is not None else SAMPLE_JOBS
    profile = None
    if args.skills:
        skills = [s.strip() for s in args.skills.split(",") if s.strip()]
    elif args.resume:
        print("📄 Extracting skills from the resume...", file=sys.stderr)
        profile = analyze_resume({"resume_text": read_text(args.resume)})["resume_profile"]
        skills = profile.skills
    else:
        print("ℹ️  No --skills or --resume given; using the sample resume's skills.", file=sys.stderr)
        skills = SAMPLE_SKILLS

    # getattr: callers that build their own Namespace (older scripts, tests) need not know these options.
    try:
        weights = RankingWeights.parse(args.weights) if getattr(args, "weights", None) else None
    except ValueError as e:
        sys.exit(f"❌ {e}")
    timings: dict = {}
    matches = find_jobs(skills, jobs, top=args.top, rerank=args.rerank, profile=profile, timings=timings,
                        ranking=getattr(args, "ranking", "skills"), weights=weights)
    if args.json:
        print(json.dumps([m.model_dump() for m in matches], indent=2, ensure_ascii=False))
    else:
        print_job_matches(matches, len(jobs), skills, timings)


def int_at_least(minimum: int):
    """argparse type: an integer no smaller than `minimum`."""

    def parse(value: str) -> int:
        try:
            number = int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"'{value}' is not a whole number")
        if number < minimum:
            raise argparse.ArgumentTypeError(f"must be at least {minimum} (got {number})")
        return number

    return parse


def main():
    parser = argparse.ArgumentParser(description="AI Career & Internship Agent")
    parser.add_argument("--resume", help="Path to resume (.pdf/.txt) or raw text")
    parser.add_argument("--job-desc", help="Path to job description file or raw text")
    parser.add_argument("--weeks", type=int_at_least(1), default=3, help="Length of the learning plan (default: 3)")
    parser.add_argument("--json", action="store_true", help="Print raw JSON instead of the report")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--find-jobs", action="store_true", help="Rank job postings by how well they fit your skills")
    mode.add_argument("--eval", action="store_true", help="Measure job-matching accuracy on hand-labeled cases")
    mode.add_argument("--bench", action="store_true", help="Benchmark job-search speed on synthetic catalogs")
    mode.add_argument("--self-test", action="store_true", help="Run the unit, accuracy and performance tests")

    search = parser.add_argument_group("job search (--find-jobs)")
    search.add_argument("--skills", help='Comma-separated skills, e.g. "Python, SQL, React" (no LLM call needed)')
    search.add_argument("--jobs", help="JSON file of job postings (default: built-in internship catalog)")
    search.add_argument("--top", type=int_at_least(1), default=5, help="Number of jobs to show (default: 5)")
    search.add_argument("--rerank", type=int_at_least(0), default=0, metavar="N",
                        help="Re-check the top N jobs with the LLM Matching Agent (N LLM calls)")
    search.add_argument("--max-llm-calls", type=int_at_least(0), default=10, metavar="N",
                        help="Refuse a search that would make more than N LLM calls (default: 10)")
    search.add_argument("--ranking", choices=RANKING_MODES, default="skills",
                        help="skills: skill coverage only (default). career: also weigh career path, experience "
                             "and transferable skills")
    search.add_argument("--weights", metavar="W",
                        help='Career-ranking weights, e.g. "technical=0.5,domain=0.2" or "40,25,20,15" '
                             "(technical, domain, experience, transferable; default 40,25,20,15)")
    args = parser.parse_args()

    # Emoji and Thai text in the report crash non-UTF-8 consoles (e.g. default Windows code pages).
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    if args.self_test:
        sys.exit(run_self_tests())
    if args.eval:
        print_eval(evaluate())
        return
    if args.bench:
        print_benchmark(benchmark())
        return

    model = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    if args.find_jobs:
        run_with_llm_errors(model, run_find_jobs, args)
        return

    state: CareerState = {
        "resume_text": read_text(args.resume) if args.resume else SAMPLE_RESUME,
        "job_description": read_text(args.job_desc) if args.job_desc else SAMPLE_JOB_DESCRIPTION,
        "weeks": args.weeks,
    }
    print(f"🚀 Analyzing with {model}...", file=sys.stderr)  # stderr keeps --json output parseable
    result = run_with_llm_errors(model, build_graph().invoke, state)

    if args.json:
        keys = ("resume_profile", "job_requirements", "match", "gap_analysis", "recommendation")
        print(json.dumps({k: result[k].model_dump() for k in keys}, indent=2, ensure_ascii=False))
    else:
        print_report(result)


if __name__ == "__main__":
    main()