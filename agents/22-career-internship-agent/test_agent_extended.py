"""
Extended test suite for agent.py (AI Career & Internship Agent).

Complements the built-in `python agent.py --self-test` with:

  1. Correctness   - canon/alternatives/SkillMatcher/scoring/reconcile_gaps/fit_plan/IO/CLI,
                     property tests (monotonicity, order-invariance, oracle re-derivation),
                     and the career ranker (--ranking career).
  2. Accuracy      - an independent HELD-OUT set of requirement cases and candidate rankings,
                     written separately from the in-repo labels so the numbers are not
                     self-graded, for both rankers. Misses are printed, not hidden.
  3. Performance   - scaling, per-search latency, memory, caching, concurrency, hostile input.
  4. Regressions   - defects this suite found and that are now fixed (Thai names, slashes in names,
                     severity rules, mistyped paths, blank --skills, quadratic parsing). Each test
                     says what used to go wrong.

No test calls the LLM or needs an API key. If python-dotenv / langchain-core /
langchain-google-genai / langgraph are not installed, tiny stand-ins are generated in a temp dir
(real packages win when present, so the same file also runs unchanged in your real environment).
pydantic is always required.

Run:   python -m unittest test_agent_extended -v        (or: pytest test_agent_extended.py -v)
Skip the slow perf tests:  SKIP_PERF=1 python -m unittest test_agent_extended
"""

import argparse
import contextlib
import importlib
import io
import json
import math
import os
import random
import statistics
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import tracemalloc
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# ---------------------------------------------------------------------------
# Dependency stand-ins (only for packages that are genuinely missing)
# ---------------------------------------------------------------------------

STUB_DIR = None

_STUBS = {
    "dotenv": {
        "__init__.py": "def load_dotenv(*args, **kwargs):\n    return False\n",
    },
    "langchain_core": {
        "__init__.py": "",
        "exceptions.py": textwrap.dedent('''
            class ModelRateLimitError(Exception): pass
            class ModelAuthenticationError(Exception): pass
            class ModelInvalidRequestError(Exception): pass
            class ModelNotFoundError(Exception): pass
        '''),
        "messages.py": textwrap.dedent('''
            class _Msg:
                def __init__(self, content): self.content = content
            class HumanMessage(_Msg): pass
            class SystemMessage(_Msg): pass
        '''),
    },
    "langchain_google_genai": {
        "__init__.py": textwrap.dedent('''
            class ChatGoogleGenerativeAI:
                def __init__(self, model=None, **kw): self.model = model
                def with_structured_output(self, schema):
                    raise RuntimeError("stub: no network in tests")
        '''),
    },
    "langgraph": {
        "__init__.py": "",
        "graph.py": textwrap.dedent('''
            START, END = "__start__", "__end__"

            class StateGraph:
                def __init__(self, state_type):
                    self.nodes, self.edges = {}, []
                def add_node(self, name, fn): self.nodes[name] = fn
                def add_edge(self, src, dst): self.edges.append((src, dst))
                def compile(self): return _Compiled(self)

            class _Compiled:
                def __init__(self, g): self.g = g
                def invoke(self, state):
                    state, done = dict(state), set()
                    deps = {n: set() for n in self.g.nodes}
                    for src, dst in self.g.edges:
                        if dst == END:
                            continue
                        srcs = src if isinstance(src, list) else [src]
                        deps[dst] |= {s for s in srcs if s != START}
                    pending = set(self.g.nodes)
                    while pending:
                        ready = [n for n in sorted(pending) if deps[n] <= done]
                        if not ready:
                            raise RuntimeError("cycle or unreachable node")
                        for n in ready:
                            state.update(self.g.nodes[n](state))
                            done.add(n)
                            pending.discard(n)
                    return state
        '''),
    },
}


def _install_stubs():
    global STUB_DIR
    for pkg, files in _STUBS.items():
        try:
            importlib.import_module(pkg)
            continue
        except ImportError:
            pass
        STUB_DIR = STUB_DIR or Path(tempfile.mkdtemp(prefix="agent_stubs_"))
        for name, body in files.items():
            target = STUB_DIR / pkg / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(body, encoding="utf-8")
    if STUB_DIR:
        sys.path.append(str(STUB_DIR))  # append: real packages always win


_install_stubs()

import agent as A  # noqa: E402

SKIP_PERF = os.getenv("SKIP_PERF") == "1"
PERF_REPORT: list[tuple[str, str]] = []


def tearDownModule():
    if PERF_REPORT:
        print("\n" + "=" * 64 + "\nMEASUREMENTS (this machine)\n" + "=" * 64, file=sys.stderr)
        for name, value in PERF_REPORT:
            print(f"  {name:<46}{value}", file=sys.stderr)


def RA(skill, importance="must-have", status="met", evidence=""):
    return A.RequirementAssessment(skill=skill, importance=importance, status=status, evidence=evidence)


def status_of(skills, requirement):
    return A.SkillMatcher(skills).judge(A.alternatives(requirement))[0]


def cli(*args, env_extra=None, timeout=120):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(filter(None, [str(STUB_DIR or ""), os.environ.get("PYTHONPATH", "")])))
    env.pop("GOOGLE_API_KEY", None)
    env.update(env_extra or {})
    return subprocess.run([sys.executable, str(HERE / "agent.py"), *args], capture_output=True, text=True,
                          env=env, cwd=HERE, timeout=timeout, encoding="utf-8")


# ===========================================================================
# 1. CORRECTNESS
# ===========================================================================


class TestNormAndCanon(unittest.TestCase):
    def test_norm_keeps_symbols_that_distinguish_languages(self):
        self.assertEqual(A.norm("C#"), "c#")
        self.assertEqual(A.norm("C++"), "c++")
        self.assertEqual(A.norm("Node.js"), "nodejs")
        self.assertEqual(len({A.norm(x) for x in ("C", "C#", "C++")}), 3)

    def test_canon_table(self):
        cases = {
            "Strong Python skills": "python", "  PYTHON  ": "python", "Postgres": "postgresql",
            "Node.js": "nodejs", "Node": "nodejs", "Next.js": "nextjs", "scikit-learn": "scikitlearn",
            "Sklearn": "scikitlearn", "Experience with SQL databases": "sql",
            "Familiarity with REST API design": "restapi", "REST APIs": "restapi", "CI/CD": "cicd",
            "C#": "c#", "C++": "c++", "Go": "go", "Golang": "go", "Power BI": "powerbi",
            "Google Cloud Platform": "gcp", "Unix": "linux", "Proficiency in Python": "python",
            "Solid understanding of SQL databases": "sql", "Good knowledge of Java programming": "java",
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(A.canon(raw), expected)

    def test_canon_never_returns_empty_for_latin_text_even_if_all_filler(self):
        for raw in ("Experience", "Programming languages", "Skills"):
            with self.subTest(raw=raw):
                self.assertNotEqual(A.canon(raw), "")

    def test_canon_empty_inputs(self):
        self.assertEqual(A.canon(""), "")
        self.assertEqual(A.canon("   "), "")

    def test_canon_is_idempotent_over_the_whole_vocabulary(self):
        vocabulary = set(A.ALIASES) | set(A.ALIASES.values()) | set(A.IMPLIES) | set(A.SIBLINGS)
        for key in sorted(vocabulary):
            with self.subTest(key=key):
                self.assertEqual(A.canon(A.canon(key)), A.canon(key))

    def test_vocabulary_tables_are_internally_consistent(self):
        for key, value in A.ALIASES.items():
            with self.subTest(alias=key):
                self.assertEqual(A.norm(key), key, "alias keys must already be norm()-ed")
                self.assertEqual(A.norm(value), value, "alias values must already be norm()-ed")
                self.assertNotIn(value, A.ALIASES, "alias chain: canon() maps only once")
        for key in A.IMPLIES:
            self.assertEqual(A.norm(key), key)
            self.assertNotIn(key, A.ALIASES, f"IMPLIES key {key!r} is unreachable: canon() rewrites it first")
        for family in A.FAMILIES:
            for skill in family:
                self.assertEqual(A.norm(skill), skill)
                self.assertNotIn(skill, A.ALIASES, f"family member {skill!r} is unreachable")
        for compound, parts in A.COMPOUNDS.items():
            self.assertIn(compound, A.IMPLIES)
            self.assertEqual(set(A.IMPLIES[compound]), set(parts))

    def test_no_implication_cycles_in_the_shipped_tables(self):
        def reachable(start):
            seen, stack = set(), [start]
            while stack:
                for nxt in A.IMPLIES.get(stack.pop(), ()):
                    if nxt not in seen:
                        seen.add(nxt)
                        stack.append(nxt)
            return seen
        for skill in A.IMPLIES:
            with self.subTest(skill=skill):
                self.assertNotIn(skill, reachable(skill))

    def test_filler_is_not_stripped_from_the_skill_itself(self):
        # Was: "database" was filler anywhere, so "Database design" collapsed to "design".
        self.assertNotEqual(A.canon("Database design"), A.canon("Design"))


class TestAlternatives(unittest.TestCase):
    def test_table(self):
        cases = {
            "Python or Java": ("python", "java"), "Python OR Java": ("python", "java"),
            "FastAPI or Django": ("fastapi", "django"), "Cloud (AWS or GCP)": ("aws", "gcp", "~cloud"),
            "Cloud (AWS/GCP)": ("aws", "gcp", "~cloud"), "AWS/GCP/Azure": ("aws", "gcp", "azure"),
            "Python, Java": ("python", "java"), "Python | Java": ("python", "java"),
            "Node.js/Express": ("nodejs", "express"), "TCP/IP": ("networking",), "UI/UX design": ("uidesign",),
            "Python or python": ("python",), "Experience with Docker": ("docker",),
            "Frameworks (Django/Flask)": ("django", "flask"), "C/C++": ("c", "c++"), "CI/CD": ("cicd",),
        }
        for raw, expected in cases.items():
            with self.subTest(raw=raw):
                self.assertEqual(A.alternatives(raw), expected)

    def test_never_empty_and_never_crashes_on_odd_input(self):
        for raw in ("", "   ", "()", "(", ")", "/", "or", " or ", "a or", "(()", "x(y", "x(y)z", "ภาษาไทย", "🚀"):
            with self.subTest(raw=raw):
                self.assertGreaterEqual(len(A.alternatives(raw)), 1)

    def test_plain_requirement_matches_a_candidate_who_lists_exactly_that(self):
        plain = {s for job in A.SAMPLE_JOBS for s in job.must_have_skills + job.nice_to_have_skills
                 if not any(t in s for t in (" or ", "/", "("))}
        self.assertGreater(len(plain), 20)
        for skill in sorted(plain):
            with self.subTest(skill=skill):
                self.assertEqual(status_of([skill], skill), "met")

    def test_requirement_with_slash_is_met_by_the_same_name(self):
        # Was: "A/B testing" was split into ("a", "btesting"), so a candidate who listed it missed it.
        for name in ("A/B testing", "Pub/Sub", "Client/Server"):
            with self.subTest(name=name):
                self.assertEqual(status_of([name], name), "met")


class TestSkillMatcher(unittest.TestCase):
    def test_transitive_implication_and_evidence(self):
        m = A.SkillMatcher(["Next.js"])
        self.assertEqual(m.judge(("react",)), ("met", "Lists Next.js, which implies it"))
        self.assertEqual(m.judge(("javascript",)), ("met", "Lists Next.js, which implies it"))

    def test_implication_cycle_terminates(self):
        with mock.patch.dict(A.IMPLIES, {"cyca": ["cycb"], "cycb": ["cyca"]}):
            m = A.SkillMatcher(["cyca"])
            self.assertEqual(m.judge(("cycb",))[0], "met")

    def test_sibling_gives_partial_only_when_no_alternative_is_met(self):
        self.assertEqual(status_of(["Flask"], "FastAPI or Django")[0:7], "partial")
        self.assertEqual(status_of(["Flask", "Django"], "FastAPI or Django"), "met")

    def test_umbrella_and_named_cloud_providers(self):
        self.assertEqual(status_of(["Cloud"], "Cloud (AWS/GCP)"), "partial")
        self.assertEqual(status_of(["Azure"], "Cloud (AWS/GCP)"), "partial")
        self.assertEqual(status_of(["GCP"], "Cloud (AWS/GCP)"), "met")
        self.assertEqual(status_of(["Python"], "Cloud (AWS/GCP)"), "missing")

    def test_compound_requires_every_part(self):
        self.assertEqual(status_of(["HTML"], "HTML/CSS"), "partial")
        self.assertEqual(status_of(["CSS"], "HTML/CSS"), "partial")
        self.assertEqual(status_of(["HTML5", "CSS3"], "HTML/CSS"), "met")
        self.assertEqual(status_of(["Python"], "HTML/CSS"), "missing")

    def test_substring_traps_do_not_match(self):
        for skills, req in ((["JavaScript"], "Java"), (["Java"], "JavaScript"), (["C"], "C#"), (["C++"], "C#"),
                            (["Go"], "Golang Kubernetes"), (["SQL"], "NoSQL"), (["Node.js"], "Java")):
            with self.subTest(skills=skills, req=req):
                self.assertEqual(status_of(skills, req), "missing")

    def test_junk_in_skill_list_is_ignored(self):
        m = A.SkillMatcher(["", "   ", "Python", "python", " PYTHON "])
        self.assertEqual(set(m.listed), {"python"})
        self.assertEqual(m.judge(("python",))[0], "met")

    def test_every_verdict_carries_non_empty_evidence(self):
        m = A.SkillMatcher(["Flask", "HTML", "Cloud"])
        for alts in (("python",), ("django",), ("htmlcss",), ("~cloud",), ("rust",), ("aws", "~cloud")):
            with self.subTest(alts=alts):
                self.assertTrue(m.judge(alts)[1])

    def test_verdicts_are_cached_per_alternatives_tuple(self):
        m = A.SkillMatcher(["Python"])
        with mock.patch.object(m, "_judge_one", wraps=m._judge_one) as spy:
            for _ in range(5):
                m.judge(("python", "java"))
            self.assertEqual(spy.call_count, 2)  # once per alternative, then cached

    def test_candidate_skill_order_does_not_change_any_status(self):
        skills = ["Next.js", "React", "PostgreSQL", "Docker", "pytest", "Azure", "HTML", "CSS", "Kotlin"]
        reqs = {s for job in A.SAMPLE_JOBS for s in job.must_have_skills + job.nice_to_have_skills}
        base = A.SkillMatcher(skills)
        rng = random.Random(3)
        for _ in range(20):
            shuffled = skills[:]
            rng.shuffle(shuffled)
            other = A.SkillMatcher(shuffled)
            for r in reqs:
                self.assertEqual(base.judge(A.alternatives(r))[0], other.judge(A.alternatives(r))[0], r)


class TestScoring(unittest.TestCase):
    def test_empty_and_extremes(self):
        self.assertEqual(A.score_assessments([]), 0)
        self.assertEqual(A.score_assessments([RA("a", status="met"), RA("b", "nice-to-have", "met")]), 100)
        self.assertEqual(A.score_assessments([RA("a", status="missing")]), 0)
        self.assertEqual(A.score_assessments([RA("a", status="partial")]), 50)

    def test_weights_hand_computed(self):
        # earned = 2*1 + 2*0.5 + 1*1 + 1*0 = 4 ; total = 6  -> 66.67 -> 67
        items = [RA("a"), RA("b", status="partial"), RA("c", "nice-to-have"), RA("d", "nice-to-have", "missing")]
        self.assertEqual(A.score_assessments(items), 67)

    def test_must_have_counts_double(self):
        must_only = [RA("a", "must-have", "met"), RA("b", "nice-to-have", "missing")]
        nice_only = [RA("a", "must-have", "missing"), RA("b", "nice-to-have", "met")]
        self.assertEqual(A.score_assessments(must_only), 67)
        self.assertEqual(A.score_assessments(nice_only), 33)

    def test_score_is_bounded_and_monotone_in_status(self):
        rng = random.Random(11)
        order = ["missing", "partial", "met"]
        for _ in range(300):
            items = [RA(f"s{i}", rng.choice(["must-have", "nice-to-have"]), rng.choice(order))
                     for i in range(rng.randint(1, 12))]
            base = A.score_assessments(items)
            self.assertTrue(0 <= base <= 100)
            i = rng.randrange(len(items))
            better = items[:]
            better[i] = items[i].model_copy(update={"status": order[min(2, order.index(items[i].status) + 1)]})
            self.assertGreaterEqual(A.score_assessments(better), base)

    def test_dedupe_keeps_first_must_have_wins_and_order(self):
        out = A.dedupe_requirements(["Python", "Postgres", "python"], ["PostgreSQL", "Docker", "Docker", "Python"])
        self.assertEqual(out, [("Python", "must-have"), ("Postgres", "must-have"), ("Docker", "nice-to-have")])
        self.assertEqual(A.dedupe_requirements([], []), [])

    def test_requirements_without_latin_letters_are_kept(self):
        # Was: canon("ภาษาไทย") == "", so the requirement vanished from the score and the LLM prompt.
        out = A.dedupe_requirements(["Python", "การสื่อสาร"], ["ภาษาอังกฤษ"])
        self.assertEqual(len(out), 3)


class TestMatchSkillsNode(unittest.TestCase):
    PROFILE = A.ResumeProfile(name="Ploy", skills=["Python"])

    def run_node(self, jd, reply):
        with mock.patch.object(A, "ask", return_value=reply) as fake:
            match = A.match_skills({"resume_profile": self.PROFILE, "job_requirements": jd})["match"]
        return match, fake

    def test_extras_dropped_skipped_counted_missing_importance_and_names_forced_from_jd(self):
        jd = A.JobRequirements(title="T", must_have_skills=["Python", "Postgres"], nice_to_have_skills=["Docker"])
        reply = A.MatchAssessment(summary="s", assessments=[
            RA("python", "nice-to-have", "met", "e"),          # wrong importance, wrong case
            RA("PostgreSQL", "must-have", "partial", "e"),     # alias of the JD's "Postgres"
            RA("Rust", "must-have", "met", "invented"),         # invented
        ])
        match, _ = self.run_node(jd, reply)
        self.assertEqual([(a.skill, a.importance, a.status) for a in match.assessments],
                         [("Python", "must-have", "met"), ("Postgres", "must-have", "partial"),
                          ("Docker", "nice-to-have", "missing")])
        self.assertIn("Not assessed", match.assessments[-1].evidence)
        self.assertEqual(match.score, round(100 * (2 + 1 + 0) / 5))

    def test_prompt_lists_each_requirement_once_and_marks_no_eligibility(self):
        jd = A.JobRequirements(title="T", must_have_skills=["Postgres"], nice_to_have_skills=["PostgreSQL"])
        _, fake = self.run_node(jd, A.MatchAssessment(assessments=[], summary=""))
        user_prompt = fake.call_args.args[2]
        self.assertEqual(user_prompt.count("(must-have)"), 1)
        self.assertNotIn("(nice-to-have)", user_prompt)
        self.assertIn("Eligibility criteria:\n(none)", user_prompt)

    def test_eligibility_matching_ignores_case_and_punctuation(self):
        jd = A.JobRequirements(title="T", eligibility=["Enrolled in a CS degree."])
        reply = A.MatchAssessment(summary="", assessments=[],
                                  eligibility=[A.EligibilityCheck(criterion="enrolled in a cs degree", status="not-met", evidence="x")])
        match, _ = self.run_node(jd, reply)
        self.assertEqual((match.eligibility[0].criterion, match.eligibility[0].status),
                         ("Enrolled in a CS degree.", "not-met"))

    def test_jd_without_requirements_scores_zero_without_crashing(self):
        match, _ = self.run_node(A.JobRequirements(title="T"), A.MatchAssessment(assessments=[], summary=""))
        self.assertEqual((match.score, match.assessments, match.eligibility), (0, [], []))

    def test_thai_eligibility_criteria_stay_separate(self):
        # Was: norm() stripped Thai letters, so every Thai criterion mapped to the key "" and they all
        # received the LAST criterion's verdict.
        jd = A.JobRequirements(title="T", eligibility=["กำลังศึกษาปริญญาตรี", "พร้อมเริ่มงานฤดูร้อน"])
        reply = A.MatchAssessment(summary="", assessments=[], eligibility=[
            A.EligibilityCheck(criterion="กำลังศึกษาปริญญาตรี", status="met", evidence="Year 3"),
            A.EligibilityCheck(criterion="พร้อมเริ่มงานฤดูร้อน", status="not-met", evidence="Busy in summer"),
        ])
        match, _ = self.run_node(jd, reply)
        self.assertEqual([c.status for c in match.eligibility], ["met", "not-met"])


class TestReconcileGaps(unittest.TestCase):
    def gap(self, skill, severity="important"):
        return A.SkillGap(skill=skill, severity=severity, evidence="jd", why_it_matters="why")

    def test_one_gap_per_unmet_requirement_with_names_restored(self):
        open_items = [RA("PostgreSQL", status="missing"), RA("Docker", "nice-to-have", "partial")]
        gaps = A.reconcile_gaps(open_items, [self.gap("postgres"), self.gap("POSTGRESQL"), self.gap("Rust")], [])
        self.assertEqual([g.skill for g in gaps], ["PostgreSQL", "Docker"])

    def test_missing_must_have_is_always_blocking_even_if_llm_says_otherwise(self):
        gaps = A.reconcile_gaps([RA("SQL", status="missing")], [self.gap("SQL", "nice-to-have")], [])
        self.assertEqual(gaps[0].severity, "blocking")

    def test_skipped_requirement_is_added_back_with_default_severity(self):
        gaps = A.reconcile_gaps([RA("SQL", status="partial"), RA("Docker", "nice-to-have", "missing")], [], [])
        self.assertEqual([(g.skill, g.severity) for g in gaps], [("SQL", "important"), ("Docker", "nice-to-have")])

    def test_sorted_by_severity_and_stable_within_a_severity(self):
        items = [RA("a", "nice-to-have", "missing"), RA("b", status="missing"), RA("c", status="partial"),
                 RA("d", status="missing"), RA("e", "nice-to-have", "partial")]
        gaps = A.reconcile_gaps(items, [], [])
        self.assertEqual([g.skill for g in gaps], ["b", "d", "c", "a", "e"])

    def test_only_not_met_eligibility_blocks(self):
        checks = [A.EligibilityCheck(criterion="x", status="not-met", evidence=""),
                  A.EligibilityCheck(criterion="y", status="unclear", evidence=""),
                  A.EligibilityCheck(criterion="z", status="met", evidence="")]
        gaps = A.reconcile_gaps([], [], checks)
        self.assertEqual([g.skill for g in gaps], ["x"])

    def test_analyze_gaps_makes_no_llm_call_when_everything_is_met(self):
        match = A.MatchResult(score=100, assessments=[RA("Python")], summary="")
        with mock.patch.object(A, "ask", side_effect=AssertionError("LLM called")):
            out = A.analyze_gaps({"match": match})["gap_analysis"]
        self.assertEqual(out.gaps, [])
        self.assertTrue(out.strengths)

    def test_llm_cannot_mark_a_nice_to_have_as_blocking(self):
        # GAP_PROMPT defines blocking as "a must-have that is missing"; the code used to only
        # upgrade severity, never downgrade an invalid "blocking".
        gaps = A.reconcile_gaps([RA("Docker", "nice-to-have", "missing")], [self.gap("Docker", "blocking")], [])
        self.assertNotEqual(gaps[0].severity, "blocking")

    def test_partial_must_have_is_never_below_important(self):
        gaps = A.reconcile_gaps([RA("SQL", "must-have", "partial")], [self.gap("SQL", "nice-to-have")], [])
        self.assertIn(gaps[0].severity, ("important", "blocking"))

    def test_two_thai_eligibility_failures_are_two_gaps(self):
        checks = [A.EligibilityCheck(criterion="กำลังศึกษาปริญญาตรี", status="not-met", evidence=""),
                  A.EligibilityCheck(criterion="พร้อมเริ่มงานฤดูร้อน", status="not-met", evidence="")]
        self.assertEqual(len(A.reconcile_gaps([], [], checks)), 2)


class TestFitPlan(unittest.TestCase):
    def plan(self, *weeks):
        return [A.WeekPlan(week=w, focus=f"w{w}", tasks=[]) for w in weeks]

    def run_fit(self, plan, weeks):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            out = A.fit_plan(plan, weeks)
        return out, err.getvalue()

    def test_exact_length_is_silent_and_renumbered(self):
        out, err = self.run_fit(self.plan(0, 1, 2), 3)          # model used 0-based weeks
        self.assertEqual([w.week for w in out], [1, 2, 3])
        self.assertEqual(err, "")

    def test_too_long_is_trimmed_with_warning(self):
        out, err = self.run_fit(self.plan(1, 2, 3, 4, 5), 3)
        self.assertEqual([w.focus for w in out], ["w1", "w2", "w3"])
        self.assertIn("keeping the first weeks", err)

    def test_too_short_is_kept_as_is_with_warning(self):
        out, err = self.run_fit(self.plan(1, 2), 4)
        self.assertEqual([w.week for w in out], [1, 2])
        self.assertIn("showing what it returned", err)

    def test_empty_plan_and_duplicate_week_numbers(self):
        self.assertEqual(self.run_fit([], 3)[0], [])
        out, _ = self.run_fit(self.plan(2, 2, 1), 3)
        self.assertEqual([w.week for w in out], [1, 2, 3])


class TestJobIndexProperties(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = A.synthetic_catalog(300, seed=5)
        cls.index = A.JobIndex(cls.catalog)
        cls.vocab = sorted({s for j in A.SAMPLE_JOBS for s in j.must_have_skills + j.nice_to_have_skills})

    def oracle(self, skills):
        """Independent re-derivation of the ranking, straight from the public pieces."""
        matcher = A.SkillMatcher(skills)
        rows = []
        for i, job in enumerate(self.catalog):
            reqs = A.dedupe_requirements(job.must_have_skills, job.nice_to_have_skills)
            assessed = [RA(s, imp, matcher.judge(A.alternatives(s))[0]) for s, imp in reqs]
            missing = sum(a.importance == "must-have" and a.status == "missing" for a in assessed)
            rows.append((-A.score_assessments(assessed), missing, i, job.id))
        return [r[3] for r in sorted(rows)]

    def test_ranking_matches_independent_oracle(self):
        rng = random.Random(1)
        for _ in range(15):
            skills = rng.sample(self.vocab, rng.randint(0, 12))
            with self.subTest(skills=skills):
                self.assertEqual([m.job.id for m in self.index.rank(skills, top=None)], self.oracle(skills))

    def test_reported_score_equals_score_recomputed_from_assessments(self):
        rng = random.Random(2)
        for m in self.index.rank(rng.sample(self.vocab, 8), top=None):
            self.assertEqual(m.score, A.score_assessments(m.assessments))
            self.assertEqual(m.missing_must_haves,
                             [a.skill for a in m.assessments if a.importance == "must-have" and a.status == "missing"])
            self.assertTrue(0 <= m.score <= 100)

    def test_top_k_is_a_prefix_of_the_full_ranking(self):
        skills = self.vocab[:9]
        full = [m.job.id for m in self.index.rank(skills, top=None)]
        for k in (1, 5, 50, 300, 10_000):
            self.assertEqual([m.job.id for m in self.index.rank(skills, top=k)], full[:k])

    def test_adding_skills_never_lowers_any_score(self):
        rng = random.Random(4)
        for _ in range(10):
            small = rng.sample(self.vocab, rng.randint(1, 6))
            big = small + rng.sample(self.vocab, rng.randint(1, 6))
            before = {m.job.id: m.score for m in self.index.rank(small, top=None)}
            after = {m.job.id: m.score for m in self.index.rank(big, top=None)}
            for job_id, score in before.items():
                self.assertGreaterEqual(after[job_id], score, job_id)

    def test_skill_order_case_and_whitespace_do_not_change_scores(self):
        skills = self.vocab[:10]
        base = {m.job.id: m.score for m in self.index.rank(skills, top=None)}
        noisy = [f"  {s.upper()} " if i % 2 else s.lower() for i, s in enumerate(reversed(skills))]
        self.assertEqual({m.job.id: m.score for m in self.index.rank(noisy, top=None)}, base)

    def test_deterministic_and_input_not_mutated(self):
        skills = ["Python", "SQL"]
        before = [j.model_dump() for j in self.catalog[:20]]
        a = [m.model_dump() for m in self.index.rank(skills, top=25)]
        b = [m.model_dump() for m in self.index.rank(skills, top=25)]
        self.assertEqual(a, b)
        self.assertEqual(skills, ["Python", "SQL"])
        self.assertEqual([j.model_dump() for j in self.catalog[:20]], before)

    def test_empty_skills_empty_catalog_and_requirementless_job(self):
        self.assertTrue(all(m.score == 0 for m in self.index.rank([], top=None)))
        self.assertEqual(A.JobIndex([]).rank(["Python"], top=5), [])
        bare = A.JobPosting(id="bare", title="Bare")
        self.assertEqual(A.JobIndex([bare]).rank(["Python"])[0].score, 0)

    def test_perfect_candidate_scores_100(self):
        job = A.JobPosting(id="j", title="J", must_have_skills=["Python", "SQL"], nice_to_have_skills=["Docker"])
        self.assertEqual(A.JobIndex([job]).rank(["Python", "PostgreSQL", "Docker"])[0].score, 100)

    def test_ties_resolve_to_catalog_order(self):
        jobs = [A.JobPosting(id=f"j{i}", title="J", must_have_skills=["Python"]) for i in range(5)]
        self.assertEqual([m.job.id for m in A.JobIndex(jobs).rank(["Python"], top=None)], [f"j{i}" for i in range(5)])


class TestFindJobs(unittest.TestCase):
    def fake_match_factory(self, scores):
        def fake(state):
            title = state["job_requirements"].title
            return {"match": A.MatchResult(score=scores.get(title, 10), assessments=[], summary="fake")}
        return fake

    def test_top_larger_than_catalog_returns_everything(self):
        self.assertEqual(len(A.find_jobs(A.SAMPLE_SKILLS, A.SAMPLE_JOBS, top=100)), len(A.SAMPLE_JOBS))

    def test_rerank_can_promote_a_job_from_outside_the_quick_top(self):
        quick = A.find_jobs(A.SAMPLE_SKILLS, A.SAMPLE_JOBS, top=6)
        promoted = quick[3].job  # quick rank 4
        fake = self.fake_match_factory({promoted.title: 99})
        with mock.patch.object(A, "match_skills", side_effect=fake):
            result = A.find_jobs(A.SAMPLE_SKILLS, A.SAMPLE_JOBS, top=2, rerank=4)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].job.id, promoted.id)
        self.assertEqual(result[0].llm_score, 99)

    def test_rerank_larger_than_catalog_checks_each_job_once(self):
        with mock.patch.object(A, "match_skills", side_effect=self.fake_match_factory({})) as fake:
            result = A.find_jobs(A.SAMPLE_SKILLS, A.SAMPLE_JOBS, top=3, rerank=50)
        self.assertEqual(fake.call_count, len(A.SAMPLE_JOBS))
        self.assertEqual(len(result), 3)

    def test_timings_are_reported(self):
        t = {}
        A.find_jobs(A.SAMPLE_SKILLS, A.SAMPLE_JOBS, timings=t)
        self.assertIn("rank_ms", t)
        self.assertNotIn("reranked", t)

    def test_non_quota_llm_errors_are_not_swallowed(self):
        with mock.patch.object(A, "match_skills", side_effect=A.ModelAuthenticationError("bad key")):
            with self.assertRaises(A.ModelAuthenticationError):
                A.find_jobs(A.SAMPLE_SKILLS, A.SAMPLE_JOBS, rerank=2)

    def test_quota_error_reports_no_rerank_in_timings(self):
        t = {}
        with mock.patch.object(A, "match_skills", side_effect=A.ModelRateLimitError("q")), \
                contextlib.redirect_stderr(io.StringIO()):
            A.find_jobs(A.SAMPLE_SKILLS, A.SAMPLE_JOBS, rerank=3, timings=t)
        self.assertNotIn("reranked", t)


class TestCareerRanker(unittest.TestCase):
    """--ranking career: the score is a weighted sum of four components the breakdown reports."""

    SKILLS = ["Python", "PostgreSQL", "Docker", "FastAPI", "Git"]

    @classmethod
    def setUpClass(cls):
        cls.ranker = A.CareerRanker(A.SAMPLE_JOBS)
        cls.full = cls.ranker.rank(cls.SKILLS, top=None)

    def test_default_ranking_is_still_skills_only(self):
        default = A.find_jobs(self.SKILLS, A.SAMPLE_JOBS, top=5)
        skills = A.JobIndex(A.SAMPLE_JOBS).rank(self.SKILLS, top=5)
        self.assertEqual([(m.job.id, m.score) for m in default], [(m.job.id, m.score) for m in skills])
        self.assertFalse(any(isinstance(m, A.CareerMatch) for m in default))

    def test_breakdown_is_bounded_and_adds_up_to_the_score(self):
        w = A.DEFAULT_RANKING_WEIGHTS.normalized()
        for m in self.full:
            with self.subTest(job=m.job.id):
                b = m.career
                for part in (b.technical, b.domain, b.experience, b.transferable):
                    self.assertTrue(0 <= part <= 1, b)
                weighted = w.technical * b.technical + w.domain * b.domain + w.experience * b.experience \
                    + w.transferable * b.transferable
                self.assertAlmostEqual(weighted, b.total, delta=0.002)  # components are rounded to 3 places
                self.assertEqual(m.score, round(100 * b.total))

    def test_skill_score_matches_the_skills_ranker(self):
        skill_scores = {m.job.id: m.score for m in A.JobIndex(A.SAMPLE_JOBS).rank(self.SKILLS, top=None)}
        for m in self.full:
            self.assertEqual(m.career.skill_score, skill_scores[m.job.id], m.job.id)

    def test_full_ranking_is_sorted_complete_and_top_k_is_its_prefix(self):
        self.assertEqual(len(self.full), len(A.SAMPLE_JOBS))
        self.assertEqual(sorted({m.job.id for m in self.full}), sorted(j.id for j in A.SAMPLE_JOBS))
        totals = [m.career.total for m in self.full]
        self.assertEqual(totals, sorted(totals, reverse=True))
        for k in (1, 3, 7):
            self.assertEqual([m.job.id for m in self.ranker.rank(self.SKILLS, top=k)],
                             [m.job.id for m in self.full[:k]])

    def test_skill_order_and_case_do_not_change_the_ranking(self):
        shuffled = [s.upper() for s in reversed(self.SKILLS)]
        self.assertEqual([(m.job.id, m.score) for m in self.ranker.rank(shuffled, top=None)],
                         [(m.job.id, m.score) for m in self.full])

    def test_single_component_weights_rank_by_that_component(self):
        for name in ("technical", "domain", "experience", "transferable"):
            with self.subTest(weight=name):
                ranker = A.CareerRanker(A.SAMPLE_JOBS, A.RankingWeights.parse(
                    ",".join(f"{n}={1 if n == name else 0}" for n in A.RankingWeights.model_fields)))
                for m in ranker.rank(self.SKILLS, top=None):
                    self.assertAlmostEqual(m.career.total, getattr(m.career, name), delta=0.001)

    def test_adding_a_project_never_lowers_experience_and_leaves_technical_fit_alone(self):
        # Was: a profile with no applied evidence fell back to skill depth, which could be higher than the
        # blend used once any evidence existed, so adding a project to a resume LOWERED experience
        # (backend-fintech: 0.851 -> 0.778).
        bare = A.ResumeProfile(name="Ploy", skills=self.SKILLS)
        with_project = bare.model_copy(update={"projects": [
            A.Project(name="Wallet API", description="Built a REST API with FastAPI and PostgreSQL in Docker",
                      skills_used=["FastAPI", "PostgreSQL", "Docker"])]})
        before = {m.job.id: m.career for m in self.ranker.rank(self.SKILLS, top=None, profile=bare)}
        after = {m.job.id: m.career for m in self.ranker.rank(self.SKILLS, top=None, profile=with_project)}
        for job_id, career in after.items():
            with self.subTest(job=job_id):
                self.assertEqual(career.technical, before[job_id].technical)
                self.assertGreaterEqual(career.experience, before[job_id].experience)
        self.assertGreater(after["backend-fintech"].experience, before["backend-fintech"].experience)

    def test_odd_inputs_do_not_crash(self):
        for skills in ([], ["ภาษาไทย"], ["🚀"], ["x" * 10_000]):
            with self.subTest(skills=skills[:1]):
                out = self.ranker.rank(skills, top=3)
                self.assertEqual(len(out), 3)
                self.assertTrue(all(0 <= m.score <= 100 for m in out))
        self.assertEqual(A.CareerRanker([]).rank(self.SKILLS, top=5), [])

    def test_weights_parse(self):
        defaults = A.DEFAULT_RANKING_WEIGHTS.normalized()
        self.assertEqual(A.RankingWeights.parse("40,25,20,15").normalized(), defaults)
        self.assertEqual(A.RankingWeights.parse(" 0.4 , 0.25,0.2,0.15 ").normalized(), defaults)
        self.assertEqual(A.RankingWeights.parse("technical=1").domain, A.DEFAULT_RANKING_WEIGHTS.domain)
        for bad in ("1,2,3", "1,2,3,4,5", "speed=1", "0,0,0,0", "-1,1,1,1", "nan,1,1,1", "inf,1,1,1",
                    "technical=x", ""):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                A.RankingWeights.parse(bad)

    def test_find_jobs_validates_ranking(self):
        with self.assertRaises(ValueError):
            A.find_jobs(self.SKILLS, A.SAMPLE_JOBS, ranking="vibes")
        career = A.find_jobs(self.SKILLS, A.SAMPLE_JOBS, top=3, ranking="career")
        self.assertEqual([m.job.id for m in career], [m.job.id for m in self.full[:3]])

    def test_cli_career_ranking_and_bad_weights(self):
        r = cli("--find-jobs", "--skills", "Python, SQL", "--ranking", "career", "--top", "2", "--json")
        self.assertEqual(r.returncode, 0, r.stderr)
        data = json.loads(r.stdout)
        self.assertEqual(len(data), 2)
        self.assertIn("career", data[0])
        bad = cli("--find-jobs", "--skills", "Python", "--ranking", "career", "--weights", "1,2")
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("bad weights", bad.stderr)


class TestInputsAndIO(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="agent_io_"))

    def test_read_text_utf8_file_with_thai(self):
        p = self.tmp / "cv.txt"
        p.write_text("สมชาย Python นักศึกษา", encoding="utf-8")
        self.assertEqual(A.read_text(str(p)), "สมชาย Python นักศึกษา")

    def test_read_text_raw_string_passthrough(self):
        self.assertEqual(A.read_text("I know Python and SQL"), "I know Python and SQL")

    def test_read_text_pdf(self):
        p = self.tmp / "cv.PDF"
        p.write_bytes(self.make_pdf("Python and SQL developer"))
        self.assertIn("Python and SQL developer", A.read_text(str(p)))

    @staticmethod
    def make_pdf(text):
        content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
                b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
                b"/Resources << /Font << /F1 5 0 R >> >> >>",
                b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
                b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
        out, offsets = b"%PDF-1.4\n", []
        for i, obj in enumerate(objs, 1):
            offsets.append(len(out))
            out += b"%d 0 obj\n" % i + obj + b"\nendobj\n"
        xref = len(out)
        out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
        out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
        out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
        return out

    def test_mistyped_resume_path_is_reported(self):
        # Was: a typo such as "resumee.pdf" was silently analyzed as the literal text "resumee.pdf".
        with self.assertRaises((FileNotFoundError, ValueError)):
            A.read_text(str(self.tmp / "resumee.pdf"))

    def test_ask_length_guard_boundary(self):
        fake_llm = mock.Mock()
        fake_llm.with_structured_output.return_value.invoke.return_value = "ok"
        with mock.patch.object(A, "get_llm", return_value=fake_llm):
            self.assertEqual(A.ask(A.ResumeProfile, "sys", "x" * A.MAX_INPUT_CHARS), "ok")
            with self.assertRaises(A.InputTooLongError):
                A.ask(A.ResumeProfile, "sys", "x" * (A.MAX_INPUT_CHARS + 1))
        messages = fake_llm.with_structured_output.return_value.invoke.call_args.args[0]
        self.assertEqual([type(m).__name__ for m in messages], ["SystemMessage", "HumanMessage"])

    def write_jobs(self, payload):
        p = self.tmp / "jobs.json"
        p.write_text(payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return str(p)

    def test_read_job_file_validation(self):
        self.assertEqual(A.read_job_file(self.write_jobs([{"title": "A"}])), [{"title": "A"}])
        for bad in ({"title": "A"}, [1, 2], ["x"], "null"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                A.read_job_file(self.write_jobs(bad))
        with self.assertRaises(json.JSONDecodeError):
            A.read_job_file(self.write_jobs("{not json"))

    def test_needs_parsing_rules(self):
        self.assertTrue(A.needs_parsing({"description": "x"}))
        self.assertFalse(A.needs_parsing({"description": "x", "must_have_skills": ["Go"]}))
        self.assertFalse(A.needs_parsing({"description": "x", "nice_to_have_skills": ["Go"]}))
        self.assertFalse(A.needs_parsing({"description": ""}))
        self.assertFalse(A.needs_parsing({"title": "T"}))
        self.assertTrue(A.needs_parsing({"description": "x", "must_have_skills": []}))

    def test_load_jobs_ids_fallbacks_and_unknown_keys(self):
        jd = A.JobRequirements(title="Parsed title", company="Parsed co", must_have_skills=["Python"])
        items = [{"id": 7, "title": "T1", "must_have_skills": ["Go"], "salary": "9999"},
                 {"description": "raw text"},
                 {"title": "Mine", "company": "Mine co", "description": "raw text"}]
        with mock.patch.object(A, "analyze_job", return_value={"job_requirements": jd}) as parse:
            jobs = A.load_jobs(items)
        self.assertEqual(parse.call_count, 2)
        self.assertEqual([j.id for j in jobs], ["7", "job-2", "job-3"])
        self.assertEqual([j.title for j in jobs], ["T1", "Parsed title", "Mine"])
        self.assertEqual([j.company for j in jobs], [None, "Parsed co", "Mine co"])
        self.assertFalse(hasattr(jobs[0], "salary"))

    def test_load_jobs_does_not_mutate_input(self):
        items = [{"title": "T", "must_have_skills": ["Go"]}]
        A.load_jobs(items)
        self.assertEqual(items, [{"title": "T", "must_have_skills": ["Go"]}])

    def test_posting_without_title_or_description_is_rejected(self):
        with self.assertRaises(Exception):
            A.load_jobs([{"must_have_skills": ["Go"]}])


class TestPlannedCallsAndFindJobsRunner(unittest.TestCase):
    def args(self, **kw):
        base = dict(resume=None, skills=None, rerank=0, jobs=None, top=5, max_llm_calls=10, json=False)
        base.update(kw)
        return argparse.Namespace(**base)

    def test_planned_llm_calls_matrix(self):
        items = [{"description": "d"}, {"title": "x", "must_have_skills": ["Go"]}]
        self.assertEqual(A.planned_llm_calls(self.args(), None), 0)
        self.assertEqual(A.planned_llm_calls(self.args(resume="cv"), None), 1)
        self.assertEqual(A.planned_llm_calls(self.args(resume="cv", skills="Go"), None), 0)
        self.assertEqual(A.planned_llm_calls(self.args(rerank=4), items), 5)

    def run_find(self, **kw):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            A.run_find_jobs(self.args(**kw))
        return out.getvalue(), err.getvalue()

    def test_over_budget_exits_before_any_llm_call(self):
        with mock.patch.object(A, "ask", side_effect=AssertionError("LLM called")), \
                mock.patch.object(A, "analyze_resume", side_effect=AssertionError("LLM called")):
            with self.assertRaises(SystemExit) as ctx:
                self.run_find(resume="cv", rerank=12)
        self.assertIn("--max-llm-calls", str(ctx.exception))

    def test_skills_are_split_trimmed_and_blank_entries_dropped(self):
        out, _ = self.run_find(skills="Python, ,SQL ,  Git", json=True)
        data = json.loads(out)
        expected = [(m.job.id, m.score) for m in A.find_jobs(["Python", "SQL", "Git"], A.SAMPLE_JOBS, top=5)]
        self.assertEqual(len(data), 5)
        self.assertEqual([(d["job"]["id"], d["score"]) for d in data], expected)

    def test_json_mode_stdout_is_pure_json_and_progress_goes_to_stderr(self):
        out, err = self.run_find(json=True)  # no skills: falls back to the sample resume
        json.loads(out)
        self.assertIn("sample resume", err)

    def test_text_report_mentions_count_and_timing(self):
        out, _ = self.run_find(skills="Python, SQL", top=3)
        self.assertIn("Top 3 of 14 jobs", out)
        self.assertIn("Ranked 14 jobs in", out)

    def test_search_with_only_blank_skills_is_refused(self):
        # Was: "--skills ','" yielded [] and printed a meaningless ranking of 0/100 jobs.
        with self.assertRaises(SystemExit):
            self.run_find(skills=" , ")


class TestLlmErrorMapping(unittest.TestCase):
    def raises(self, exc):
        def boom():
            raise exc
        with self.assertRaises(SystemExit) as ctx:
            A.run_with_llm_errors("m-1", boom)
        return str(ctx.exception)

    def test_each_error_has_a_one_line_message(self):
        self.assertIn("quota exceeded for m-1", self.raises(A.ModelRateLimitError("x")))
        self.assertIn("rejected the API key", self.raises(A.ModelAuthenticationError("x")))
        self.assertIn("rejected the API key", self.raises(A.ModelInvalidRequestError("... API_KEY_INVALID ...")))
        self.assertIn("'m-1' not found", self.raises(A.ModelNotFoundError("x")))
        self.assertIn("characters", self.raises(A.InputTooLongError("Input is 5 characters")))

    def test_other_invalid_requests_are_not_masked_as_bad_keys(self):
        with self.assertRaises(A.ModelInvalidRequestError):
            A.run_with_llm_errors("m", mock.Mock(side_effect=A.ModelInvalidRequestError("schema problem")))

    def test_success_passes_value_through(self):
        self.assertEqual(A.run_with_llm_errors("m", lambda a, b: a + b, 2, 3), 5)

    def test_int_at_least(self):
        parse = A.int_at_least(0)
        self.assertEqual(parse(" 3 "), 3)
        self.assertEqual(parse("0"), 0)
        for bad in ("-1", "abc", "3.5", ""):
            with self.subTest(bad=bad), self.assertRaises(argparse.ArgumentTypeError):
                parse(bad)


class TestPipelineEndToEnd(unittest.TestCase):
    """Full 5-node graph with the LLM replaced by a scripted fake."""

    def make_fake(self, calls, lock):
        def fake(schema, system, user):
            with lock:
                calls.append(schema.__name__)
            if schema is A.ResumeProfile:
                return A.ResumeProfile(name="Ploy", skills=["Python", "SQL", "Git"])
            if schema is A.JobRequirements:
                return A.JobRequirements(title="Backend Intern", company="Acme", eligibility=["Enrolled in CS"],
                                         must_have_skills=["Python", "SQL", "Docker"],
                                         nice_to_have_skills=["Postgres"], responsibilities=["Build APIs"])
            if schema is A.MatchAssessment:
                return A.MatchAssessment(summary="Decent fit.", assessments=[
                    RA("Python", "must-have", "met", "e"), RA("SQL", "must-have", "met", "e"),
                    RA("Docker", "must-have", "missing", "e"), RA("Postgres", "nice-to-have", "partial", "e")],
                    eligibility=[A.EligibilityCheck(criterion="Enrolled in CS", status="met", evidence="Year 3")])
            if schema is A.GapAnalysis:
                return A.GapAnalysis(strengths=["Python"], gaps=[
                    A.SkillGap(skill="Docker", severity="nice-to-have", evidence="jd", why_it_matters="deploys")])
            if schema is A.Recommendation:
                return A.Recommendation(
                    learning_plan=[A.WeekPlan(week=w, focus=f"Week {w} topic", tasks=["t"]) for w in (1, 2, 3, 4)],
                    project_ideas=[A.ProjectIdea(title="P", description="d", skills_covered=["Docker"])],
                    resume_improvements=["Quantify results"],
                    interview_questions=[A.InterviewQuestion(question="Q?", why_asked="w", how_to_prepare="h")])
            raise AssertionError(schema)
        return fake

    def run_graph(self, weeks=3):
        calls, lock = [], threading.Lock()
        with mock.patch.object(A, "ask", side_effect=self.make_fake(calls, lock)):
            result = A.build_graph().invoke({"resume_text": "cv", "job_description": "jd", "weeks": weeks})
        return result, calls

    def test_state_flows_through_all_five_stages(self):
        result, calls = self.run_graph(weeks=3)
        self.assertEqual(sorted(calls[:2]), ["JobRequirements", "ResumeProfile"])
        self.assertEqual(calls[2:], ["MatchAssessment", "GapAnalysis", "Recommendation"])
        # total = 2+2+2+1 = 7 ; earned = 2+2+0+0.5 = 4.5 -> 64
        self.assertEqual(result["match"].score, 64)
        gaps = {g.skill: g.severity for g in result["gap_analysis"].gaps}
        self.assertEqual(gaps, {"Docker": "blocking", "Postgres": "nice-to-have"})
        self.assertEqual([w.week for w in result["recommendation"].learning_plan], [1, 2, 3])

    def test_report_renders_without_error(self):
        result, _ = self.run_graph(weeks=3)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            A.print_report(result)
        text = out.getvalue()
        self.assertIn("MATCH SCORE: 64/100", text)
        self.assertIn("Week 3", text)
        self.assertNotIn("Week 4", text)
        self.assertIn("Eligibility (not part of the score)", text)

    def test_main_json_output_is_parseable_and_complete(self):
        calls, lock = [], threading.Lock()
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(A, "ask", side_effect=self.make_fake(calls, lock)), \
                mock.patch.object(sys, "argv", ["agent.py", "--json", "--weeks", "3"]), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            A.main()
        data = json.loads(out.getvalue())
        self.assertEqual(set(data), {"resume_profile", "job_requirements", "match", "gap_analysis", "recommendation"})
        self.assertEqual(len(calls), 5)
        self.assertIn("Analyzing with", err.getvalue())

    def test_report_handles_minimal_results(self):
        result = {
            "resume_profile": A.ResumeProfile(name="X"), "job_requirements": A.JobRequirements(title="T"),
            "match": A.MatchResult(score=0, assessments=[], summary=""),
            "gap_analysis": A.GapAnalysis(),
            "recommendation": A.Recommendation(learning_plan=[], project_ideas=[], resume_improvements=[],
                                               interview_questions=[]),
        }
        with contextlib.redirect_stdout(io.StringIO()):
            A.print_report(result)


class TestCli(unittest.TestCase):
    def test_find_jobs_json_via_real_process(self):
        r = cli("--find-jobs", "--skills", "Python, SQL, Git", "--json")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(len(json.loads(r.stdout)), 5)

    def test_non_utf8_console_does_not_crash_on_emoji(self):
        r = cli("--find-jobs", "--skills", "Python", env_extra={"PYTHONIOENCODING": "ascii", "LC_ALL": "C"})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("Top 5 of 14", r.stdout)

    def test_argument_validation_exit_codes(self):
        for args in (("--top", "0"), ("--weeks", "0"), ("--rerank", "-1"), ("--find-jobs", "--eval")):
            with self.subTest(args=args):
                self.assertEqual(cli(*args).returncode, 2)

    def test_over_budget_refusal_needs_no_api_key(self):
        jobs = Path(tempfile.mkdtemp()) / "jobs.json"
        jobs.write_text(json.dumps([{"title": "T", "description": "d"}]), encoding="utf-8")
        r = cli("--find-jobs", "--skills", "Python", "--jobs", str(jobs), "--max-llm-calls", "0")
        self.assertEqual(r.returncode, 1)
        self.assertIn("needs 1 LLM calls", r.stderr)
        self.assertNotIn("Traceback", r.stderr)

    def test_eval_command_runs_and_prints_both_levels(self):
        r = cli("--eval")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("REQUIREMENT ACCURACY", r.stdout)
        self.assertIn("RANKING QUALITY", r.stdout)

    def test_built_in_self_test_passes(self):
        if SKIP_PERF:
            self.skipTest("SKIP_PERF=1")
        r = cli("--self-test", timeout=300)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        expected = unittest.defaultTestLoader.loadTestsFromModule(A).countTestCases()
        self.assertIn(f"Ran {expected} tests", r.stderr)


class TestBenchmarkHelpers(unittest.TestCase):
    def test_synthetic_catalog_is_deterministic_unique_and_well_formed(self):
        a, b = A.synthetic_catalog(500, seed=1), A.synthetic_catalog(500, seed=1)
        self.assertEqual([j.model_dump() for j in a], [j.model_dump() for j in b])
        self.assertNotEqual([j.model_dump() for j in a[:50]], [j.model_dump() for j in A.synthetic_catalog(500, seed=2)[:50]])
        self.assertEqual(len({j.id for j in a}), 500)
        for j in a:
            self.assertTrue(2 <= len(j.must_have_skills) <= 5 and 1 <= len(j.nice_to_have_skills) <= 4)

    def test_synthetic_catalog_alternative_ratio_matches_generator_probability(self):
        # 15% of requirements are generated as "X or Y". The other 85% come from a vocabulary in which
        # some entries already contain " or " (e.g. "Kotlin or Java"), so expect 0.15 + 0.85 * base.
        catalog = A.synthetic_catalog(4000, seed=9)
        reqs = [s for j in catalog for s in j.must_have_skills + j.nice_to_have_skills]
        ratio = sum(" or " in s for s in reqs) / len(reqs)
        base = sum(" or " in v for v in VOCAB) / len(VOCAB)
        self.assertAlmostEqual(ratio, 0.15 + 0.85 * base, delta=0.02)

    def test_benchmark_result_shape(self):
        for ranking in A.RANKING_MODES:
            with self.subTest(ranking=ranking):
                r = A.benchmark(sizes=(200,), queries=5, ranking=ranking)[0]
                self.assertEqual(set(r), {"ranking", "jobs", "build_ms", "p50_ms", "p95_ms", "jobs_per_sec"})
                self.assertEqual(r["ranking"], ranking)
                self.assertGreater(r["jobs_per_sec"], 0)
                self.assertLessEqual(r["p50_ms"], r["p95_ms"] + 1e-9)

    # statistics.quantiles() raises StatisticsError for a single sample before Python 3.13, which used to
    # crash benchmark(queries=1) on older interpreters (the README benchmarks on 3.10).
    def test_benchmark_with_a_single_query(self):
        A.benchmark(sizes=(50,), queries=1)

    def test_print_benchmark_and_print_eval_do_not_crash(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            A.print_benchmark(A.benchmark(sizes=(100,), queries=3))
            A.print_eval(A.evaluate())
        self.assertIn("JOB SEARCH BENCHMARK", out.getvalue())


# ===========================================================================
# 2. ACCURACY (independent held-out labels)
# ===========================================================================

# (candidate skills, requirement, status a careful human reviewer would assign)
# Written independently of agent.REQUIREMENT_CASES. Deliberately includes realistic spellings the
# tables may not know ("Python 3.11", "T-SQL", "Docker Compose") so the number reflects real input.
HELD_OUT_REQUIREMENTS = [
    # --- met: aliases / spellings
    (["JS"], "JavaScript", "met"), (["TS"], "TypeScript", "met"), (["Golang"], "Go", "met"),
    (["Amazon Web Services"], "AWS", "met"), (["Google Cloud Platform"], "GCP", "met"),
    (["K8s"], "Kubernetes", "met"), (["PySpark"], "Apache Spark", "met"), (["Vue.js"], "Vue", "met"),
    (["Scikit Learn"], "scikit-learn", "met"), (["Microsoft Excel"], "Excel", "met"),
    (["Power BI"], "PowerBI", "met"), (["PostgreSQL"], "Postgres", "met"), (["REST"], "REST API design", "met"),
    (["Machine learning"], "ML", "met"), (["python"], "PYTHON", "met"), ([" Python "], "Python", "met"),
    (["C#"], "C#", "met"),
    # --- met: implication
    (["Pandas"], "Python", "met"), (["Express.js"], "Node.js", "met"), (["React"], "JavaScript", "met"),
    (["TensorFlow"], "Deep Learning", "met"), (["Playwright"], "Test automation", "met"),
    (["MariaDB"], "SQL", "met"), (["SQL Server"], "SQL", "met"), (["GitLab"], "Git", "met"),
    (["Jenkins"], "CI/CD", "met"), (["NLP"], "Machine Learning", "met"), (["Deep Learning"], "Machine Learning", "met"),
    # --- met: wording around the skill
    (["Python"], "Proficiency in Python", "met"), (["SQL"], "Solid understanding of SQL databases", "met"),
    (["Docker"], "Familiarity with Docker", "met"), (["Java"], "Good knowledge of Java programming", "met"),
    (["React"], "Basic React", "met"),
    # --- met: alternatives
    (["Vue"], "React or Vue", "met"), (["Azure"], "AWS/GCP/Azure", "met"), (["Rust"], "Go or Rust", "met"),
    (["HTML5", "CSS3"], "HTML/CSS", "met"),
    # --- partial
    (["Vue"], "React", "partial"), (["SQLite"], "PostgreSQL", "partial"), (["GCP"], "AWS", "partial"),
    (["Cypress"], "Selenium", "partial"), (["Looker"], "Tableau", "partial"), (["Godot"], "Unity", "partial"),
    (["TensorFlow"], "PyTorch", "partial"), (["DynamoDB"], "MongoDB", "partial"), (["Django"], "Flask", "partial"),
    (["CSS"], "HTML/CSS", "partial"), (["C"], "C++", "partial"), (["Kotlin"], "Java or Swift", "partial"),
    (["MySQL"], "PostgreSQL or SQLite", "partial"),
    # --- missing
    (["Python"], "Rust", "missing"), (["Java"], "Python", "missing"), (["SQL"], "MongoDB", "missing"),
    (["MATLAB"], "Python", "missing"), (["R"], "Python", "missing"), (["Tableau"], "Excel", "missing"),
    (["Linux"], "Windows", "missing"), (["iOS"], "Android", "missing"), (["Swift"], "Kotlin", "missing"),
    (["Docker"], "Terraform", "missing"), (["Photoshop"], "Figma", "missing"), (["Python"], "Java or C++", "missing"),
    (["C++"], "C#", "missing"),
    # --- realistic spellings outside the tables (expected to expose vocabulary gaps)
    (["Python 3.11"], "Python", "met"), (["Java 17"], "Java", "met"), (["T-SQL"], "SQL", "met"),
    (["Bash scripting"], "Bash", "met"), (["Docker Compose"], "Docker", "met"), (["Spring Boot"], "Spring", "met"),
]

HELD_OUT_CANDIDATES = {
    "web-fullstack": (["JavaScript", "Node.js", "Express", "MongoDB", "React", "Git", "HTML", "CSS"],
                      {"frontend-ecommerce": 2, "fullstack": 2, "ux-ui": 1}),
    "analyst": (["SQL", "Excel", "Tableau", "Python", "Statistics"],
                {"data-analyst": 2, "data-engineer": 1, "backend-fintech": 1}),
    "security-student": (["Networking", "Linux", "Wireshark", "Python", "Penetration testing"],
                         {"security": 2, "devops": 1}),
    "qa": (["Selenium", "Python", "Software testing", "Git", "CI/CD", "SQL"],
           {"qa-automation": 2, "backend-fintech": 1}),
    "ios-dev": (["Swift", "SwiftUI", "iOS SDK", "Git", "Firebase"], {"ios": 2, "android": 1}),
    "ml-researcher": (["Python", "PyTorch", "NLP", "Machine Learning", "Linux", "Docker"],
                      {"ml-engineer": 2, "data-analyst": 1}),
    "devops-junior": (["Linux", "Bash", "Docker", "Git"], {"devops": 2}),
    "game-student": (["C#", "Unity", "Git"], {"game-dev": 2}),
    "embedded-student": (["C", "C++", "STM32", "FreeRTOS", "Python"], {"embedded": 2}),
}


class TestHeldOutAccuracy(unittest.TestCase):
    def test_label_sanity(self):
        ids = {j.id for j in A.SAMPLE_JOBS}
        for name, (_, labels) in HELD_OUT_CANDIDATES.items():
            self.assertTrue(set(labels) <= ids, name)
            self.assertIn(2, labels.values(), name)
        self.assertTrue(all(s in ("met", "partial", "missing") for _, _, s in HELD_OUT_REQUIREMENTS))
        self.assertEqual(len({(tuple(s), r) for s, r, _ in HELD_OUT_REQUIREMENTS}), len(HELD_OUT_REQUIREMENTS))
        in_repo = {(tuple(s), r) for s, r, _ in A.REQUIREMENT_CASES}
        self.assertFalse(in_repo & {(tuple(s), r) for s, r, _ in HELD_OUT_REQUIREMENTS}, "held-out set must not overlap")

    def test_requirement_accuracy_on_held_out_cases(self):
        misses, confusion = [], {}
        for skills, requirement, expected in HELD_OUT_REQUIREMENTS:
            got = status_of(skills, requirement)
            confusion[(expected, got)] = confusion.get((expected, got), 0) + 1
            if got != expected:
                misses.append((skills, requirement, expected, got))
        n = len(HELD_OUT_REQUIREMENTS)
        accuracy = 1 - len(misses) / n
        baseline = statistics.mean(
            ("met" if r.strip().lower() in {s.strip().lower() for s in skills} else "missing") == e
            for skills, r, e in HELD_OUT_REQUIREMENTS)
        PERF_REPORT.append((f"held-out requirement accuracy ({n} cases)", f"{accuracy:.1%} (exact-name baseline {baseline:.1%})"))
        for (e, g), c in sorted(confusion.items()):
            if e != g:
                PERF_REPORT.append((f"  confusion: expected {e} -> got {g}", str(c)))
        for skills, requirement, expected, got in misses:
            PERF_REPORT.append((f"  MISS {skills} vs {requirement!r}", f"expected {expected}, got {got}"))
        self.assertGreaterEqual(accuracy, 0.85, misses)
        self.assertGreater(accuracy, baseline + 0.30)

    def test_zero_false_met_on_strict_negatives(self):
        # A false "met" is the costly error: it tells the applicant they qualify when they do not.
        false_met = [(s, r) for s, r, e in HELD_OUT_REQUIREMENTS if e == "missing" and status_of(s, r) == "met"]
        self.assertEqual(false_met, [])
        false_met_partial = [(s, r) for s, r, e in HELD_OUT_REQUIREMENTS if e == "partial" and status_of(s, r) == "met"]
        self.assertEqual(false_met_partial, [])

    RANKERS = {"skills": A.JobIndex, "career": A.CareerRanker}

    def test_ranking_quality_on_held_out_candidates(self):
        mean = lambda rows, k: statistics.mean(r[k] for r in rows)  # noqa: E731
        rows_b = [A.ranking_metrics(A.baseline_rank(A.SAMPLE_JOBS, skills), labels)
                  for skills, labels in HELD_OUT_CANDIDATES.values()]
        rows = {}
        for ranker, cls in self.RANKERS.items():
            index = cls(A.SAMPLE_JOBS)
            rows[ranker] = []
            for name, (skills, labels) in HELD_OUT_CANDIDATES.items():
                ranked = [m.job.id for m in index.rank(skills, top=None)]
                rows[ranker].append(A.ranking_metrics(ranked, labels))
                PERF_REPORT.append((f"  {ranker} {name} top-3", ", ".join(ranked[:3])))
        for metric in rows_b[0]:
            PERF_REPORT.append((f"held-out ranking {metric}", f"skills {mean(rows['skills'], metric):.2f} / "
                                f"career {mean(rows['career'], metric):.2f} / baseline {mean(rows_b, metric):.2f}"))
        # The exact-name baseline already scores Top-1 1.00 here, so these labels are easy: they catch
        # regressions, they do not show which ranker is better.
        for ranker, ranker_rows in rows.items():
            with self.subTest(ranker=ranker):
                self.assertGreaterEqual(mean(ranker_rows, "Top-1"), 0.80)
                self.assertGreaterEqual(mean(ranker_rows, "NDCG@5"), 0.85)
                self.assertGreaterEqual(mean(ranker_rows, "Recall@5"), 0.85)
                self.assertGreaterEqual(mean(ranker_rows, "NDCG@5"), mean(rows_b, "NDCG@5"))

    def test_strong_fit_is_never_ranked_below_a_poor_fit(self):
        for ranker, cls in self.RANKERS.items():
            index = cls(A.SAMPLE_JOBS)
            for name, (skills, labels) in HELD_OUT_CANDIDATES.items():
                order = [m.job.id for m in index.rank(skills, top=None)]
                for strong in (j for j, g in labels.items() if g == 2):
                    worse_than = [j for j in order[:order.index(strong)] if labels.get(j, 0) == 0]
                    self.assertLessEqual(len(worse_than), 2,
                                         f"{ranker}/{name}: {strong} sits below poor fits {worse_than}")


class TestRankingMetrics(unittest.TestCase):
    def test_perfect_ranking(self):
        m = A.ranking_metrics(["a", "b", "c"], {"a": 2, "b": 1})
        self.assertEqual((m["Top-1"], m["MRR"], m["Recall@5"]), (1.0, 1.0, 1.0))
        self.assertAlmostEqual(m["NDCG@5"], 1.0)

    def test_hand_computed_example(self):
        m = A.ranking_metrics(["a", "b", "c"], {"b": 2, "c": 1})
        dcg = 3 / math.log2(3) + 1 / math.log2(4)
        ideal = 3 / math.log2(2) + 1 / math.log2(3)
        self.assertEqual(m["Top-1"], 0.0)
        self.assertEqual(m["MRR"], 0.5)
        self.assertAlmostEqual(m["NDCG@5"], dcg / ideal)
        self.assertEqual(m["Recall@5"], 1.0)

    def test_no_strong_fit_and_cutoff(self):
        m = A.ranking_metrics(["x", "y", "z", "w", "v", "target"], {"target": 2})
        self.assertEqual(m["MRR"], 1 / 6)
        self.assertEqual(m["Recall@5"], 0.0)
        self.assertEqual(m["NDCG@5"], 0.0)
        self.assertEqual(A.ranking_metrics(["a"], {"a": 1})["MRR"], 0.0)

    def test_metrics_stay_within_unit_interval_on_random_rankings(self):
        rng = random.Random(8)
        ids = [f"j{i}" for i in range(14)]
        for _ in range(200):
            ranked = rng.sample(ids, len(ids))
            labels = {j: rng.choice([0, 1, 2]) for j in rng.sample(ids, rng.randint(1, 6))}
            for value in A.ranking_metrics(ranked, labels).values():
                self.assertTrue(0.0 <= value <= 1.0 + 1e-9)

    def test_baseline_rank_is_exact_name_only(self):
        job_a = A.JobPosting(id="a", title="A", must_have_skills=["PostgreSQL"])
        job_b = A.JobPosting(id="b", title="B", must_have_skills=["Python"])
        self.assertEqual(A.baseline_rank([job_a, job_b], ["Postgres"]), ["a", "b"])  # tie, catalog order
        self.assertEqual(A.baseline_rank([job_a, job_b], ["python"]), ["b", "a"])


class TestDocumentationClaims(unittest.TestCase):
    """README.md publishes accuracy numbers; keep them honest."""

    def test_readme_accuracy_table_matches_what_eval_prints(self):
        r = A.evaluate()
        self.assertEqual(r["requirement_cases"], 43)
        self.assertAlmostEqual(r["requirement_accuracy"], 0.930, places=3)
        self.assertAlmostEqual(r["baseline_requirement_accuracy"], 0.163, places=3)
        skills, career, baseline = (r["ranking"][k] for k in ("skills", "career", "baseline"))
        for metric, s_expected, c_expected, b_expected in (("Top-1", .89, .89, .78), ("MRR", .91, .92, .87),
                                                           ("NDCG@5", .95, .97, .91), ("Recall@5", .90, .93, .84)):
            with self.subTest(metric=metric):
                self.assertAlmostEqual(skills[metric], s_expected, places=2)
                self.assertAlmostEqual(career[metric], c_expected, places=2)
                self.assertAlmostEqual(baseline[metric], b_expected, places=2)

    def test_readme_says_the_three_misses_are_spring_boot_mongodb_and_react_native(self):
        misses = {(tuple(s), req) for s, req, _, _ in A.evaluate()["misses"]}
        self.assertEqual(misses, {(("Spring Boot",), "Java"), (("MongoDB",), "NoSQL databases"), (("React",), "React Native")})

    def test_builtin_catalog_claims(self):
        self.assertEqual(len(A.SAMPLE_JOBS), 14)
        self.assertEqual(len({j.id for j in A.SAMPLE_JOBS}), 14)
        self.assertEqual(len(A.EVAL_CANDIDATES), 9)

    def test_readme_example_output_for_its_own_command_line(self):
        # README: python agent.py --find-jobs --skills "Python, PostgreSQL, Docker, FastAPI, Git"
        top = A.JobIndex(A.SAMPLE_JOBS).rank(["Python", "PostgreSQL", "Docker", "FastAPI", "Git"], top=1)[0]
        self.assertEqual(top.job.id, "backend-fintech")
        self.assertEqual(top.score, 91)
        self.assertEqual([a.skill for a in top.assessments if a.status == "missing"], ["Cloud (AWS/GCP)"])
        self.assertEqual(sum(a.status == "met" for a in top.assessments), 6)

    def test_readme_documents_the_sample_resume_ranking_miss(self):
        # README: the matcher ranks the frontend job first for the sample resume because it lists React,
        # while a human reading "basic React" would rank the backend role first. Pinned so a change is noticed.
        top2 = [m.job.id for m in A.JobIndex(A.SAMPLE_JOBS).rank(A.SAMPLE_SKILLS, top=2)]
        self.assertEqual(top2[0], "frontend-ecommerce")
        self.assertEqual(A.EVAL_CANDIDATES["sample-resume"][1]["backend-fintech"], 2)
        self.assertNotEqual(top2[0], "backend-fintech")


# ===========================================================================
# 3. PERFORMANCE
# ===========================================================================


def best_of(fn, repeats=3):
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return min(times)


def median_search_ms(index, vocab, queries=12, seed=0, top=10):
    rng = random.Random(seed)
    times = []
    for _ in range(queries):
        skills = rng.sample(vocab, rng.randint(4, 10))
        start = time.perf_counter()
        index.rank(skills, top=top)
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


VOCAB = sorted({s for j in A.SAMPLE_JOBS for s in j.must_have_skills + j.nice_to_have_skills})


@unittest.skipIf(SKIP_PERF, "SKIP_PERF=1")
class TestPerformance(unittest.TestCase):
    def test_index_build_and_search_scale_roughly_linearly(self):
        small_n, big_n = 5_000, 20_000
        small_cat, big_cat = A.synthetic_catalog(small_n), A.synthetic_catalog(big_n)
        build_small = best_of(lambda: A.JobIndex(small_cat))
        build_big = best_of(lambda: A.JobIndex(big_cat))
        small_idx, big_idx = A.JobIndex(small_cat), A.JobIndex(big_cat)
        search_small = median_search_ms(small_idx, VOCAB)
        search_big = median_search_ms(big_idx, VOCAB)
        PERF_REPORT.append((f"index build {small_n:,} / {big_n:,} jobs", f"{build_small:.0f} ms / {build_big:.0f} ms"))
        PERF_REPORT.append((f"search p50 {small_n:,} / {big_n:,} jobs", f"{search_small:.1f} ms / {search_big:.1f} ms"))
        self.assertLess(build_big / build_small, 8, "index build is worse than ~linear")
        self.assertLess(search_big / search_small, 8, "search is worse than ~linear")

    def test_absolute_latency_budgets(self):
        catalog = A.synthetic_catalog(10_000)
        index = A.JobIndex(catalog)
        p50 = median_search_ms(index, VOCAB, queries=20)
        PERF_REPORT.append(("search p50, 10k jobs, top 10", f"{p50:.1f} ms"))
        self.assertLess(p50, 150)
        self.assertLess(best_of(lambda: A.JobIndex(catalog)), 2_000)

    def test_career_ranker_scales_linearly_and_meets_budgets(self):
        small_cat, big_cat = A.synthetic_catalog(5_000), A.synthetic_catalog(20_000)
        build_small = best_of(lambda: A.CareerRanker(small_cat), 2)
        build_big = best_of(lambda: A.CareerRanker(big_cat), 2)
        search_small = median_search_ms(A.CareerRanker(small_cat), VOCAB)
        search_big = median_search_ms(A.CareerRanker(big_cat), VOCAB)
        p50_10k = median_search_ms(A.CareerRanker(A.synthetic_catalog(10_000)), VOCAB, queries=20)
        PERF_REPORT.append(("career build 5,000 / 20,000 jobs", f"{build_small:.0f} ms / {build_big:.0f} ms"))
        PERF_REPORT.append(("career search p50 5,000 / 20,000 jobs", f"{search_small:.1f} ms / {search_big:.1f} ms"))
        PERF_REPORT.append(("career search p50, 10k jobs, top 10", f"{p50_10k:.1f} ms"))
        self.assertLess(build_big / build_small, 8, "career build is worse than ~linear")
        self.assertLess(search_big / search_small, 8, "career search is worse than ~linear")
        self.assertLess(p50_10k, 250)

    def test_default_builtin_search_is_effectively_instant(self):
        ms = best_of(lambda: A.find_jobs(A.SAMPLE_SKILLS, A.SAMPLE_JOBS, top=5), repeats=5)
        PERF_REPORT.append(("built-in 14-job search incl. index build", f"{ms:.2f} ms"))
        self.assertLess(ms, 50)

    def test_only_returned_jobs_are_explained(self):
        index = A.JobIndex(A.synthetic_catalog(3_000))
        with mock.patch.object(A.JobIndex, "_explain", autospec=True, side_effect=A.JobIndex._explain) as spy:
            index.rank(VOCAB[:8], top=7)
        self.assertEqual(spy.call_count, 7)

    def test_top_k_selection_is_not_slower_than_full_sort_by_much(self):
        index = A.JobIndex(A.synthetic_catalog(20_000))
        top10 = best_of(lambda: index.rank(VOCAB[:8], top=10))
        PERF_REPORT.append(("rank top-10 vs full sort (20k jobs)", f"{top10:.0f} ms vs {best_of(lambda: index.rank(VOCAB[:8], top=None), 1):.0f} ms"))
        self.assertLess(top10, 400)

    def test_second_identical_search_is_not_slower(self):
        index = A.JobIndex(A.synthetic_catalog(10_000))
        first = best_of(lambda: index.rank(VOCAB[:8]), 1)
        second = best_of(lambda: index.rank(VOCAB[:8]), 3)
        self.assertLess(second, first * 2 + 20)

    def test_memory_of_a_10k_job_index(self):
        catalog = A.synthetic_catalog(10_000)
        A.canon.cache_clear()
        A.alternatives.cache_clear()
        tracemalloc.start()
        try:
            index = A.JobIndex(catalog)
            index.rank(VOCAB[:8])
            current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        PERF_REPORT.append(("memory, 10k-job index + one search", f"{current / 1e6:.1f} MB held, {peak / 1e6:.1f} MB peak"))
        self.assertLess(peak / 1e6, 250)

    def test_concurrent_searches_match_serial_results(self):
        index = A.JobIndex(A.synthetic_catalog(5_000))
        queries = [VOCAB[i:i + 8] for i in range(0, 40, 4)]
        serial = [[(m.job.id, m.score) for m in index.rank(q, top=10)] for q in queries]
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=8) as pool:
            parallel = list(pool.map(lambda q: [(m.job.id, m.score) for m in index.rank(q, top=10)], queries * 3))
        elapsed = (time.perf_counter() - start) * 1000
        PERF_REPORT.append((f"{len(queries) * 3} concurrent searches on 8 threads, 5k jobs", f"{elapsed:.0f} ms"))
        self.assertEqual(parallel, serial * 3)

    def test_caches_make_repeated_parsing_cheap(self):
        A.canon.cache_clear()
        A.alternatives.cache_clear()
        for _ in range(3):
            for job in A.SAMPLE_JOBS:
                for s in job.must_have_skills + job.nice_to_have_skills:
                    A.alternatives(s)
        info = A.alternatives.cache_info()
        self.assertGreater(info.hits, info.misses)

    def test_hostile_long_requirement_strings_do_not_hang(self):
        for label, text in (("open parens", "(" * 10_000 + "x"), ("many ors", " or ".join(["go"] * 20_000)),
                            ("nested slashes", "/".join(["a"] * 20_000)), ("one giant word", "x" * 200_000)):
            A.alternatives.cache_clear()
            start = time.perf_counter()
            A.alternatives(text)
            ms = (time.perf_counter() - start) * 1000
            PERF_REPORT.append((f"alternatives() on {label} ({len(text):,} chars)", f"{ms:.0f} ms"))
            self.assertLess(ms, 2_000, label)

    def test_alternatives_is_linear_on_unbalanced_parentheses(self):
        # The old regex r"(.*?)\((.+)\)\s*" backtracked: 4x the input cost ~16x the time (68 ms at 5k chars,
        # 1.1 s at 20k, 4.3 s at 40k). Requirement strings come from user-supplied --jobs files.
        def cost(n):
            def once():
                A.alternatives.cache_clear()
                A.alternatives("(" * n + "x")
            return best_of(once, 2)

        small, big = cost(5_000), cost(20_000)
        PERF_REPORT.append(("alternatives() '(' x5k -> x20k (4x input)", f"{small:.0f} ms -> {big:.0f} ms ({big / small:.1f}x; linear would be 4x)"))
        self.assertLess(big / small, 8)

    def test_large_candidate_skill_list_is_fast(self):
        skills = [f"Skill{i}" for i in range(20_000)] + VOCAB
        ms = best_of(lambda: A.SkillMatcher(skills))
        PERF_REPORT.append(("SkillMatcher on 20k listed skills", f"{ms:.0f} ms"))
        self.assertLess(ms, 1_000)

    def test_rerank_overhead_is_parallel_not_serial(self):
        delay = 0.15

        def slow(state):
            time.sleep(delay)
            return {"match": A.MatchResult(score=50, assessments=[], summary="s")}

        with mock.patch.object(A, "match_skills", side_effect=slow):
            start = time.perf_counter()
            A.find_jobs(A.SAMPLE_SKILLS, A.SAMPLE_JOBS, top=8, rerank=8)
            elapsed = time.perf_counter() - start
        PERF_REPORT.append(("rerank 8 jobs, 150 ms fake LLM, 4 workers", f"{elapsed * 1000:.0f} ms (serial would be 1200 ms)"))
        self.assertLess(elapsed, delay * 8 * 0.6)

    def test_import_time_is_reasonable(self):
        start = time.perf_counter()
        r = cli("--find-jobs", "--skills", "Python", "--json")
        wall = time.perf_counter() - start
        PERF_REPORT.append(("CLI cold start + search (subprocess)", f"{wall * 1000:.0f} ms"))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertLess(wall, 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
