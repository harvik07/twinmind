# TwinMind dashboard: UX redesign spec (Prompt 11, Stage A)

**Audience:** a factory manager with no machine-learning knowledge, who wants an
answer and a reason in five minutes, without help.

**Scope of Stage B:** UI layer only (`app/`, `src/twinmind/ui/`, `.streamlit/`,
tests, README, screenshots). No change to the agent, critic, twin, benchmark
code, pre-registrations or model-card results. CI stays network-free; the app
stays one process and light on RAM. Three points in §11 need your decision
before Stage B starts.

---

## A1. Audit of the current UI (v1.0-dashboard)

Taken from the four pages as they render today and from `docs/dashboard_*.png`.
Severity: **H** stops a manager, **M** slows or misleads them, **L** cosmetic.

### Whole app

| # | Where | Problem | Sev |
|---|---|---|---|
| 1 | Sidebar | "Demo mode (replayed benchmark results)" then "• TWINMIND_DEMO=1 is set": an environment variable shown to a manager. "Replayed" and "benchmark" mean nothing to them. | H |
| 2 | Sidebar | "Every number comes from the simulator or its surrogate": two terms the manager doesn't know, stated before they've done anything. | M |
| 3 | Page names | "Ask", "What-if", "Evidence", "History": "Ask" doesn't say what you get; "Evidence" sounds like a legal file. | M |
| 4 | Everywhere | No first-visit guidance, no glossary, no hint of what goals are possible. | H |
| 5 | Toolbar | Streamlit's "Deploy" button and menu are visible: an invitation to break things. | L |
| 6 | Theme | Default Streamlit red sliders and buttons; colour carries no meaning; the light/dark result depends on the browser. | L |

### Ask page

| # | Where | Problem | Sev |
|---|---|---|---|
| 7 | Goal box | A blank text area with one example placeholder. The manager has to guess the vocabulary (which KPIs exist, how to phrase a limit). | H |
| 8 | "Planning context (a 28-day demand scenario)" | A dropdown of the numbers 0–19, default 8. Meaningless and unexplained; most users will either ignore it or worry about it. | H |
| 9 | Buttons | "Run live" in live mode, "Find the closest replayed goal" in demo mode, plus "Show this replay" further down: two competing actions, neither of which says what happens. | H |
| 10 | Replay picker | Options read "G07 — A clean feasible goal: no critic flag, accepted first time." Benchmark IDs, "feasible", "critic flag". | H |
| 11 | Live run | Takes 2–4 minutes, and **clicking anything else on the page stops it** (recorded as aborted). The only warning is a grey caption. Hostile. | H |
| 12 | Progress | "Explorer searched plans on the surrogate and shortlisted them on the twin" and "LLM calls: 2 / 6": internals, not progress a manager understands. | M |
| 13 | Result headline | "✅ Plan found" then "Recommended plan, verified on the twin." It never says *what to do* or *what it gives*. | H |
| 14 | Result: G09 | The headline says "Recommended" while profit falls from £344,569 to −£243,182. The side-effects box now catches this, but it sits mid-page as five bullets and a sentence about "the 95% interval of each change". | H |
| 15 | "Objective: Maximise OEE / Constraints: Unplanned failures at most 1.50" | "Objective", "Constraints", "OEE" unexplained; "1.50 failures" reads oddly as a limit. | M |
| 16 | "Verdict reason: improves the objective over the current policy" | Circular wording and jargon. | M |
| 17 | "Plan changes (16 levers)" | Collapsed by default, and inside it raw code names with two decimals and no units: `pm_interval_M1: 168.00 → 405.49`. The single most actionable thing is the hardest to read. | H |
| 18 | KPI table | 8 rows × 8 columns: "Change 95% CI", "Plan 95% CI", "Source" repeated as "verified on twin (30 reps)" on every row, "Note" truncated to "⚠️ indic…". No row says "better" or "worse". | H |
| 19 | Weak-KPI badge | "indicative: surrogate R² 0.91 / 0.73, and sensitive to the health-model assumption (feasibility flipped in 7 of 40 benchmark cells under a fixed-failure-rate hazard)". Correct, and incomprehensible to the audience. | H |
| 20 | Critic flags | "weak kpi constraint", "novel plan": code names as headings. | M |
| 21 | Coverage | "🟡 Training coverage: Edge of training coverage: the surrogate has seen few plans like this." Jargon; colour dot carries the meaning. | M |
| 22 | Explanation | The AI's text uses `downtime_hours`, "recommendation_reason", "T1 95th percentile 0.987", "the surrogate is extrapolating", numbers like "23580.0", and its own markdown headings, so it's long and looks like a log. | H |
| 23 | Footer | "Replayed: the plan and its twin numbers are from the benchmark run (260 twin runs)". | L |
| 24 | Infeasible result | Options appear as "On-time delivery at least 99.0%. Closest any verified plan came: 84.8%. *relax…*" with no headline saying "this can't be done". | M |

### What-if page

| # | Where | Problem | Sev |
|---|---|---|---|
| 25 | 16 sliders | Code names (`pm_interval_M3`, `speed_setpoint_M4`), two decimals, no units, alphabetical-ish order with no grouping. Help tooltips repeat config text ("in operating hours"). | H |
| 26 | "Start from" + "Load" | A dropdown of "Replayed plan: G01 — The critic acted: it swapped plans…" and a separate Load button: two steps for one action. | M |
| 27 | "Surrogate estimate" | Heading is jargon, and the estimate table has **no current-setup column**, so the user can't tell whether their change helped. | H |
| 28 | "Verify on twin" / "Replications" slider 2–10 | "Verify", "twin", "replications" are all jargon; a manager can't choose a replication count. "60 s of twin time" ditto. | H |
| 29 | Demo mode | An info box about files "not on this machine" / demo mode, then a replay — the page does almost nothing useful without hunting for "Load". | M |

### Evidence page

| # | Where | Problem | Sev |
|---|---|---|---|
| 30 | B1–B5 table | "non-inferior", "normalised gap (bound +0.10)", "mean(agent − twin-only) = -0.335, 95% CI…", "spec-match", "no-room goals". Unreadable for the audience. | H |
| 31 | B5 note | "checker v1… plain-decimal facts". Right for a reviewer, opaque for a manager. | M |
| 32 | Mutation table | "direction_flip (KPI sentence)". | M |
| 33 | Limitations | Whole model-card sections pasted as markdown with internal references ("§3", "sf_tpe", "H2"). | M |
| 34 | Model cards | 200–750-line documents in a box; the default card is F. | M |
| 35 | Figures | Dropdown of file stems ("f bench gap"). | L |

### History page

| # | Where | Problem | Sev |
|---|---|---|---|
| 36 | Table | Status values are code (`ok`, `spec_invalid`); Detail holds exception names ("RuntimeError: …"). | M |
| 37 | Agent run log | "LLM calls", "Twin runs", and the file path `reports/agent_runs.jsonl`. | M |
| 38 | Reopen | An "Open" dropdown keyed by timestamp · label · truncated goal; you can't click a row. | L |

**Summary:** every page leads with mechanism, not answer. The honesty labels are
all present but written for a reviewer. Nothing tells the manager what to type,
which button to press, or what the result means for their factory.

---

## A2. Information architecture

| New page (plain name) | Replaces | What moves where |
|---|---|---|
| **Plan** (home) | Ask | Goal form (cards → limits → "Find a plan"), own-words alternative, result. Saved examples move to a small "See a saved example" link under the form. |
| **Try changes** | What-if | Settings grouped into four sections; quick estimate vs detailed check side by side. "Start from" is two buttons, not a dropdown + Load. |
| **How reliable is this?** | Evidence | Plain summary first; the five tests in plain words; then limitations; model cards and figures move into "For reviewers" expanders. |
| **Past results** | History | One plain table; click a row to reopen. The agent run log moves into "Technical details". |

Global:

- **Sidebar:** product name, one line ("Finds production plans and checks them in a detailed simulation of your line"), the mode chip in plain words ("Live: plans are found now" / "Example mode: showing saved example results"), and a "Glossary" link (an expander or dialog). The *reason* for example mode moves into Technical details.
- **Every result** ends with **"Show technical details"**, holding everything the old page showed, verbatim, with the exact technical labels (`verified on twin (30 reps)`, the R² badge, novelty score, flag codes, CIs, raw explanation source).
- **Toolbar:** hidden (`client.toolbarMode = "viewer"`).

---

## A3. Plan page

### Wireframe

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Plan                                                                     │
│ Find better settings for your line, checked in a detailed simulation.   │
│                                                                          │
│ ┌─ Welcome (first visit, [Got it ✕]) ───────────────────────────────────┐ │
│ │ ① Choose what you want to improve.                                    │ │
│ │ ② Set the limits you can live with (or skip them).                    │ │
│ │ ③ Press "Find a plan". It takes about 2–4 minutes.                    │ │
│ └───────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│ 1  What do you want to improve?                                          │
│ ┌────────────────┐ ┌────────────────┐ ┌────────────────┐                 │
│ │ 💷 Make more    │ │ 🔧 Reduce       │ │ 🚚 Deliver      │                 │
│ │    profit       │ │    breakdowns   │ │    orders on    │                 │
│ │ e.g. "Earn more │ │ e.g. "Fewer     │ │    time         │                 │
│ │ without missing │ │ machine break-  │ │ e.g. "Ship more │                 │
│ │ deliveries"     │ │ downs a month"  │ │ orders on the   │                 │
│ └────────────────┘ └────────────────┘ │ day they're due"│                 │
│ ┌────────────────┐ ┌────────────────┐ └────────────────┘                 │
│ │ 🏷️ Lower cost   │ │ 📦 Keep stock   │ ┌────────────────┐                 │
│ │    per unit     │ │    from running │ │ ✏️ Describe it  │                 │
│ │ e.g. "Make each │ │    out          │ │    in your own  │                 │
│ │ unit cheaper"   │ │ e.g. "Fewer days│ │    words        │                 │
│ └────────────────┘ │ short of stock" │ └────────────────┘                 │
│                    └────────────────┘                                    │
│                                                                          │
│ 2  Any limits?   (only those that fit the chosen goal; all optional)     │
│ ☑ Most breakdowns I accept in 4 weeks     [  1  ] ⓘ                       │
│ ☐ Lowest on-time delivery I accept (%)    [ 80  ] ⓘ                       │
│ ☐ Highest chance of running out of stock  [ 30% ] ⓘ                       │
│ ☑ Allow overtime?                         (•) Yes  ( ) No  ⓘ              │
│ ▸ More options (which month of orders to plan for)                       │
│                                                                          │
│ 3  [ Find a plan ]   Usually takes 2–4 minutes. You can keep using the   │
│                      page while it works.                                │
│                                                                          │
│ No time? See a saved example result ▸                                    │
└──────────────────────────────────────────────────────────────────────────┘
```

Cards are `st.button`s inside a 3-column grid (`st.columns`), styled as
cards; the chosen card gets a visible "Selected ✓" state (border + word, not
colour alone). On a tablet the grid becomes two columns, on a phone one.

### Step 1: goal cards → goal spec

Each card builds an `ObjectiveSpec` directly (no language model needed, so a
card goal costs one model call, for the summary, instead of two). The text shown
in Past results is generated from the card and the limits ("Make more profit,
with at most 1 breakdown in 4 weeks").

| Card | Example line | Objective (sense) | Limits offered (constraint, default) | Saved example in example mode |
|---|---|---|---|---|
| 💷 Make more profit | "Earn more without missing deliveries" | `profit` (max) | breakdowns ≤ 1 (off); on-time ≥ 80% (off); stock-out ≤ 30% (off); overtime yes/no | **G01** s1 (profit, breakdowns ≤ 1): exact family |
| 🔧 Reduce breakdowns | "Fewer machine breakdowns a month" | `unplanned_failures` (min) | profit ≥ £200,000 (on); on-time ≥ 80% (off) | **G10** s2 (lower downtime, profit ≥ £200k): *related*, says so |
| 🚚 Deliver orders on time | "Ship more orders on the day they're due" | `on_time_delivery_rate` (max) | profit ≥ £200,000 (on); cost per unit ≤ £30 (off) | **G07** s1 (profit and on-time together): *related*, says so |
| 🏷️ Lower cost per unit | "Make each unit cheaper" | `unit_cost` (min) | on-time ≥ 80% (on); stock-out ≤ 30% (off) | **G03** s2 (cost per unit, on-time ≥ 80%): exact family |
| 📦 Keep stock from running out | "Fewer days short of stock" | `stockout_probability` (min) | profit ≥ £200,000 (on); overtime yes/no | **AA1** s1 (profit, stock-out ≤ 25%): *related*, says so |
| ✏️ Describe it in your own words | — | from the language planner (live) | none (the planner reads them from the text) | closest saved example by wording (today's rule) |

"Allow overtime? No" sets `free_dims` = the 16 settings minus `overtime_hours`,
which then stays at the current setup's value (0 h). All other settings stay
free. Defaults are the benchmark's own thresholds where one exists (G03, G10),
otherwise round numbers near the current setup.

**Example mode** never pretends the saved example answers the user's exact
goal. It shows "Saved example result, not a live run", then, when the example
differs, one line: *"This saved example was for a related goal: lower downtime,
keeping profit at £200,000 or more. Your settings were not used."*

**Stage B demo-cache change:** add G03 s2 and G10 s2 from the existing benchmark
cells via `scripts/build_demo_cache.py` (no new runs, no model calls; the
builder re-checks both explanations with checker v2, and both pass today).

### Step 2: limits

Each limit is a checkbox ("use this limit") and a number box in business units,
with help text. Only limits that fit the card are shown.

| Label | Constraint | Unit / range | Help text |
|---|---|---|---|
| Most breakdowns I accept in 4 weeks | `unplanned_failures <= x` | 0–10, step 0.5 | "Unexpected machine failures over the 4-week plan, averaged over 30 simulated months." |
| Lowest on-time delivery I accept (%) | `on_time_delivery_rate >= x/100` | 50–100% | "Share of ordered units delivered on the day they were due." |
| Highest chance of running out of stock (%) | `stockout_probability <= x/100` | 0–100% | "Share of days with at least some demand you could not supply." |
| Lowest profit I accept (£, 4 weeks) | `profit >= x` | £0–£600,000 | "Money left after all costs over the 4 weeks." |
| Highest cost per unit I accept (£) | `unit_cost <= x` | £10–£100 | "All costs divided by units made." |
| Allow overtime? | `free_dims` without `overtime_hours` when No | Yes / No | "No keeps overtime at zero hours a day." |

Validation in plain words before running: a limit outside its range, or two
limits that can't both hold (the spec's own contradiction check) →
"These two limits can't both be met: … Change one of them." The spec is built
and validated in the view model, so a bad form never reaches the agent.

**More options** (collapsed): "Which month of orders to plan for" — a select of
"Example month 1 … 20" (dev contexts 0–19, default as today). Plain help: "Each
example month has a different pattern of customer orders."

### Own-words alternative

The ✏️ card opens a text box with three clickable example sentences under it
("Make more profit but keep breakdowns to one a month", …). Live: the existing
language planner turns it into a goal; if it asks a question, show the question
as a plain prompt with the text box below it ("We need one more detail: …").
Example mode: closest saved example by wording, labelled.

### Step 3: "Find a plan" and progress

One primary button: **Find a plan**. Under it: "Usually takes 2–4 minutes. You
can keep using the page while it works." (2–4 minutes = model card E's 106–205 s
per live goal, measured on a quiet machine.)

**Decision: run the work in one background thread, polled by `st.fragment`.**

- The button starts a single `threading.Thread` in the Streamlit process. It
  runs the agent and writes progress into a job object held in
  `st.cache_resource` (one per process). The thread makes no Streamlit calls.
- A fragment with `run_every="1s"` reads the job and redraws only the progress
  panel, so clicks elsewhere (glossary, More numbers, other pages) **no longer
  stop the run**.
- One run at a time: the existing `RunGuard` lock is taken before the thread
  starts and released in the thread's `finally`. While it's held, "Find a plan"
  is disabled with the line "A plan is being worked out (started 14:02). It
  will appear here when it's ready."
- Every run gets a "started" entry in Past results immediately and a finished /
  failed entry from the thread. If the app is restarted mid-run, the orphaned
  "started" entry is shown as "Stopped before finishing" — never silently lost.
- Why not lock the controls instead: Streamlit reruns on every click, so a
  synchronous run can only be protected by disabling every widget on every
  page, and a page change would still stop it. A thread is the only way to keep
  the page usable. Cost: one extra thread, no extra process, no extra memory
  beyond the run itself; the run's twin work is the same.
- This relaxes Prompt 10's "no background workers" to exactly **one** worker
  thread, one run at a time — see §11, decision 1.

Progress view (plain steps, ticked as nodes finish; elapsed time shown):

```
Finding a plan for: Make more profit, at most 1 breakdown in 4 weeks
  ✓ Understanding your goal
  ✓ Searching for plans                       (about 1 minute)
  ◌ Checking the best candidates in the detailed simulation (30 runs)
  ○ Reviewing for risks
  ○ Writing the summary
  Elapsed 1:42 · usually 2–4 minutes
```

Node mapping: planner → Understanding your goal (ticked at once for card goals,
which have no planner step); explorer → Searching for plans; evaluator →
Checking the best candidates; critic → Reviewing for risks (a retry shows
"Checking another candidate"); explainer → Writing the summary. The model-call
count moves to Technical details.

### Result layout

Shown with the G01 saved example's real numbers (16 settings change; one shift
plus 3.85 h overtime = 11.85 running hours a day).

```
┌──────────────────────────────────────────────────────────────────────────┐
│ [✅ Recommended plan]            How much to trust this: ●○○ Rough guide │
│                                                                          │
│ Change 16 settings to raise profit from £297,771 to £381,879 over 4 weeks│
│ (+£84,109), keeping breakdowns at or below 1.                            │
│                                                                          │
│ What to change                                                           │
│  • Service the welding machine (M3) every 237 running hours instead of   │
│    120 — less often (about every 20 working days instead of 7½)          │
│  • Run the line at 128% of normal pace instead of 100%                   │
│  • Overtime: 3.9 hours a day instead of none                             │
│  … 13 more ▸                                                             │
│                                                                          │
│ What you get   (checked with the detailed simulation, 30 runs)           │
│ ┌ Profit ⓘ ───────┐ ┌ On-time delivery ⓘ ┐ ┌ Breakdowns ⓘ ─┐ ┌ Cost/unit ⓘ┐│
│ │ £297,771 →      │ │ 81.3% → 81.5%      │ │ 1.6 → 0.9     │ │ £28.18 →   ││
│ │ £381,879        │ │ ➖ no clear change   │ │ ⬇ better ⚠    │ │ £24.79     ││
│ │ ⬆ better        │ │                    │ │ rough guide   │ │ ⬇ better   ││
│ └─────────────────┘ └────────────────────┘ └───────────────┘ └────────────┘│
│ ▸ More numbers (units made, downtime, equipment efficiency, stock-outs)  │
│                                                                          │
│ Things to watch                                                          │
│  ⚠️ Medium · Your breakdown limit is met on average, but not in every     │
│     simulated month.                                                     │
│  ℹ️ Low · Breakdown and downtime numbers are rough guides: the model is   │
│     less accurate here.                                                  │
│                                                                          │
│ Why this plan                                                            │
│  It makes more profit than your current setup, by a clear margin, and    │
│  keeps breakdowns within your limit on average.                          │
│  ▸ Read the AI assistant's explanation                                   │
│                                                                          │
│ How much to trust this: ●○○ Rough guide  (2 of 3 checks below failed)    │
│  ✓ Numbers checked with the detailed simulation (30 runs)                │
│  ✗ Your limits are met on average, not in every simulated month          │
│  ✓ Settings are close to ones the model has seen before                  │
│  ✗ The goal involves breakdown or downtime numbers, which are rough      │
│                                                                          │
│ ▸ Show technical details                                                 │
└──────────────────────────────────────────────────────────────────────────┘
```

**(a) Headline and status chip.** Chip = icon + word + colour, never colour
alone:

| Agent outcome | Chip | Headline template (all numbers from the twin table) |
|---|---|---|
| `ok`, recommended (incl. `degraded_template`) | ✅ **Recommended plan** (green) | "Change N settings to {raise/lower} {objective KPI} from {before} to {after} over 4 weeks ({delta}){, keeping {limit} …}." |
| `ok`, not recommended (`no_improvement`) | ➖ **No better plan found: your current setup is already close** (amber) | "Keep your current setup. The best plan found changed {KPI} by {delta}, which is too small to tell apart from normal month-to-month variation." |
| `infeasible` | ⛔ **This goal can't be met: closest options below** (red) | "No plan met all your limits. The closest any checked plan came: {KPI} {closest} (you asked for {op} {threshold})." |
| `needs_clarification` | ❓ **We need one more detail** (blue) | the planner's question |
| `spec_invalid` | ⛔ **We couldn't use this goal** (red) | §A8 message |

For a weighted objective (G07-style) the headline names both KPIs.
"Too small to tell apart" = the paired 95% interval of the change includes
zero; the interval itself is in Technical details.

**(b) What to change.** Every changed setting, in plain words with units,
current → new, sorted by size of change (relative to its range), first three
shown, rest under "N more". Phrasing comes from the glossary (A7): service
intervals add "more/less often" and an approximate working-day equivalent
(hours ÷ the plan's hours per day: 8 × shifts + overtime); percentages for
speed and pace; units for stock. Settings held fixed by the goal (supplier,
price) are not listed as changes; they appear in Technical details.

**(c) What you get.** Four `st.metric` cards, each with before → after, a
**direction word** ("better", "worse", "no clear change") and an arrow; colour
follows the word (green/red/grey). "No clear change" when the change's 95%
interval includes zero. The four cards are the goal's own KPIs first, then
filled from profit, on-time delivery, breakdowns, cost per unit. Breakdown and
downtime cards carry "⚠ rough guide" under the value. The header line says
**"checked with the detailed simulation (30 runs)"** once for the block. The
other KPIs are in a "More numbers" table: KPI, Current setup, Plan, Change,
better/worse/no clear change. One table, one label, never mixed with
estimates.

**(d) Things to watch.** Critic flags and side effects in plain words, each
with an icon and a severity word:

| Source | Plain text | Severity |
|---|---|---|
| side effect (Prompt 10 rule, unchanged) | "{KPI} gets worse: {before} → {after}. It isn't part of your goal, so check you can live with it." | High if KPI is profit or on-time delivery, else Medium |
| `not_robustly_feasible` | "Your limits are met on average, but not in every simulated month." | Medium |
| `weak_kpi_constraint` | "Your goal uses breakdown or downtime numbers, which are rough guides: the model is less accurate here." | Low |
| `novel_plan` | "These settings are unlike most the model has seen, so its quick estimates were less reliable. The numbers above come from the detailed simulation." | Low |
| `no_improvement` | "The best plan found isn't clearly better than your current setup." | Info |
| `infeasible_unverified` | "None of the shortlisted plans met every limit in the detailed simulation." | Info |
| weak KPI shown on the page (always, when breakdown/downtime cards appear) | "Breakdown and downtime numbers are rough guides: the model is less accurate here." | Low |

Severity is a fixed table in code, not judged per result.

**(e) Why this plan.** See §11 decision 2. Proposed: a short **plain summary
generated by the UI from the facts** (which goal KPI moved by how much, whether
limits hold, the verdict), then an expander **"Read the AI assistant's
explanation"** holding the model's text verbatim with the caption "Written by
the AI assistant; it passed an automated check of numbers and KPI directions"
and the nested "What this check does not catch" list. Reason: the model's text
contains jargon we cannot reword without changing the agent (G09's says "the
surrogate is extrapolating"; others use `downtime_hours`,
"recommendation_reason").

**(f) How much to trust this.** A deterministic rule from existing facts, shown
with its reasons. No probabilities, no scores.

Four checks:

| # | Check | Passes when | Source |
|---|---|---|---|
| T1 | Checked with the detailed simulation | the numbers shown come from twin replications (≥ 2) | `baseline_comparison.n_reps`, or a What-if check |
| T2 | Limits met in every simulated month | the plan is recommended AND `feasible_robust`; **not applicable** when not recommended or there are no limits | evaluation, flags |
| T3 | Familiar territory | novelty score ≤ 0.987382 (the T1 95th percentile) | `plan.novelty` |
| T4 | No breakdown/downtime numbers in the goal | objective and limits avoid `unplanned_failures` and `downtime_hours` | spec |

Rule:

- T1 fails → **●○○ Rough guide** ("Quick estimate only, not checked").
- Otherwise count the failed checks among T2–T4 (not-applicable counts as passed):
  0 → **●●● Checked**, 1 → **●●○ Partly checked**, 2–3 → **●○○ Rough guide**.

Every check is listed under the indicator with ✓/✗ and one plain sentence.
Applied to today's saved examples: G07, AA1, N01, I01, G03 → Checked; G10 →
Partly checked (downtime goal); G01 → Rough guide (limits not robust +
breakdowns); G09 → Rough guide (less familiar + breakdowns). Side effects do
**not** lower trust: trust is about how solid the numbers are, not whether the
plan is good; side effects live in Things to watch.

**(g) Show technical details.** Everything the v1.0 page showed: exact labels
(`verified on twin (30 reps)`), the full 8-row table with both 95% CIs, the
R² badge text verbatim, flag codes, novelty score and band, objective and
constraint spec, raw setting names and values, model calls used against the
cap, twin runs, wall time, demand month (context id), explanation source.

---

## A4. Try changes page

```
┌──────────────────────────────────────────────────────────────────────────┐
│ Try changes                                                              │
│ Move a setting and see roughly what happens. Then check it properly.     │
│                                                                          │
│ Start from:  [ Current setup ]  [ Recommended plan ]   (Recommended plan │
│              is greyed out until you have one on the Plan page)          │
│                                                                          │
│ ▾ Shifts and overtime    Shifts per day [2] · Overtime per day [0.0 h]   │
│ ▸ Maintenance            Service interval for each machine (running h)   │
│ ▸ Machine speed          Overall line pace (%) · Speed of each machine (%)│
│ ▸ Stock and ordering     Reorder level · Order size · Safety stock       │
│ [Reset to current setup]                                                 │
│                                                                          │
│ Quick estimate  (instant, rough: not checked)                            │
│ ┌ Profit ┐ ┌ On-time ┐ ┌ Breakdowns ┐ ┌ Cost/unit ┐  each: current →    │
│ │ … → …  │ │ …       │ │ … rough    │ │ …         │  estimate, word      │
│ ▸ More numbers                                                           │
│ Familiar territory: these settings are close to ones the model has seen. │
│                                                                          │
│ [ Check with the detailed simulation ]  Takes a few seconds (10 runs).   │
│                                                                          │
│ Checked with the detailed simulation (10 runs)                           │
│ ┌ Profit ┐ ┌ On-time ┐ …   values with "likely between £… and £…" in help │
│ Took 4.2 s. (slower than usual: the computer may be busy)                │
│ ▸ Show technical details                                                 │
└──────────────────────────────────────────────────────────────────────────┘
```

- **Start from** is two buttons. "Recommended plan" uses the last result shown
  on the Plan page (or a saved example's plan in example mode, labelled).
- **Settings** use friendly names, units and help text from the glossary, in
  the four sections. Whole numbers where the setting is whole (shifts);
  sensible steps elsewhere; percentages for speed and pace. A small "changed"
  marker next to any setting moved away from the current setup.
- **Quick estimate** shows four cards against the **current setup's quick
  estimate** (fixes audit #27), so the direction word is meaningful. Label:
  "Quick estimate (instant, rough: not checked)". No interval, ever.
- **Familiarity** line replaces the novelty band, in plain words (familiar /
  less familiar / unfamiliar: "treat the quick estimate with caution").
- **Check with the detailed simulation**: one button, "Takes a few seconds"; it
  runs 10 replications under the existing 60 s cap (the user no longer picks a
  count). Result block labelled "Checked with the detailed simulation (N runs)"
  with the real N; cards show the checked value, and "likely between X and Y"
  (the 95% interval in plain words) in each card's help. Measured seconds are
  shown; a slow machine gets "slower than usual: the computer may be busy" —
  not an error.
- Estimates and checked numbers are separate blocks with separate headings,
  never one table (Prompt 10 rule).
- **Example mode:** settings work; estimates and checks are unavailable, and
  the page says so in one line: "Quick estimates and checks need the full
  installation. Start from a saved example's plan to see its checked numbers."

---

## A5. How reliable is this?

1. **Plain summary first** (fixed text, written from the card F results):

   > We tested the planner on 20 goals, twice each, against a simple search
   > with no AI and no reviewer step, and decided in advance what would count
   > as a pass. **3 of 5 tests passed and 2 failed.** It finds plans at least
   > as good as the simple search; its reviewer step catches bad
   > recommendations; it says so when a goal can't be met. It failed on two
   > things: its recommended plans met the limits in every simulated month
   > slightly less often than the simple search's (22 of 23 against 31 of 32),
   > and 5 of its 40 written explanations failed the automatic check at the
   > time.

2. **The five tests**, as five identical cards (same size, same layout, chip
   = icon + word: "✅ Passed" / "❌ Failed"):

   | Test | Plain question | Plain result |
   |---|---|---|
   | B1 | Does it find plans at least as good as a simple search? | Passed: its plans were better on average. |
   | B2 | Do its recommended plans meet your limits in every simulated month at least as often? | Failed: 22 of 23 against 31 of 32. |
   | B3 | Does the reviewer step stop bad recommendations? | Passed: 1 bad recommendation with it, 12 without. |
   | B4 | Does it say so when a goal is impossible or pointless? | Passed: all 4 impossible goals, all 3 pointless ones. |
   | B5 | Does it understand goals and explain results correctly? | Failed: it understood all 20 goals, but 5 of 40 explanations failed the check. |

   Under B5, plain: "The explanation check has been improved since this test,
   but the improved version has not been retested." Each card has "Show the
   exact test" with the pre-registered wording, the deciding number and the
   B5/checker caveats verbatim (as today).

3. **Known limits** in plain bullets (from the cards' "Underperformed" sections,
   rewritten): a simulator not a factory; breakdown/downtime numbers rough;
   the explanation check misses some mistakes; small test; vague goals
   sometimes not questioned. Each with "Read the full section" → the original
   text.

4. **For reviewers** (collapsed): model cards, figures with readable titles,
   the mutation table, the checker caveats, all verbatim.

---

## A6. Past results

| When | Your goal | Outcome | Type |
|---|---|---|---|
| Today 14:06 | Make more profit, at most 1 breakdown in 4 weeks | ✅ Recommended plan | 🟢 Live |
| Today 13:50 | Lower cost per unit | 📁 Saved example (related goal) | 📁 Saved example, not a live run |
| Yesterday 17:12 | "Squeeze more money out…" | ⛔ Stopped before finishing | 🟢 Live |

- `st.dataframe` with `on_select="rerun"`, single-row selection: clicking a row
  reopens the result below in the Plan result layout.
- Outcomes are the chip words from A3; "Type" is "Live" vs "Saved example, not
  a live run" (Prompt 10 rule: labelled apart).
- Errors appear as their A8 sentence, never an exception name; the raw detail
  is in Technical details. The agent's own run log (CLI runs) moves to
  Technical details.
- Empty state: "Nothing here yet. Results you find on the Plan page appear here."

---

## A7. Glossary (single source of truth)

`src/twinmind/ui/glossary.py`: frozen dataclasses `KPITerm` and `SettingTerm`,
a `GLOSSARY` mapping, and a plain-word list for terms such as "detailed
simulation". Used by: the cards' help text, the What to change sentences, the
Try changes sliders, the More numbers table, the glossary panel, the trust
reasons, the explanation caption, and the tests (every KPI and setting must
have an entry; every label shown must come from it).

### KPIs (4-week plan period)

| Code | Friendly name | Unit | Meaning (one line) | Better |
|---|---|---|---|---|
| `profit` | Profit | £ over 4 weeks | Money left after all costs. | Higher |
| `units_produced` | Units made | units over 4 weeks | Finished units off the end of the line. | Higher |
| `on_time_delivery_rate` | On-time delivery | % of units ordered | Share of ordered units delivered on the day they were due. | Higher |
| `unplanned_failures` | Breakdowns | count over 4 weeks | Unexpected machine failures (not planned services). *Rough guide.* | Lower |
| `downtime_hours` | Downtime | machine-hours over 4 weeks | Hours machines stood still, planned services included. *Rough guide.* | Lower |
| `unit_cost` | Cost per unit | £ per unit | All costs divided by units made. | Lower |
| `stockout_probability` | Days short of stock | % of days | Share of days with some demand you could not supply. | Lower |
| `oee` | Equipment efficiency (OEE) | % | How much of the staffed time machines spent making good units at full speed; 100% is perfect. | Higher |

### Settings (16)

| Code | Section | Friendly name | Unit / range | Meaning | Note |
|---|---|---|---|---|---|
| `shift_count` | Shifts and overtime | Shifts per day | 1–3 eight-hour shifts | How many 8-hour shifts run each working day. | whole number |
| `overtime_hours` | Shifts and overtime | Overtime per day | 0–4 hours | Paid extra hours added to each working day. | |
| `throughput_multiplier` | Machine speed | Overall line pace | 70–130% of normal | Target production rate for the whole line. | |
| `speed_setpoint_M1`…`M5` | Machine speed | Speed of the cutting / forming / welding / finishing / packing machine (M1–M5) | 80–120% of normal | How fast this machine runs; faster can wear it sooner. | |
| `pm_interval_M1`…`M5` | Maintenance | Service interval, cutting / … / packing machine (M1–M5) | 24–720 running hours | Running hours between planned services. Shorter = serviced more often. | "about N working days" shown alongside |
| `reorder_point` | Stock and ordering | Reorder when stock falls to | 0–9,000 units | When stock on hand plus on order drops to this, order more. | |
| `order_quantity` | Stock and ordering | Order size | 100–10,000 units | How many units each order brings in. | |
| `safety_stock` | Stock and ordering | Safety stock | 0–4,000 units | Extra stock kept as a buffer against surprises. | |

Goal-fixed (never offered as settings, shown only in Technical details):
`supplier_choice` (Supplier), `price_multiplier` (Price level).

Plain-word dictionary (default view):

| Technical | Plain |
|---|---|
| surrogate prediction | quick estimate |
| verified on twin (N reps) | checked with the detailed simulation (N runs) |
| replayed benchmark result | saved example result, not a live run |
| novelty band | familiar / less familiar / unfamiliar territory |
| robustly feasible | meets your limits in every simulated month |
| constraint | limit |
| objective | what you want to improve |
| current policy | current setup |
| CI | likely between X and Y (help text only) |
| context | example month of orders |

---

## A8. Errors and empty states

Each says what happened and what to do next. Raw detail goes to Technical
details, with key-like text redacted as today.

| Situation | Message |
|---|---|
| No internet / AI service unreachable | "We couldn't reach the AI service, so no new plan was found. Check your internet connection and try again. Meanwhile, here is a saved example result for a similar goal." (+ saved example, labelled) |
| AI service busy (rate limit / timeout) | "The AI service is busy right now. Wait a minute and press Find a plan again." (card goals still run if only the summary failed: "The plan is ready, but the AI summary is unavailable, so a standard summary is shown.") |
| Goal not understood (own words) | "We couldn't turn that into a goal we can plan for. Try one of the goal cards above, or say what you want more or less of, for example: 'Make more profit with at most one breakdown a month'." |
| Planner asks a question | "We need one more detail: {question}" with the text box kept filled in. |
| Off-topic or unsafe request | "That isn't something this tool can plan. It finds production settings for your line." |
| No better plan | Chip ➖ + "Keep your current setup. The best plan found wasn't clearly better." Next step: "Try a different goal, or relax a limit." |
| Goal impossible | Chip ⛔ + closest options as "If you could accept on-time delivery of 85% instead of 99%, a plan exists." Next step: "Change a limit and press Find a plan again." |
| Limits contradict | "These two limits can't both be met: {a} and {b}. Change one of them." (before running) |
| Run stopped (app restarted) | "This run stopped before finishing, so there is no result. Press Find a plan to start again." |
| Run failed (other error) | "Something went wrong while working out the plan, and no result was produced. Try again; if it happens again, see Technical details." |
| Another run in progress | "A plan is already being worked out (started 14:02). It will appear on the Plan page when it's ready." |
| Example mode, Try changes | "Quick estimates and checks need the full installation. Start from a saved example's plan to see its checked numbers." |
| Past results empty | "Nothing here yet. Results you find on the Plan page appear here." |

---

## A9. Test plan

1. **Jargon ban on the default view.** AppTest renders every page and every
   saved example, every card's limit form, the progress view (with a fake
   job), and each A8 state. A helper walks the element tree and collects text
   (markdown, captions, metrics' labels/values/help, buttons, chips, table
   cells, widget labels and help) **skipping the subtrees of expanders labelled
   "Show technical details", "Show the exact test", "For reviewers", "Read the
   full section" and "Read the AI assistant's explanation"**. Banned
   (case-insensitive, word-bounded): surrogate, twin, epistemic, novelty,
   replication(s), hazard, seed, context, spec, lever(s), Sobol, OOD, benchmark,
   critic, objective, constraint, feasible, verified, plus `\bCI\b` (case-
   sensitive) and "confidence interval". "OEE" is allowed only inside the
   glossary's "Equipment efficiency (OEE)". The test fails with the offending
   word and where it appeared. A second test proves the walker finds a planted
   banned word, so the ban can't pass vacuously.
2. **Card → spec.** For every card × every combination of its limits on/off at
   defaults and at range ends: the view model builds an `ObjectiveSpec` that
   passes `ObjectiveSpec.from_dict(spec.to_dict())`, has the card's objective
   and sense, and the right `free_dims` for "Allow overtime? No". Contradicting
   limits produce the A8 message, not an exception. In example mode, AppTest
   presses each card's "Find a plan" and gets a result labelled "Saved example
   result, not a live run", with the "related goal" line exactly when the saved
   example's objective differs.
3. **Trust rule.** Unit tests on synthetic results for every branch (T1 fail;
   0/1/2/3 failed among T2–T4; T2 not applicable) and pinned levels for the
   saved examples (G07/AA1/N01/I01/G03 Checked, G10 Partly checked, G01/G09
   Rough guide). The reasons list always has four lines. No digit-percent
   ("%"), "probability" or "chance" appears in the trust block.
4. **Prompt 10 honesty rules, in the new wording:**
   - Quick estimate vs "checked with the detailed simulation (N runs)": never
     in one table or one card group; estimates never show "likely between";
     checked blocks always state N and have intervals in help text.
   - Every saved example shows "Saved example result, not a live run";
     Past results labels live and saved examples differently.
   - Side effects: G09 lists profit; a goal KPI is never listed; same rule
     tests as today, through the new Things to watch text.
   - Weak-KPI warning: the plain sentence appears whenever a breakdown or
     downtime number is shown in the default view, and the exact R² badge text
     is in Technical details.
   - No "optimal", "guaranteed", "proven" anywhere rendered (default *and*
     technical views), as today.
   - "verified" never used for explanation text (default and technical).
   - No epistemic std anywhere; familiarity derives from the novelty band only.
   - B1–B5 render as five cards with identical structure; both "Failed"
     chips present; B5 caveat visible.
5. **Background run.** Unit tests of the job object with a fake runner: one
   run at a time; progress steps in order; finished, failed and orphaned
   "started" entries all reach Past results; the lock is released on success
   and on exception. AppTest: a click elsewhere during a (fake, blocking) run
   does not abort it.
6. **Glossary completeness.** Every KPI in `KPI_NAMES` and every one of the 16
   settings has a glossary entry with friendly name, unit, meaning and
   direction; every slider and metric label on the pages comes from it.

---

## 11. Decisions for you before Stage B

1. **One background thread.** Prompt 10 said "no background workers". This
   spec uses exactly one worker thread in the Streamlit process, one run at a
   time, so clicks don't kill a run. Alternative: keep runs synchronous and
   lock every control during the run (a page change would still stop it).
   **Recommended: the thread.**
2. **"Why this plan" in the default view.** The AI's own text contains jargon
   (`downtime_hours`, "surrogate", "recommendation_reason", "T1 95th
   percentile"), and rewording it means changing the agent's prompt, which
   Stage B may not do. Proposed: a UI-generated plain summary in the default
   view, with the AI's text verbatim one click away ("Read the AI assistant's
   explanation"), with its check caption and gaps. Alternative: show the AI
   text in the default view and exempt it from the jargon test.
   **Recommended: the plain summary plus the expander.**
3. **Progress for card goals.** Card goals skip the planner and run through
   `run_on_spec`, which has no streaming variant; `run_agent_streaming` exists
   only for typed goals. Options: (a) add `run_on_spec_streaming` to
   `agents/graph.py` — a new entry point beside the existing one, no change to
   any node, tested to return exactly what `run_on_spec` returns (the same
   approach as `run_agent_streaming` in Prompt 10); (b) route card goals
   through the planner as generated text (costs an extra model call and
   re-introduces parsing risk). **Recommended: (a)**, but it touches
   `graph.py`, so it needs your OK under the UI-only rule.
4. **Demo cache additions:** G03 s2 and G10 s2, from existing cells, rebuilt
   with the existing builder (re-checked with checker v2). OK?
