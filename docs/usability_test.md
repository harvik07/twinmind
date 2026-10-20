# TwinMind usability test: three tasks and an observation sheet

**Who:** one person who runs or manages production and has no data-science
background. **Where:** a laptop with the app in **demo mode**
(`python -m streamlit run app/main.py -- --demo`), opened on the Plan page,
browser at full screen. **How long:** about 20 minutes.

**Facilitator rules:** read each task aloud exactly as written, then say
nothing that helps. If asked, reply "What would you do if I weren't here?".
Ask the tester to think aloud. Stop a task at the time limit and mark it "not
completed"; that is a finding, not a failure of the tester.

---

## The three tasks (read aloud)

**Task 1 — Get a plan (limit: 5 minutes).**
"You want the line to make more money this month. Use the app to get a
recommendation. When you have it, tell me in your own words what it says you
should change and what you would gain."

*Success:* the tester chooses *Make more profit*, presses the main button, and
states at least one setting change and the profit before → after, in their own
words.

**Task 2 — Decide whether to trust it (limit: 4 minutes).**
"Before you act on this, would you trust it? Tell me how sure the app is, and
one thing it says you should watch out for."

*Success:* the tester finds *How much to trust this* and names the level
(Checked / Partly checked / Rough guide) with one reason, and names one item
from *Things to watch* (for example, that limits are met only on average, or
that breakdown numbers are rough).

**Task 3 — Try your own change (limit: 6 minutes).**
"Someone suggests servicing the welding machine less often. Find where you'd
try that, and tell me whether you can see what it would do, and how reliable
that would be."

*Success:* the tester opens *Try changes*, finds the welding machine's service
interval under *Maintenance*, and explains in their own words that demo mode
can't estimate a new change (or, in live mode, reads the quick estimate and
says it is rough and should be checked with the detailed simulation).

---

## Observation sheet

Tester code: ______  Date: ______  Facilitator: ______  Mode: demo / live

Prior experience with planning software: none / some / a lot

| | Task 1: get a plan | Task 2: trust it | Task 3: try a change |
|---|---|---|---|
| Completed? (yes / with hesitation / no) | | | |
| Time taken (min:sec) | | | |
| First click (what, where) | | | |
| Wrong turns (count, and where) | | | |
| Asked for help? (what about) | | | |
| Words they didn't understand (exact words) | | | |
| Said the result in their own words? (quote) | | | |
| Misread a number or label? (which) | | | |
| Ease, 1 (very hard) to 5 (very easy), their rating | | | |

**Watch for, and tick if seen:**

- [ ] Didn't notice the demo-mode banner, or thought their limits were used
- [ ] Didn't notice "Saved example result, not a live run"
- [ ] Didn't scroll past the headline sentence
- [ ] Confused *Checked*, *Partly checked* and *Rough guide*
- [ ] Read a "rough guide" number as certain
- [ ] Missed *Things to watch*
- [ ] Opened *Show technical details* (note why)
- [ ] Opened the AI-written explanation and found it confusing
- [ ] Looked for an "undo" or "back" that wasn't there
- [ ] Used the Glossary in the sidebar

**After the tasks (ask, then write their words):**

1. "In one sentence, what is this app for?"
2. "What, if anything, would stop you using it at work?"
3. "Was there a word or a screen that made you stop and think?"
4. "If you could change one thing, what would it be?"

**Overall, from the facilitator:**

- Biggest problem seen: ______________________________________________
- Second biggest: ____________________________________________________
- Something that worked better than expected: _________________________
