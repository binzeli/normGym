# LLM-judge validation study (human ratings of judged norm beliefs)

A small web study that checks the LLM judge used in the paper. Participants
read the **true norm**, an AI player's **belief** about it, and the LLM
judge's 1–7 rating of that belief with its reason, and say how much they
**agree with the judge** on a 1–5 scale. Two interface modes (`MODE=` /
`--mode`): **`two`** (wave 1) first asks for the participant's own 1–7 rating,
then reveals the judge and asks the agreement question; **`agree`** (wave 2,
default) shows the judge up front and asks only the agreement question.

```
Welcome / consent (Prolific id)
  → Instructions        (the setting, the judge's scale, the agreement question)
  → Item 1 … 10         (true rule + guess + judge's rating and reason → agree 1–5, optional comment)
  → Done                (completion code)
```

Design: **30 participants × 10 items = 300 distinct beliefs**, each rated by
one person. The 300 are drawn from `norm_beliefs_500.txt` with an equal quota
per judge Likert level (42–43 each) and, within a level, round-robin over the
4 norm families. Excluded from the pool: beliefs that are literally `none` (a
trivial 1/7) and beliefs over **70 words**, so each item is readable at a
glance. The true rule is shown in a **short plain-language form**
(`norm_short.json`: the rule without cell coordinates, quadrant boundaries or
the exact queue path; all under 90 words). The LLM judge saw the full text;
both are kept in the design and the data (`ground_truth` vs
`ground_truth_short` / `ground_truth_shown`). Each participant's 10 items span
the judge's scale and all four families; their order is random.

### Wave 2 (follow-up after the first 18 participants)

`study_design_wave2.json`: 12 slots × 10 items over all four families and
all seven judge levels (17–18 per level), drawn from items nobody rated in
wave 1 (ratings by the excluded fast participant do not count); `--allow_rerate`
lets a level fall back to wave-1 items for a second rater if it runs out of
fresh ones (the design file lists any under `rerated_entries`). Built with:

```bash
python experiment/human_judge/make_study_design.py --n_participants 12 \
  --exclude_rated_in experiment/human_judge/data --ignore_rater 11b --allow_rerate \
  --out experiment/human_judge/study_design_wave2.json
```

Serve it with `DESIGN=experiment/human_judge/study_design_wave2.json ./experiment/human_judge/run.sh …`.
Each design has its own slot ledger (`data/_assignments.json` for
`study_design.json`, `data/_assignments_<design stem>.json` otherwise), so you
can switch between waves and each one resumes at its lowest free slot.

To continue wave 1 (two-question interface, remaining slots of the 30-slot design):

```bash
MODE=two ADMIN_TOKEN=some-secret SHARE=1 ./experiment/human_judge/run.sh
```

## Files

| file | role |
|---|---|
| `norm_beliefs_500.txt` | the source dump: 500 judged beliefs (model, norm, belief, ground truth, judge Likert + reason) |
| `parse_beliefs.py` | parses the txt into item dicts and attaches the short rule text (`--out items.json` to dump) |
| `norm_short.json` | the 50 plain-language rule texts shown to participants (edit here; then regenerate the design) |
| `make_study_design.py` | samples the 300 items and deals them to 30 slots → `study_design.json`; `--check` verifies it |
| `study_design.json` | the wave-1 design (30 slots); `study_design_wave2.json` the 12-slot follow-up. Regenerate with `--seed`, `--raters_per_item`, `--families`, `--judge_levels`, `--exclude_rated_in` … |
| `app.py` | the Flask app |
| `templates/`, `static/` | pages and stylesheet |
| `run.sh` | start the server (+ `SHARE=1` public link via cloudflared/ngrok) |
| `export_results.py` | flattens `data/` to `results/ratings.csv` and computes the agreement statistics |
| `Dockerfile`, `render.yaml` | always-on deployment (Render) |
| `data/` | participant data (git-ignored) |

## Run

```bash
# local: http://localhost:5002
python experiment/human_judge/app.py

# the real study: completion code + admin token, public link, Mac kept awake
COMPLETION_CODE=<your code> ADMIN_TOKEN=some-secret SHARE=1 ./experiment/human_judge/run.sh
```

Slot assignment is by arrival order (lowest free slot) and persisted to
`data/_assignments.json`, so restarting the server continues the design. A
returning participant id resumes at the first item they have not answered.

Admin endpoints (need `ADMIN_TOKEN`):

| URL | what |
|---|---|
| `/admin/assignments?token=…` | who holds which slot, how many items answered, free slots |
| `/admin/release?token=…&pid=PID` | free a dropout's slot so the next arrival takes it |
| `/admin/results?token=…` | zip of everything under `data/` |
| `/admin/csv?token=…` | one flat CSV row per rating |

## Data

Per participant, under `data/<pid>/`:

* `<pid>_begin_<ts>.json` – assignment + user agent
* `<pid>_itemNN_<ts>.json` – one file per rating: `entry`, `norm`, `family`,
  `model`, `belief`, `ground_truth_shown`, `judge_likert`, `judge_reason`,
  **`agree_with_judge`** (1–5), `human_likert` (1–7 in mode `two`, `null` in mode `agree`),
  `mode`, `design_file`, `comment`, and browser timings (ms from page load to
  own rating / reveal / answer / submit)
* `<pid>_summary_<ts>.json` – all ratings in one file, written when they finish

Analysis:

```bash
python experiment/human_judge/export_results.py       # -> results/ratings.csv, results/stats.json
```

`stats.json` reports, overall and per family: the distribution and mean of the
agreement rating and its mean per judge level, plus, for wave-1 records that
have a human 1–7 rating, Spearman/Pearson with the judge, exact and within-1
agreement, and the mean human rating per judge level.
