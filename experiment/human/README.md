# normGym Human Experiment

A multi-page web study built on the original single-page web player (since
replaced by the free-play server `runner/human_play.py`). It keeps the
live, browser-playable normGym game (frame streaming, the rule-violation modal,
the timeline scrubber, replacement agents) and wraps it in a real experiment flow.

**The rules are hidden.** Participants are *not told* the norm — the whole point is
that they must **infer each rule themselves** from the `−20` "Rule Violation"
feedback. The flow is a shared intro, then several hidden norms in a row:

```
Welcome / consent
  → Setting           (the world + how to read characters: colours, "You", … dots, ✓ check)
  → Controls          (keys, scoring, and "there is a hidden rule")
  → Practice          (play once, a tutorial norm, to feel the violation feedback)
  → for each hidden test norm (×2 in the current study):
        Interactive round   (one continuous round to work the rule out)
        Test 1              (short trial applying it, same world as the learning round)
        New-place tour      (annotated picture of the new world: door / chair /
                             counter / carpet — the café tutorial, for that world)
        Test 2              (same, in a NEW-MAP world — does the rule transfer?)
                            each test is one of the pipeline's 3 runs of that
                            stage, drawn per participant
  → Thank-you
```

**Round endings differ by type.** Practice and the 2 Test rounds **end once your
own character finishes its goal and leaves through a door** (disappears from the
screen); a step cap is the backstop. The **Interactive** round is continuous — when
you finish and leave, a fresh character takes over and play keeps going; it runs
until the step cap **or** until you click the **“Finish round”** button, which only
becomes clickable **after you have completed your first character**. Gameplay uses
the **same render config as the free-play server `runner/human_play.py`** — 4 interpolation
sub-frames per step and the same playback speed.

Test norms are **assigned from a counterbalanced rotation** over the pool you
specify (see [§2.1](#21-the-real-study-6-participants--6-norms)), so the design
fills exactly instead of drifting; the **tutorial norm is excluded** from that
pool so the intro can't leak a test rule. Every page has a **Continue** button,
several participants can take it at the **same time** (game state is
per-connection), and **every action is logged** to `experiment/human/data/`.

---

## 1. Install

From the repo root:

```bash
pip install -r requirements.txt   # includes eventlet for smooth real-time frame streaming
```

Without `eventlet` the app falls back to threaded mode (frames arrive a little
batchier).

---

## 2. Run locally

Give it a **pool of test norms** and a **tutorial norm** for the intro. Each
participant does `--blocks` of the test norms (default 2), taken from a
counterbalanced rotation:

```bash
python experiment/human/app.py \
    --norm_types xcell_must_visit occupation_seat_no_bottom_left  movement_clockwise_move \
    --tutorial_norm straight_line \
    --world_type cafe
```

Open <http://localhost:5001>.

- The **tutorial norm** is used only for the Example + Practice intro and is
  **excluded** from the test pool (so it never leaks a test rule).
- For a single test norm, pass it and `--blocks 1`.

Each test norm is played as **1 Interactive round + 2 Test rounds** (3 episodes):
the llm/rl pipeline runs `num_tests` same-world tests and one test per world in
the norm's `norm_new_map_worlds`, and the human plays **one run of each of those
two stages** — one test in the world the norm was learned in, one in a new-map
world. **Which** of the pipeline's three runs of a stage a participant gets is
**drawn per participant** (uniformly, from a private RNG keyed by participant +
norm + stage), so the study spreads over all three settings instead of everyone
replaying run 1. The draw is stable: a reload, a server restart, or a returning
participant id replays the same run.

Because Test 2 happens somewhere the participant has never played, it is
preceded by a **tour of that world** (`/new-map-intro/<block>`), built the same
way as the tutorial's café layout page: an annotated picture with arrows to the
**door, chair, counter and grey carpet**. Nothing about it is hand-drawn — the
picture is that world's own board rendered by the game's renderer with no
characters on it (`/scene/<world>.png`, rendered once per world and cached), and
every arrow endpoint is computed from the cell it points at, so the page is
correct for whichever of the ten worlds a participant drew.

### 2.1 The real study: 25 norms × 2 participants, 25 people × 2 norms

The live design is **the 25-norm validation set** ([`norms_25.txt`](norms_25.txt):
half of each category — 8 queue, 6 region, 5 trajectory, 6 allocation — chosen
from the first wave's results), **25 participants, 2 norms each**, with
`straight_line` as the tutorial norm. (The first wave, 2026-09-12/13, ran the
same design over all 50 norms with 50 participants: `study_design.json`; pass
`DESIGN=experiment/human/study_design.json` to `run_study.sh` to serve it.)

- every norm is played by **exactly 2 participants** — once as scenario 1 and
  once as scenario 2, so order is counterbalanced per norm;
- a participant's two norms come from **different categories** (queue 16,
  region 12, trajectory 11, allocation 11 — the paper's four families, as tagged
  in `experiment/train/config.py:NORM_GROUPS`);
- **25 participants fill the design exactly**; a 26th wraps back to slot 1.

The assignment is the fixed table in [`study_design_25.json`](study_design_25.json),
built and verified by [`make_study_design.py`](make_study_design.py) (with
`--norms_file norms_25.txt`): one cycle over the norms with no two neighbours of
the same category, where slot *i* plays `cycle[i]` then `cycle[i+1]`. Just run
[`run_study.sh`](run_study.sh), which is the same as:

```bash
python experiment/human/app.py \
    --assignment design \
    --design_file experiment/human/study_design_25.json \
    --tutorial_norm straight_line \
    --world_type cafe
```

In design mode the norm pool and the number of norms per participant come from
the file (`--norm_types` may be omitted; if given it must list the same norms).
Which of the pipeline's three runs of each test stage a participant replays is
drawn per participant, as in every other mode. Every round's `round_start` event
and summary carry the norm's `norm_category`.

Slots are handed out **in arrival order, lowest free slot first**, and recorded
in `data/_assignments.json`:

- the ledger is **persistent**: restarting the server continues the design, and a
  returning `participant_id` keeps the norms it was first given;
- a **dropout** (someone who began but will not finish) should have their slot
  freed with `http://localhost:5001/admin/release?token=YOUR_TOKEN&pid=PID`
  (needs `ADMIN_TOKEN`); the next arrival takes that slot, so the design still
  fills. The freed entry is kept under `released` in the ledger; the
  participant's round data is untouched;
- progress check: `http://localhost:5001/admin/assignments?token=YOUR_TOKEN` →
  who got what, `free_slots`, `remaining_per_norm`, and `per_category` totals;
- a changed design file (or a changed pool/`--blocks` in balanced mode) archives
  the old ledger to `data/_assignments_<timestamp>.json` and starts the new
  design at slot 1. Regenerate the design only on purpose
  (`python experiment/human/make_study_design.py`; `--check` verifies the
  committed file, and `run_study.sh` runs that check before starting).

For a smaller pilot, `--assignment balanced` (the default without a design
file) walks a counterbalanced rotation over `--norm_types`: with *K* norms and
`--blocks B`, *K* participants cover every norm exactly *B* times, each norm
appearing once in each order position. `--assignment random` is the old
unbalanced behaviour (independent sample per participant).

**Seeds & parity with the llm/rl runs.** Rounds are seeded and built exactly like
`experiment/run/run_experiment.py`: the per-norm base seeds, capacity, and
step budgets come from `experiment/run/experiment_config.json`, and agents
are drawn with the pipeline's own builders (`experiment/run/common.py`) in
the same order. The interactive round uses the norm's base seed; same-world test
*k* uses base seed + *k*; new-map test *k* uses base seed + `num_tests` + *k*
(the pipeline's `seed_offset` for that stage). Whichever run *k* a participant
draws, they face the **identical setting** (same world, scripted characters,
entrances, targets, colors, arrival order, same subject slot and color) as the
llm/rl subject did in that exact run of norm X. Each round logs `stage_run` (the
*k* it drew) and `pipeline_run_id` — `test_002`, `new_map_test_001_library` —
which is the pipeline's own run id, so human rows join straight to llm/rl rows.

Common flags (each also has an env-var form for cloud deploys, in parentheses):

| flag | default | meaning |
|------|---------|---------|
| `--norm_types` | *(required)* | pool of hidden **test** rules (`NORM_TYPES`, space-separated) |
| `--tutorial_norm` | `straight_line` | rule used for the Example + Practice intro (`TUTORIAL_NORM`) |
| `--blocks` | `2` | how many test norms each participant does (`BLOCKS`) |
| `--assignment` | `balanced` | `balanced` = counterbalanced rotation in arrival order (each norm gets exactly `--blocks` participants per full pass); `design` = the fixed study design in `--design_file` (the real study, see 2.1); `random` = independent sample per participant (`ASSIGNMENT`) |
| `--design_file` | `experiment/human/study_design.json` | study design for `--assignment design` (`DESIGN_FILE`) |
| `--world_type` | `cafe` | scene: cafe, bank, library, … (`WORLD_TYPE`, falls back to the shared config) |
| `--capacity` | per-norm from shared config | max characters on the map (`CAPACITY`) |
| `--max_steps` | `max_steps` from shared config | step cap for the interactive round (`MAX_STEPS`) |
| `--test_max_steps` | `test_max_steps` from shared config | step cap per practice/test round (`TEST_MAX_STEPS`) |
| `--demo_max_steps` | `80` | hard cap on the autoplay Example round (`DEMO_MAX_STEPS`) |
| `--port` | `5001` | server port (`PORT`) |
| `--random_seed` | none | override the per-norm seeds from the shared config (leave unset to match the llm/rl runs) |

See `python experiment/human/app.py --help` for the full list.

---

## 3. Share a link with remote participants

The server runs on **your laptop**; a tunnel gives it a public `https://…` URL you
can paste into an email/Slack. The link stays alive as long as your laptop and the
tunnel are running.

### Option A — Cloudflare Tunnel (free, no account)

```bash
# install once:  brew install cloudflared   (mac)
# or download from https://github.com/cloudflare/cloudflared/releases

# terminal 1 — start the study server (the real study)
python experiment/human/app.py \
    --norm_types edge_left_line movement_clockwise_move \
    --tutorial_norm straight_line --blocks 2

# terminal 2 — open a public link
cloudflared tunnel --url http://localhost:5001
```

`cloudflared` prints a line like
`https://random-words.trycloudflare.com` — that's the link to send participants.

**Whose laptop runs what?** The whole game runs on *your* laptop; participants only
need a browser. Their keypresses go to your machine, frames come back, and
**their data is written to `data/` on your laptop** — all participants collect
in one place. The catch: the link only works while your laptop is awake and the
tunnel is running.

### Option B — ngrok

```bash
# install once:  brew install ngrok   (and add your authtoken once)
ngrok http 5001
```

### One-command helper

```bash
# THE REAL STUDY (2 norms / 2 participants / both each) + a public cloudflared link:
./experiment/human/run_study.sh

# same, local only:
SHARE=0 ./experiment/human/run_study.sh
```

`run_study.sh` just pins the study flags (`--norm_types` … `--tutorial_norm
straight_line --blocks 2 --assignment balanced`) plus the Prolific
`COMPLETION_CODE`, and defers to `run.sh`, which you can also call directly with
your own pool:

```bash
# local only:
./experiment/human/run.sh --norm_types movement_clockwise_move occupation_seat_in_order xcell_must_visit

# start server AND a public link (needs cloudflared or ngrok installed):
SHARE=1 ./experiment/human/run.sh --norm_types movement_clockwise_move occupation_seat_in_order xcell_must_visit
```

---

## 4. Deploy an always-on link on Render (recommended — no laptop needed)

This gives you a permanent `https://<name>.onrender.com` link that runs 24/7 on
Render's servers. **Your laptop does not need to be on.** You deploy once; data is
saved on Render and you download it with one click (section 4.2).

The repo already ships everything needed: `experiment/human/Dockerfile`,
`experiment/human/render.yaml`, and the in-app data-download endpoint.

### 4.1 One-time setup

1. **Push this repo to GitHub** (Render deploys from a GitHub repo).
2. In the [Render dashboard](https://dashboard.render.com): **New → Web Service**,
   connect your repo.
3. Configure:
   - **Runtime / Language:** `Docker`
   - **Dockerfile Path:** `experiment/human/Dockerfile`
   - **Docker Build Context Directory:** `.` (the repo root — the image needs `src/`)
   - **Instance Type:** `Starter` or higher. *(Not Free — Free sleeps after ~15 min
     and is too slow for the matplotlib rendering. Starter stays always-on.)*
   - **Health Check Path:** `/health`
4. **Add a Disk** (so data survives restarts/redeploys):
   - Mount Path: `/app/experiment/human/data`  •  Size: `1 GB`
   - The disk also holds `_assignments.json`; without it a redeploy would restart
     the balanced rotation at participant 1 and unbalance the design.
5. **Environment variables** (Add from the dashboard):

   | key | example value |
   |---|---|
   | `NORM_TYPES` | `edge_left_line movement_clockwise_move movement_no_odd_turn occupation_door_all_exit_A occupation_seat_class_specific xcell_forbidden` |
   | `TUTORIAL_NORM` | `straight_line` |
   | `BLOCKS` | `2` |
   | `ASSIGNMENT` | `balanced` |
   | `WORLD_TYPE` | `cafe` |
   | `SECRET_KEY` | *(click “Generate”)* |
   | `ADMIN_TOKEN` | *(click “Generate” — needed to download data)* |

6. **Create Web Service.** First build takes a few minutes. When it's live, Render
   shows your URL — that's the link you send participants. Their data is recorded
   on Render automatically (one shared pool), no matter whose laptop they use.

> Prefer not to click through the form? Copy `experiment/human/render.yaml` to the
> **repo root**, then use **New → Blueprint** — Render reads all of the above from
> that file (it auto-generates `SECRET_KEY` and `ADMIN_TOKEN` for you).

### 4.2 Download your data (no SSH needed)

The app exposes two **token-protected** endpoints. Grab the `ADMIN_TOKEN` value from
your Render service's Environment tab, then in your browser:

- **All data as a zip:**  `https://<your-app>.onrender.com/admin/results?token=YOUR_TOKEN`
  (logs only; add `&images=1` to also bundle the saved step images — much larger)
- **Just list the files:** `https://<your-app>.onrender.com/admin/list?token=YOUR_TOKEN`
- **Assignment progress:** `https://<your-app>.onrender.com/admin/assignments?token=YOUR_TOKEN`

(Without a valid `?token=` these return 403, so participants can't reach your data.)

### Alternative — any VM / Docker host

```bash
# build from the repo root (the image needs src/)
docker build -f experiment/human/Dockerfile -t normgym-exp .

docker run -p 5001:5001 \
  -e NORM_TYPES="movement_clockwise_move occupation_seat_in_order xcell_must_visit" \
  -e TUTORIAL_NORM="straight_line" \
  -e ADMIN_TOKEN="pick-a-long-secret" \
  -v $(pwd)/experiment/human/data:/app/experiment/human/data \
  normgym-exp
```

The `-v` mount keeps participant data on the host. On a cloud VM you'd typically
also put a reverse proxy (nginx/caddy) with TLS in front of port 5001.

---

## 5. Collected data

Everything lands in `experiment/human/data/` (git-ignored). File names encode
the participant, block, norm, phase, and episode:

- `_assignments.json` — study-wide ledger (not per participant): the
  counterbalanced schedule, the next slot to hand out, and every participant id
  with its slot and norms. Deleting it restarts the rotation at participant 1.
- `<pid>_assignment_<ts>.json` — the participant's assigned **test** norms in the
  order they played them, the tutorial norm, and age. One per participant.
- `<pid>_b<block>_<norm>_<stage>_ep<episode>_<ts>.jsonl` — one JSON object per
  event: `session_start`, `round_start`, `await_input`, `action`, `step`,
  `game_over`. Each `step` record carries the player's position, target, per-step
  reward, cumulative reward, collisions, running violation count, and whether the
  player broke the rule that step. Each `action` record is the raw key pressed.
- `..._summary.json` — final totals for that round (block, episode, norm, steps,
  collisions, violations, final reward, duration, frames saved).
- Intro rounds are named `<pid>_demo_<norm>_<ts>` and `<pid>_practice_<norm>_<ts>`.

Step images are **not** kept in `data/` — they go to a separate git-ignored
folder, `experiment/human/data_figures/`, with one subfolder per participant:

- `data_figures/<pid>/norm<n>_<norm>/<interactive|test<k>|new_map_test<k>>/step_NNN.png`
  (the `<k>` is the pipeline run the participant drew, not a count of rounds)
  — a picture of **every step the participant played**, so you can see the
  actions they chose. One image per step (the settled position at the end of
  that step), numbered to match the `step` records in that round's `.jsonl`.
  Written only when a round completes, so there are no half-round folders; a
  replayed round moves the old images to `<round>_prev_<ts>/`. Practice and demo
  rounds are not saved. Budget roughly **7 MB per test round** and **13 MB per
  interactive round** — with 2 norms × (1 interactive + 2 tests) that is ~55 MB
  per participant (~330 MB for a 6-participant study).

`<stage>` is `demo` (Example), `practice`, `interactive` (the per-norm continuous
round), `test` (the same-world trial), or `new_map_test` (the new-world trial) —
the same stage names the llm/rl pipeline uses. `<block>` is the 0-indexed
test-norm position; `<episode>` is 0 for the interactive round, 1 for the
same-world test, and 2 for the new-map test. Each round's `session_start` /
`round_start` / summary also carry `stage_run` (which of the pipeline's runs of
that stage was drawn) and `pipeline_run_id` (that run's pipeline name).

To analyze the test trials, e.g.:

```python
import json, glob
for f in sorted(glob.glob("experiment/human/data/*_test_ep*_summary.json")):
    s = json.load(open(f))
    print(s['participant_id'], 'norm', s['norm_type'], 'trial', s['episode'],
          '-> violations', s['norm_violations'], 'reward', s['final_reward'])
```

### Metrics

`experiment/human/eval/eval.py` computes the same metrics as the LLM pipeline's
`experiment/evaluation/eval.py`, so human and model numbers are directly
comparable. It reads `data/` and writes one
`eval/results/metrics_summary_<norm>.json` per norm:

```bash
python experiment/human/eval/eval.py            # every participant in data/
python experiment/human/eval/eval.py --judge    # + NormBeliefScore (LLM judge)
```

See [eval/README.md](eval/README.md) for the metric definitions.

---

## 6. How it maps to the original single-page player

| concern | single-page player | this app |
|---|---|---|
| pages | single page | welcome / setting / controls / practice / 6 main episodes / done |
| rule | shown | **hidden** — inferred from the `−20` violation feedback |
| structure | one episode | shared intro + (1 interactive + 2 test) × 2 hidden norms |
| sessions | one global game | one `Session` per Socket.IO connection (concurrent) |
| logging | none | full per-participant JSONL + summary, per episode |
| example demo | n/a | autoplay round with **no** human (oracle characters only) |
| game engine | — | identical frame-stream / modal / replacement logic, ported verbatim |

The game engine itself (`_game_loop`, frame streaming, violation timing) is copied
from the original and only re-scoped from a global dict to a per-connection
`Session`, so gameplay behaves exactly as before.
