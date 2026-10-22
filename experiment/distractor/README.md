# Distractor experiment

Question: on norms a model already solves 100% violation-free in both test
stages, does it still infer the real norm when every background agent also
shows an irrelevant regularity — or does it report that regularity as the norm?
Nobody is ever penalised for the distractor: the checker only knows the run's
real norm.

Nothing in `src/` or `experiment/run/` is modified. `run_distractor.py`
calls the original `run_experiment.main()` (same CLI, config, seeds, 200-step
interactive stage, 3 same-world + 3 new-map tests, eval and belief judge) and
applies the distractor in memory. All results go to
`experiment/results_distractor/<agent_type>/<model>/cafe/<norm>/`, and each norm
directory gets a `distractor.json` saying what was applied.

## GPT-5.6 (`DISTRACTOR=auto`, the default)

The 8 norms GPT-5.6 solved 100% in both stages (`movement_move1_wait1`, the
distractor rhythm itself, excluded):

| Norms | Background agents | Subject |
|---|---|---|
| `movement_no_turn_mod3_0`, `movement_move2_wait1`, `movement_move3_wait1` | all exit through the **right door** (entrances stay random) | exit door is a **free choice**: prompt says "leave through a door", any door completes the goal |
| `occupation_door_A_forbidden`, `occupation_door_B_forbidden`, `occupation_door_class_specific_BCA`, `occupation_door_class_specific_CAB`, `occupation_seat_no_bottom_left` | all keep the **`movement_move1_wait1` rhythm** (never two moves in a row) on top of the real norm | unchanged |

Reading the results:
- exit-right norms differ from the baseline in two ways (distractor + free-exit
  prompt line). `DISTRACTOR=none FREE_EXIT=1` runs the free-exit control with
  random background exits (`experiment/results_distractor_control/`); without it
  a high right-door share cannot be told apart from "the right door was nearest".
- exit-right runs are seed-identical to the baseline except the background
  exits. Rhythm runs start identical, then diverge because the pace differs.
- baseline beliefs can already mention exiting or pacing, so `analyze.py` runs
  its keyword screen on the baseline too as the control. Read what it flags.

## Results: GPT-5.6, 8 norms (2026-09-19)

One run per norm (`openai/gpt-5.6-sol`, `memory_llm`, temperature 0.7, same
seeds and config as the baseline in `experiment/results/`). Jobs 630240 and
630469, no failed API calls, $38.56 in total. Free oracle-subject sanity runs of
all 8 setups (`results_distractor/oracle_with_norm/`) reach 100% success with 0
violations, so the distractors do not cause congestion by themselves.

Violation-free success (VFS) is over 3 runs per stage. "Judge" is the
norm-belief judge score of the final interactive belief, the only thing the
subject carries into the tests. Every cell is baseline → distractor.

| Norm | Distractor | Test VFS | New-map VFS | Judge | Interactive violations |
|---|---|---|---|---|---|
| `movement_move2_wait1` | exit-right + free exit | 100% → 100% | 100% → 100% | 0.67 → 1.00 | 13 → 6 |
| `movement_move3_wait1` | exit-right + free exit | 100% → 100% | 100% → 100% | 1.00 → 1.00 | 4 → 4 |
| `movement_no_turn_mod3_0` | exit-right + free exit | 100% → 100% | 100% → 100% | 1.00 → 0.83 | 6 → 6 |
| `occupation_door_class_specific_BCA` | rhythm | 100% → 100% | 100% → 100% | 1.00 → 1.00 | 1 → 0 |
| `occupation_door_class_specific_CAB` | rhythm | 100% → 100% | 100% → **67%** | 0.83 → 1.00 | 1 → 2 |
| `occupation_seat_no_bottom_left` | rhythm | 100% → **67%** | 100% → **0%** | 0.50 → 0.50 | 2 → 2 |
| `occupation_door_A_forbidden` | rhythm | 100% → **0%** | 100% → **0%** | 0.50 → 0.17 | 2 → 1 |
| `occupation_door_B_forbidden` | rhythm | 100% → **0%** | 100% → **33%** | 0.83 → 0.17 | 1 → 0 |

Task success stayed 100% in every test run: where VFS dropped, the subject
finished its goals and broke the norm on the way (one violation per failed run).

### Findings

1. **GPT never reported a distractor as the norm.** The keyword screen in
   `analyze.py` matched 0% of the interactive-step beliefs in all 8 distractor
   runs (and in the baselines): no belief states a right-door rule, and none
   states a move/wait rhythm under the rhythm distractor. So the failures below
   are not GPT reporting the distractor as the norm; they are GPT failing to
   find a weakly signalled norm (finding 3).

2. **The exit-right distractor had no effect (3 movement norms).** VFS stayed
   100% and the final beliefs are the true norm. With its exit door a free
   choice GPT did not copy the background agents either: it left through the
   right door in 5 of 19 interactive lives and 4 of 18 test runs, no more than
   the random door assignment of the baseline (5 of 16 and 6 of 18). These norms
   give dense feedback (a penalty on most moves until the rhythm is learnt), and
   GPT locks onto the rule early: under `movement_move2_wait1` its last
   interactive violation was at step 18 with the distractor and step 25 without.

3. **Under the rhythm distractor the outcome split by how the true norm shows.**
   - Class-specific door norms (BCA, CAB) were robust: the colour → door mapping
     is visible on every background exit, and the final beliefs are correct and
     almost word for word the baseline's. CAB's one new-map violation happened
     with a correct belief.
   - The prohibition norms (`door_A_forbidden`, `door_B_forbidden`,
     `seat_no_bottom_left`) collapsed. What they forbid is only visible as
     something nobody does, so GPT learns them mainly from its own penalties,
     and it ended with wrong beliefs:

     | Norm | Baseline final belief | Distractor final belief |
     |---|---|---|
     | `door_A_forbidden` | each colour class has its own door (wrong, but compliant in the tests) | "doors are not restricted by social group" |
     | `door_B_forbidden` | "the bottom doorway is for entering the cafe, not leaving" (correct) | "respect first-come queue order and right-of-way ... yield to earlier or current occupants" |
     | `seat_no_bottom_left` | do not sit at a single-seat table (wrong, but compliant) | "chairs are assigned by social group" |

   - In the `door_B_forbidden` distractor run GPT got **no** penalty in the
     interactive stage: it left through the right door in all 6 lives, never
     tried the bottom door, and so never received the signal that taught it the
     norm in the baseline. Its belief there is the closest thing to distractor
     uptake in the data: background agents pausing every other step appear to
     have been read as yielding / queueing. The keyword screen does not catch
     this, so read the beliefs rather than trusting that column alone.

## Opus 5 (`DISTRACTOR=exit_bottom`)

All background agents exit through the bottom door; the 13 norms Opus 5 solved
100% (`occupation_door_all_exit_B` excluded). Blocked since 2026-09-19: the
provider refuses the pipeline's prompt for Opus (content_filter, "reverse
engineering or duplicating model outputs"), so no run exists.

## Files

| File | Purpose |
|---|---|
| `run_distractor.py` | launcher; norm -> distractor table; refuses the baseline results tree (a run wipes its norm directory) |
| `verify_setup.py` | proves each run differs from the baseline only as intended; no LLM calls |
| `analyze.py` | baseline vs distractor: violation-free success, judge score, distractor rule in the beliefs, doors the subject left through |

Per norm, run it and then evaluate the raw runs it reports:

    python experiment/distractor/run_distractor.py --norm_type movement_move2_wait1 \
        --agent_type memory_llm --provider openrouter --model <model>
    python experiment/evaluation/eval.py <raw_runs dir printed by the run>
    # --agent_type oracle_with_norm: a free sanity run (no LLM calls)
    python experiment/distractor/analyze.py --show_beliefs
