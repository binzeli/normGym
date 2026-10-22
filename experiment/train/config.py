"""Shared configuration for the norm-inference RL fine-tuning pipeline.

The subject the policy plays is the pipeline's ``memory_llm`` agent: the same
prompts, the same ``{reasoning, action, belief, confidence}`` JSON response,
the same reward values. Nothing under ``src/`` or ``experiment/run`` is
modified — training only *drives* the existing pipeline.
"""
import json
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
TRAIN_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Norm groups
# ---------------------------------------------------------------------------
#
# The 50 norms in src/norms/ fall into the paper's four families ("groups"),
# Table 1 of the paper: Queue (16), Region (12), Trajectory (11),
# Allocation (11). Every training version (v1/v2/v3) is run once per group;
# nothing is shared across groups:
#   data/<group>/…                      snapshots, judge records, SFT jsonl
#   checkpoints/sft/<group>/…           SFT cold starts (`current` -> the one RL uses)
#   checkpoints/v<N>_<name>/<group>/<RUN_NAME>/   RL adapters
#   plots/<group>/                      curves
# The active group is chosen with the NORM_GROUP environment variable
# (default: queuing); the launchers export it, so every script that imports
# TRAIN_NORMS / TEST_NORMS from here sees the right split.
#
# Each group: 'train' / 'test' lists + a 'category' tag. 'test' norms are
# never seen in training (unseen variants of seen subfamilies, held out one
# per subfamily like the 2026-08-26 queuing split); they are only used
# offline (eval_checkpoint.py, v3 eval_records.py). 'category' picks the
# category-specific prompt wording — the v2 identification evidence hint
# (vigal/v2_belief_rloo/identification.py:EVIDENCE_HINTS) and the v3
# observer prompt (vigal/v3_violation_judge/judge_task.py:CATEGORY_HINTS) —
# because the queue wording assumes the goal is the counter, which does not
# hold for the other families.

NORM_GROUPS = {
    # Queue — the 16 queuing (line) norms from src/norms/lining.py
    # (straight_line is the human-study tutorial norm and is excluded).
    # Split 2026-08-26: 11 train / 5 held-out test, one held-out per line
    # family with mixed handedness.
    'queuing': {
        'category': 'queue',
        'train': [
            'zigzag_right1_line',
            'zigzag_right2_line',
            'zigzag_left1_line',
            'bend_right1_line',
            'bend_left1_line',
            'bend_left2_line',
            'edge_left_line',
            'edge_right_line',
            'square_right_line',
            'staircase_left_line',
            'lshape_right_line',
        ],
        'test': [
            'zigzag_left2_line',
            'bend_right2_line',
            'square_left_line',
            'staircase_right_line',
            'lshape_left_line',
        ],
    },
    # Region — which cells the agent may end up in; decidable from the
    # entered cell alone. 12 norms across 5 subfamilies: all-exit doors (3),
    # forbidden doors (3), seats-only regions (3), no-seats regions (2),
    # forbidden zone (1). Split 2026-09-07: 8 train / 4 test, one held-out
    # per multi-member subfamily; xcell_forbidden has no in-family sibling
    # (xcell_must_visit is a Trajectory norm) so it stays in train.
    'region': {
        'category': 'region',
        'train': [
            'occupation_door_all_exit_A',
            'occupation_door_all_exit_C',
            'occupation_door_A_forbidden',
            'occupation_door_B_forbidden',
            'occupation_seat_bottom_left_only',
            'occupation_seat_top_left_only',
            'occupation_seat_no_bottom_left',
            'xcell_forbidden',
        ],
        'test': [
            'occupation_door_all_exit_B',
            'occupation_door_C_forbidden',
            'occupation_seat_bottom_right_only',
            'occupation_seat_no_bottom_right',
        ],
    },
    # Trajectory — the form of the agent's own motion; decidable from its own
    # transition, the clock, or its own history. 11 norms across 5
    # subfamilies: circulation (2), turn parity (2), turn mod 3 (3),
    # move-k-wait-1 (3), visit-zone-first (1). Split 2026-09-07: 7 train /
    # 4 test, one held-out per multi-member subfamily; xcell_must_visit is a
    # singleton so it stays in train.
    'trajectory': {
        'category': 'trajectory',
        'train': [
            'movement_clockwise_move',
            'movement_no_even_turn',
            'movement_no_turn_mod3_0',
            'movement_no_turn_mod3_2',
            'movement_move1_wait1',
            'movement_move3_wait1',
            'xcell_must_visit',
        ],
        'test': [
            'movement_counterclockwise_move',
            'movement_no_odd_turn',
            'movement_no_turn_mod3_1',
            'movement_move2_wait1',
        ],
    },
    # Allocation — how a contested resource is divided among co-present
    # agents, by identity, count, or order. 11 norms across 5 subfamilies:
    # zone capacity (2), single-class zone (1), class doors (3), class
    # seating (3), seats-in-order (2). Split 2026-09-07: 7 train / 4 test,
    # one held-out per multi-member subfamily; the single-class zone norm is
    # a singleton so it stays in train.
    'allocation': {
        'category': 'allocation',
        'train': [
            'occupation_region_capacity_1',
            'occupation_region_single_class_only',
            'occupation_door_class_specific',
            'occupation_door_class_specific_BCA',
            'occupation_seat_class_specific',
            'occupation_seat_class_specific_rot1',
            'occupation_seat_in_order',
        ],
        'test': [
            'occupation_region_capacity_2',
            'occupation_door_class_specific_CAB',
            'occupation_seat_class_specific_rot2',
            'occupation_seat_in_order_right',
        ],
    },
}

NORM_GROUP = os.environ.get('NORM_GROUP', 'queuing')
if NORM_GROUP not in NORM_GROUPS:
    raise KeyError(f'NORM_GROUP={NORM_GROUP!r} is not one of {sorted(NORM_GROUPS)}')
if NORM_GROUPS[NORM_GROUP]['train'] is None:
    raise ValueError(f'norm group {NORM_GROUP!r} has no train/test split yet — '
                     f'fill it in experiment/train/config.py:NORM_GROUPS')

TRAIN_NORMS = list(NORM_GROUPS[NORM_GROUP]['train'])
TEST_NORMS = list(NORM_GROUPS[NORM_GROUP]['test'])
# The active group's norm-category tag; the default for every category-
# specific prompt knob (v2 --category, v3 harvest --category).
NORM_CATEGORY = NORM_GROUPS[NORM_GROUP]['category']

# ---------------------------------------------------------------------------
# Base model / paths
# ---------------------------------------------------------------------------

# Switched from Qwen3-VL-8B-Instruct on 2026-08-19 (see RUN_LOG.md): same
# SigLIP2 vision encoder, ~half the LLM cost per update. The abandoned 8B
# lineage lives in checkpoints/{sft,grpo_vllm}; the 4B lineage uses *_4b dirs.
BASE_MODEL = 'Qwen/Qwen3-VL-4B-Instruct'

DATA_DIR = os.path.join(TRAIN_DIR, 'data')
GROUP_DATA_DIR = os.path.join(DATA_DIR, NORM_GROUP)          # data/<group>/
SFT_DATASET = os.path.join(GROUP_DATA_DIR, 'sft_train.jsonl')
SFT_VAL_DATASET = os.path.join(GROUP_DATA_DIR, 'sft_val.jsonl')
CKPT_DIR = os.path.join(TRAIN_DIR, 'checkpoints')
SFT_OUTPUT_DIR = os.path.join(CKPT_DIR, 'sft', NORM_GROUP, 'sft')   # checkpoints/sft/<group>/<name>
ROLLOUT_SCRATCH_DIR = os.path.join(TRAIN_DIR, 'rollouts')

# Existing API-model runs used as the SFT cold-start corpus
# (experiment/results/memory_llm/<model_dir>/<world>/<norm>/...).
RESULTS_ROOT = os.path.join(PROJECT_ROOT, 'experiment', 'results')
TRACE_AGENT_KIND = 'memory_llm'
TRACE_MODEL_DIRS = [
    'anthropic_claude-opus-5',
    'openai_gpt-5.6-sol',
]

# ---------------------------------------------------------------------------
# Environment settings (mirrors experiment/run/run_experiment.py defaults)
# ---------------------------------------------------------------------------

EXPERIMENT_CONFIG_PATH = os.path.join(
    PROJECT_ROOT, 'experiment', 'run', 'experiment_config.json'
)


def load_experiment_config():
    with open(EXPERIMENT_CONFIG_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


_EXP_CONFIG_CACHE = None


def benchmark_capacity(norm):
    """The norm's crowd size in the benchmark: experiment_config.json's
    per-norm ``norm_capacity`` override, else its global ``capacity``.
    (Same resolution as experiment/train/spiral/episode_trainer.py.)"""
    global _EXP_CONFIG_CACHE
    if _EXP_CONFIG_CACHE is None:
        _EXP_CONFIG_CACHE = load_experiment_config()
    return int(_EXP_CONFIG_CACHE.get('norm_capacity', {}).get(norm)
               or _EXP_CONFIG_CACHE['capacity'])


def default_env_kwargs():
    """The exact env kwargs run_experiment.py passes to make_env."""
    return {
        'use_sprites': True,
        'use_isometric': False,
        'entrance_side': 'bottom',
        'map_type': 'default',
        'interpolation_steps': 0,
    }


# RL rollout episodes are shortened interactive stages. The evaluation
# pipeline keeps its own max_steps (200, from experiment_config.json); this
# only bounds training episodes so rollouts stay affordable.
ROLLOUT_WORLDS = ['cafe']
ROLLOUT_CAPACITY = 8
ROLLOUT_MAX_STEPS = 90
ROLLOUT_NUM_NO_NORM_AGENTS = 1

# Training rollout seeds are drawn from this range so they can never collide
# with the fixed per-norm evaluation seed blocks (all < 100_000).
ROLLOUT_SEED_LOW = 1_000_000
ROLLOUT_SEED_HIGH = 9_000_000

LORA_DEFAULTS = dict(
    r=32,
    alpha=64,
    dropout=0.05,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj'],
)

# ---------------------------------------------------------------------------
# PPO defaults
# ---------------------------------------------------------------------------
#
# Every episode is an independent (norm, world, seed) draw; the baseline is a
# learned value head over the policy's hidden state, and advantages come from
# GAE over the turn sequence of each episode.

PPO_OUTPUT_DIR = os.path.join(CKPT_DIR, 'ppo_4b')

PPO_DEFAULTS = dict(
    # 102702 diagnosis (RUN_LOG 08-22/23): the equal-weight reward put real
    # signal in every batch (success 25-44% at start) but the policy never
    # took a meaningful step — approx_kl ~1e-5/update (~1000x under the
    # usual ~0.01 target), clip_frac 0.00 for 16 straight updates (the
    # +/-0.2 trust region never engaged once), and success DECAYED to
    # ~0.1 as the accumulated micro-steps followed the noise-robust
    # direction (caution/wait). Fix the optimizer, not the reward: bigger
    # steps (lr x5 — clip and the KL anchor are the guardrails, and both
    # are idle) and less advantage noise (2x episodes/update => ~4-10
    # successes per batch instead of 2-5). Healthy range after this:
    # kl 1e-4..1e-2, clip_frac > 0 but < ~0.2, refkl < ~0.3.
    # 104123 postscript: 5e-5 delivered real steps (kl 2-6e-4) and reached
    # the freeze basin in 6 updates — confirming freeze is reward-structural
    # (fixed by curriculum, not lr) — but also produced a format-stability
    # spike at u5 (bad_turns 13 -> 116; the truncation penalty pulled it
    # back to 4 by u6). 3e-5 keeps kl in the healthy band with less tail
    # churn now that the curriculum, not step size, carries the fix.
    episodes_per_update=32,   # independent episodes per update
    num_updates=200,
    lr=3e-5,
    value_lr=1e-4,            # separate (larger) lr for the value head
    clip_eps=0.2,
    value_clip=0.2,           # clipped value loss radius (in scaled-return units); 0 disables
    vf_coef=0.1,              # weight of the value loss against the policy loss
    # KL-to-reference penalty anchoring the policy to the SFT adapter. NOT
    # optional decoration: with 0.0 the clip only bounds per-update movement,
    # nothing bounds cumulative drift, and the first overfit run (99152,
    # 2026-08-20) degenerated in ~6 updates — reasoning collapsed to word
    # salad, actions to "wait", parse rate 1.00 -> 0.50. 0 disables the ref
    # model (only for deliberate experiments).
    kl_beta=0.05,
    gamma=1.0,                # per-env-step discount (reward-to-go bookkeeping)
    gae_lambda=0.95,          # GAE lambda over the turn sequence of an episode
    reward_scale=0.1,         # rewards are multiplied by this before GAE/value fitting
    ppo_epochs=2,             # optimisation passes over the update's turns
    minibatches=4,            # optimizer steps per epoch
    format_bonus=0.5,         # added to the last turn's reward when every turn parsed
    # Training-time reward shaping (rollout.attach_rewards). Applied to the
    # PPO learning signal only — env, raw runs, and logged metrics keep the
    # true benchmark rewards (goal +10, step -0.1, collision -5, violation
    # -20). Rationale (RUN_LOG 08-22): every run through 101225 converged to
    # the freeze policy (return -8, success 0 vs oracle +16.5) because -20
    # violations drown the sparse +10 goals, and 101225 additionally shed the
    # JSON format because unparseable output cost nothing (fallback = wait).
    # 102005 (scale 0.5, coef 0.2) still froze: approach netting +0.1/step vs
    # wait -0.1 never moved the policy off "wait is less risky", the +10 goal
    # was never sampled once, and PPO equilibrated (kl ~1e-5, clip 0).
    # Equal-weight redesign (RUN_LOG 08-22): task success and norm compliance
    # carry the SAME total weight in the training signal. Task is bounded
    # (+10/goal, +20 max) but flat per-violation penalties are not, so any
    # flat value either drowns the task (freeze) or trivializes norms.
    # violation_budget replaces the env's flat -20 with geometric charges
    # (k-th violation: -budget/2^k = -10, -5, -2.5, ...), total bounded by
    # the budget = the full task reward. Marginal cost never hits zero, so
    # there is no free-violations regime. Eval still grades the true -20.
    violation_budget=20.0,        # overrides violation_penalty_scale when > 0
    violation_penalty_scale=1.0,  # legacy flat rescale; inactive under budget
    # 104472 (RUN_LOG 08-23): the violation budget bounded norms, but flat
    # -5 collisions stayed uncapped and became the new drowner — successful
    # episodes averaged 5.6 collisions (-28, one hit -75) by u17, so
    # succeeding scored WORSE than failing (trainret -11.9 vs -5.9) and PPO
    # suppressed success itself (0.56 -> 0.03 in one update). Same geometric
    # fix: k-th collision costs budget/2^k. Budget 10 keeps the first
    # collision at the env's true -5 and bounds the total at the single-goal
    # task reward. Eval still grades the true flat -5.
    collision_budget=10.0,
    progress_shaping_coef=0.5,    # potential phi = -coef * dist-to-target;
                                  # approach nets +coef-0.1/step, wait -0.1
    parse_penalty=0.5,        # charged at every turn that isn't valid JSON
    truncation_penalty=2.0,   # extra charge when the bad turn never closes its
                              # JSON — run 102005's drift mode: a free-text
                              # preamble/repetition loop eats the 1024-token
                              # budget and the object is cut off mid-string
    temperature=0.7,
    top_p=0.95,
    max_new_tokens=1024,
    # Entropy bonus (104472 postmortem, RUN_LOG 08-23): the freeze is a mode
    # collapse onto 'wait' (87% of turns by u20), and nothing in the loss
    # resisted it — clip only bounds per-update movement, and the SFT anchor
    # never engaged (refkl sat at ~0.001 all run). Standard PPO per-token
    # entropy regularizer; small, so it resists collapse without fighting
    # convergence. 0 disables.
    ent_coef=0.01,
    # Freeze WATCH thresholds — log-only, they never halt the run (08-23:
    # the halting version false-fired at 105363 u11, killing the healthiest
    # run of the lineage at its best update, so a halt bought GPU-hours and
    # cost nights).
    # [WATCH] prints when wait share exceeds freeze_wait_frac (RUN_LOG 08-21:
    # >55-60% = freeze winning at capacity 8) or the critic inverts (ev <
    # -0.1 — in 104472 ev went negative at u17, one update before the policy
    # fell over); [FROZEN] prints after freeze_patience consecutive updates
    # where success has actually collapsed. Runs go to num_updates; cancel by
    # hand and resume from adapter_best if a postmortem calls for it.
    freeze_wait_frac=0.6,
    freeze_patience=2,
)
