# ViGaL-style single-step RLOO — the three versions

Training method family: independent single-step snapshots, K samples each,
leave-one-out advantage, OpenRLHF clipped per-token loss, 1 epoch, no KL
(ViGaL, arXiv:2506.08011). The versions differ in **what is rewarded**:

| Version | Folder | Reward | Init |
|---|---|---|---|
| v1 | [`v1_action_rloo/`](v1_action_rloo/README.md) | the rule verifier's verdict on the chosen **action** (1/0 + 0.1 format) | SFT |
| v2 | [`v2_belief_rloo/`](v2_belief_rloo/README.md) | an LLM judge's verdict on the stated **belief** (1 iff rated above the carried-over belief; action not rewarded) | SFT |
| v3 | [`v3_violation_judge/`](v3_violation_judge/README.md) | the env's verdict on a yes/no **violation judgment** of another agent's move (observer task) | base model |
| v4 | [`v4_joint_action_judge/`](v4_joint_action_judge/README.md) | **both** v1 and v3 at once: mixed batches (half action snapshots graded by the rule verifier, half judge records graded by the env verdict), one shared LoRA | SFT |

Each folder is self-contained (own trainer, harvester/records, verifier) and
imports only the version-agnostic modules one level up (`config`, `modeling`,
`policy_client`, `rollout`, `rollout_pool`). Launchers:
`scripts/finetune/run_v{1_action_rloo,2_belief_rloo,3_violation_judge,4_joint_action_judge}.sh`;
weights: `checkpoints/v<N>_<name>/<group>/<RUN_NAME>/`. Other training
method families (e.g. GRPO / PPO / DPO) go in sibling folders of `vigal/`.