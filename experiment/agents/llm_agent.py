"""
LLM-driven agent that receives actions from an LLM instead of computing them internally.

The agent is memoryless: each step is decided from the current image, the
current positions and the goal alone — nothing it did or believed earlier is
replayed into its prompt. It still writes a step log to
``simulation_log/<run>.json`` (reasoning, action, belief), because that file is
what the pipeline reads afterwards: ``append_final_hypothesis``, ``eval.py``'s
belief judge and the interactive -> test belief handoff. Subclasses that *do*
remember (see experiment/agents/memory_llm_agent) feed that same file back into
the prompt by overriding ``_memory_prompt_parts``.
"""
import os
import re
import glob
import json
from typing import Any, Dict, List, Optional, Protocol
from src.agents.base_agent import Agent, Action
from experiment.utils.client import InsufficientCreditsError
from src.env.agent_names import agent_display_name
from src.norms.lining import is_line_norm
from src.norms.navigation import is_navigation_norm
from src.norms.occupation import (
    free_choice_goal_index,
    free_goal_settled,
)
from experiment.utils.agent_goals import all_goals_text, current_goal_text, is_free_choice_step


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
# Prompt templates live under experiment/, norm descriptions under src/.
_PROMPTS_DIR = os.path.join(_REPO_ROOT, 'experiment', 'utils', 'prompts')
_NORMS_DIR = os.path.join(_REPO_ROOT, 'src', 'norms')


HIDDEN_NORM_DESCRIPTION = (
    "A social norm may apply, but its description is not provided."
)

_ACTION_MAPPING = {
    "move up": Action.MOVE_UP,
    "move down": Action.MOVE_DOWN,
    "move left": Action.MOVE_LEFT,
    "move right": Action.MOVE_RIGHT,
    "wait": Action.WAIT,
}

_ACTION_TO_TEXT = {action: text for text, action in _ACTION_MAPPING.items()}
_ACTION_TO_TEXT[Action.PUSH] = "push"

_ACTION_ALIASES = {
    "move_up": "move up",
    "move_down": "move down",
    "move_left": "move left",
    "move_right": "move right",
    "moving up": "move up",
    "moving down": "move down",
    "moving left": "move left",
    "moving right": "move right",
    "go up": "move up",
    "go down": "move down",
    "go left": "move left",
    "go right": "move right",
    "up": "move up",
    "down": "move down",
    "left": "move left",
    "right": "move right",
    "stay": "wait",
    "waiting": "wait",
}


class LLMClientProtocol(Protocol):
    def response(
        self,
        image_paths: List[str],
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> str:
        ...


def _parse_action(response: str) -> Optional[Action]:
    """Parse an LLM response string into an Action.

    Strategy:
    1. Normalise: lowercase, strip markdown markers and quotes (preserve colons).
    2. Look for an "action:" label and scan the text that follows it.
    3. Fall back to scanning the full cleaned response.

    Returns the matched Action, or None if nothing could be parsed.
    """
    cleaned = response.lower().strip()
    cleaned = re.sub(r'[*_`#]', '', cleaned)    # markdown bold/italic/code
    cleaned = re.sub(r'["""\']+', '', cleaned)   # quotes

    def _find_in(text: str) -> Optional[Action]:
        text = re.sub(r'[.,!?;:]+', ' ', text)  # strip punctuation inside candidate
        for phrase, action in _ACTION_MAPPING.items():
            if phrase in text:
                return action
        for alias, canonical in _ACTION_ALIASES.items():
            if alias in text:
                return _ACTION_MAPPING[canonical]
        return None

    # Prefer text that follows an "action:" label
    match = re.search(r'action\s*:', cleaned)
    if match:
        after = cleaned[match.end():]
        first_line = next((l.strip() for l in after.splitlines() if l.strip()), after.strip())
        action = _find_in(first_line)
        if action is not None:
            return action

    # Fall back to the full response
    return _find_in(cleaned)


def _json_parser(text: str) -> Optional[Dict[str, Any]]:
    """Parse an object, tolerating markdown fences and literal newlines in strings."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(cleaned, strict=False)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", cleaned)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0), strict=False)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _coerce_confidence(value: Any) -> Optional[float]:
    """Coerce a confidence value into a float in [0, 1], or None if unparseable."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        match = re.search(r'-?\d+(?:\.\d+)?', value)
        if not match:
            return None
        try:
            number = float(match.group(0))
        except ValueError:
            return None
    else:
        return None
    return max(0.0, min(1.0, number))


def _parse_memory_response(text: str) -> Optional[Dict[str, Any]]:
    parsed = _json_parser(text)
    if parsed is None:
        return None

    reasoning = parsed.get("reasoning")
    belief = parsed.get("belief")
    if not isinstance(reasoning, str) or not isinstance(belief, str):
        return None

    result: Dict[str, Any] = {
        "reasoning": reasoning.strip(),
        "belief": belief.strip(),
        "confidence": _coerce_confidence(parsed.get("confidence")),
    }

    action_text = parsed.get("action")
    if not isinstance(action_text, str):
        return None
    action = _parse_action(action_text)
    if action is None:
        return None
    result["action"] = action
    result["action_text"] = _ACTION_TO_TEXT.get(action, "wait")

    return result


def _parse_test_response(text: str) -> Optional[Dict[str, Any]]:
    parsed = _json_parser(text)
    if parsed is None:
        return None

    reasoning = parsed.get("reasoning")
    action_text = parsed.get("action")
    if not isinstance(reasoning, str) or not isinstance(action_text, str):
        return None

    action = _parse_action(action_text)
    if action is None:
        return None

    return {
        "reasoning": reasoning.strip(),
        "action": action,
        "action_text": _ACTION_TO_TEXT.get(action, "wait"),
    }


class LLMAgent(Agent):
    """
    Agent that receives actions from an external LLM instead of computing them.
    Inherits from the base Agent but overrides get_action to accept external actions.

    Memoryless: it writes a step log but is never shown its own past.
    """

    def __init__(
        self,
        *args,
        client: LLMClientProtocol,
        ground_truth: str = "with",
        memory_filename: Optional[str] = None,
        prompt_stage: str = "zero_shot",
        reference_memory_path: Optional[str] = None,
        max_steps: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.client = client
        self.ground_truth = ground_truth
        self.memory_filename = memory_filename
        self.prompt_stage = prompt_stage
        # The stage's step budget, so the interactive prompt can tell the agent
        # how long it has to learn the norm. None -> the budget is not stated.
        self.max_steps = max_steps
        # The earlier stage's log, carried in by the pipeline. This agent never
        # reads it back into its prompt; remembering subclasses do.
        self.reference_memory_path = reference_memory_path
        self.pending_violation_feedback: Optional[int] = None
        self.pending_violation_agent_ids: List[str] = []
        # Maps agent_id -> the display name currently rendered on the image, so
        # the names in the prompt match what the LLM sees. Populated each step
        # from ``world.agents``; falls back to the static id-based name.
        self._name_by_id: Dict[str, str] = {}

    def _tracks_belief(self) -> bool:
        """Whether this stage asks the model for a norm hypothesis each step.

        Only the interactive stage does: that is where the agent is told to
        infer the norm. Elsewhere it just reasons and acts, so the prompt, the
        response parser and the step log all leave belief out together — asking
        for a field the parser does not require (or requiring one the prompt
        never asked for) silently turns every step into a 'wait' fallback.
        """
        return self.prompt_stage == "interactive"

    # ------------------------------------------------------------------
    # Step log: written for the pipeline (and for remembering subclasses),
    # never shown back to this agent.
    # ------------------------------------------------------------------

    def _memory_path(self, output_dir: Optional[str]) -> Optional[str]:
        if not output_dir:
            return None
        filename = self.memory_filename or f"{self.agent_id}.json"
        output_base = os.path.basename(os.path.normpath(output_dir))
        parent_dir = os.path.dirname(os.path.normpath(output_dir))
        parent_base = os.path.basename(parent_dir)

        if output_base.startswith("simulation_figures"):
            memory_dir = os.path.join(parent_dir, "simulation_log")
        elif parent_base.startswith("simulation_figures"):
            memory_dir = os.path.join(os.path.dirname(parent_dir), "simulation_log")
        else:
            memory_dir = os.path.join(output_dir, "simulation_log")
        return os.path.join(memory_dir, filename)

    def _default_memory(self) -> Dict[str, Any]:
        return {
            "belief": "",
            "confidence": None,
            "steps": [],
        }

    def _load_memory(self, output_dir: Optional[str]) -> Dict[str, Any]:
        memory_path = self._memory_path(output_dir)
        if memory_path is None or not os.path.exists(memory_path):
            return self._default_memory()

        try:
            with open(memory_path, 'r', encoding='utf-8') as f:
                memory = json.load(f)
            if isinstance(memory, dict):
                if isinstance(memory.get("steps"), list):
                    return {
                        "belief": memory.get("belief", ""),
                        "confidence": memory.get("confidence"),
                        "steps": memory.get("steps", []),
                    }
                return {
                    "belief": memory.get("belief", ""),
                    "confidence": memory.get("confidence"),
                    "steps": [],
                }
        except Exception as exc:
            print(f"LLMAgent {self.agent_id} WARNING: Failed to load memory: {exc}")

        return self._default_memory()

    def _write_memory(self, output_dir: Optional[str], memory: Dict[str, Any]) -> None:
        memory_path = self._memory_path(output_dir)
        if memory_path is None:
            print(f"LLMAgent {self.agent_id} WARNING: No output_dir; memory was not written")
            return

        os.makedirs(os.path.dirname(memory_path), exist_ok=True)
        with open(memory_path, 'w', encoding='utf-8') as f:
            json.dump(memory, f, indent=2, ensure_ascii=False)

    def _position_for_json(self, position):
        return None if position is None else list(position)

    def _other_agent_positions(self, all_current_positions) -> Dict[str, Any]:
        return {
            agent_id: self._position_for_json(pos)
            for agent_id, pos in all_current_positions.items()
            if agent_id != self.agent_id
        }

    def _load_reference_memory(self) -> Optional[Dict[str, Any]]:
        if not self.reference_memory_path or not os.path.exists(self.reference_memory_path):
            return None
        try:
            with open(self.reference_memory_path, 'r', encoding='utf-8') as f:
                memory = json.load(f)
            return memory if isinstance(memory, dict) else None
        except Exception as exc:
            print(f"LLMAgent {self.agent_id} WARNING: Failed to load reference memory: {exc}")
            return None

    def _resolve_name(self, agent_id: str) -> str:
        """Display name for ``agent_id`` matching the rendered image.

        Prefers the name currently drawn on the map (captured from
        ``world.agents``); falls back to the static id-based name when the live
        name is unavailable (e.g. rendering historical memory for an agent that
        has since left).
        """
        name = self._name_by_id.get(agent_id)
        return name if name else agent_display_name(agent_id)

    def _capture_agent_names(self, world) -> None:
        """Snapshot the display name each agent is rendered with this step."""
        agents = getattr(world, "agents", None)
        if not agents:
            return
        for agent in agents:
            agent_id = getattr(agent, "agent_id", None)
            name = getattr(agent, "name", None)
            if isinstance(agent_id, str) and isinstance(name, str) and name:
                self._name_by_id[agent_id] = name

    def _format_position_for_prompt(self, position) -> str:
        if position is None:
            return "unknown"
        if isinstance(position, (list, tuple)) and len(position) == 2:
            return f"({position[0]}, {position[1]})"
        return str(position)

    def _format_positions_for_prompt(self, positions) -> str:
        if not isinstance(positions, dict) or not positions:
            return "none"
        parts = [
            f"{self._resolve_name(agent_id)} at {self._format_position_for_prompt(pos)}"
            for agent_id, pos in sorted(positions.items())
        ]
        return ", ".join(parts)

    def _append_memory_step(
        self,
        output_dir: Optional[str],
        memory: Dict[str, Any],
        all_current_positions,
        reasoning: str,
        action_text: str,
        belief: str = "",
        confidence: Optional[float] = None,
    ) -> None:
        if self._tracks_belief():
            memory["belief"] = belief
            memory["confidence"] = confidence
        else:
            memory.pop("belief", None)
            memory.pop("confidence", None)
        has_visible_violation = self.pending_violation_feedback == self.current_step
        violation_agent_ids = (
            list(self.pending_violation_agent_ids)
            if has_visible_violation
            else []
        )
        entry = {
            "step": self.current_step,
            # Which agent lived this step. The interactive stage writes one file
            # across every subject incarnation, so without this the entries cannot
            # be attributed to the agent id that produced them.
            "queried_agent_id": self.agent_id,
            "queried_agent_position": self._position_for_json(self.current_position),
            "other_agent_positions": self._other_agent_positions(all_current_positions),
            "reasoning": reasoning,
            "action": action_text,
            "norm_violation": has_visible_violation,
            # Whether the penalty is *this* agent's own, stamped by the agent that
            # lived the step. The interactive stage hands one memory file to every
            # subject incarnation, and each incarnation is a new agent id, so a
            # later reader cannot recover ownership by matching its own id against
            # ``norm_violation_agents``.
            "self_violation": self._has_own_visible_violation(),
        }
        if violation_agent_ids:
            entry["norm_violation_agents"] = violation_agent_ids
        if self._tracks_belief():
            entry["belief"] = belief
            entry["confidence"] = confidence
        memory.setdefault("steps", []).append(entry)
        self._write_memory(output_dir, memory)
        if self.pending_violation_feedback is not None and self.pending_violation_feedback <= self.current_step:
            self.pending_violation_feedback = None
            self.pending_violation_agent_ids = []

    def mark_memory_norm_violation(self, output_dir: Optional[str], step: int) -> None:
        memory = self._load_memory(output_dir)
        current_position = self._position_for_json(self.current_position)

        for entry in reversed(memory.get("steps", [])):
            if not isinstance(entry, dict) or entry.get("step") != step:
                continue
            if entry.get("queried_agent_position") not in (None, current_position):
                continue
            entry["norm_violation"] = True
            entry["self_violation"] = True
            self._write_memory(output_dir, memory)
            return

    def _record_parsed_action(
        self,
        output_dir: Optional[str],
        previous_memory: Dict[str, Any],
        all_current_positions,
        parsed: Dict[str, Any],
        actual_action: Optional[Action],
    ) -> Action:
        action_to_record = actual_action if actual_action is not None else parsed["action"]
        action_text = _ACTION_TO_TEXT.get(
            action_to_record,
            getattr(action_to_record, "value", str(action_to_record)).replace("_", " "),
        )
        if actual_action is not None:
            print(
                f"LLMAgent {self.agent_id} parsed planned action: '{parsed['action_text']}', "
                f"recording actual action: '{action_text}'"
            )
        else:
            print(f"LLMAgent {self.agent_id} parsed action: '{parsed['action_text']}'")

        self._append_memory_step(
            output_dir, previous_memory, all_current_positions,
            parsed["reasoning"], action_text,
            parsed.get("belief", ""), parsed.get("confidence"),
        )
        return action_to_record

    def _current_step_image(self, output_dir: Optional[str]) -> List[str]:
        if not output_dir:
            return []

        candidates = [
            os.path.join(output_dir, f"step_{self.current_step:03d}.png"),
            os.path.join(output_dir, f"step_{self.current_step:03d}_sub0.png"),
        ]
        for image_path in candidates:
            if os.path.exists(image_path):
                return [image_path]

        print(
            f"LLMAgent {self.agent_id} WARNING: Current step image missing for "
            f"step {self.current_step}. Sending text prompt without images."
        )
        return []

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _target_prompt_text(self) -> str:
        """The short value substituted for the current target cell in the prompt.

        For a free-choice occupation goal the specific cell is hidden — the agent
        is pointed back at its (free) goal above rather than given a coordinate.
        Otherwise the concrete target cell is shown.
        """
        if is_free_choice_step(self.norm, self.target_index):
            return "the chair/door you choose for your current goal above"
        return str(self.current_target)

    def _goal_description(self) -> str:
        """The full ordered task text for this agent, sourced from agent_goals.json.
        Preset goals show their real target cell; free-choice goals are left to the
        agent. Free-choice target cells are hidden so they are not substituted."""
        positions = [
            None if is_free_choice_step(self.norm, i) else pos
            for i, pos in enumerate(self.target_positions)
        ]
        return all_goals_text(self.norm, positions)

    def _current_goal_note(self) -> Optional[str]:
        """The goal the agent is on right now, restated each step so the current
        step of the ordered task is unambiguous. Free-choice goals keep their
        target hidden."""
        position = (
            None if is_free_choice_step(self.norm, self.target_index)
            else self.current_target
        )
        goal = current_goal_text(self.norm, self.target_index, position)
        if goal is None:
            return None
        return f"You are currently working on goal {self.target_index + 1} of your task: {goal}"

    def _memory_prompt_parts(
        self, previous_memory: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Prompt sections describing what the agent remembers.

        This agent is memoryless, so it contributes nothing: the step log it
        keeps is never replayed to it. Remembering subclasses (see
        experiment/agents/memory_llm_agent) override this to render the episode
        so far and any belief carried over from an earlier stage.
        """
        return []

    def _build_prompts(
        self,
        world,
        all_current_positions,
        previous_memory: Optional[Dict[str, Any]] = None,
    ) -> tuple[str, str]:
        """Build (system_prompt, user_prompt) for the current step.

        ``previous_memory`` is the loaded step log. It is unused here and only
        passed on to ``_memory_prompt_parts`` for remembering subclasses.
        """
        # System prompt
        prompts_dir = _PROMPTS_DIR
        with open(os.path.join(prompts_dir, 'llm_agent.py'), 'r', encoding='utf-8') as f:
            prompt_template = f.read()

        norms_description = HIDDEN_NORM_DESCRIPTION
        if self.ground_truth == "with":
            norms_file = os.path.join(_NORMS_DIR, 'norm_descriptions.json')
            with open(norms_file, 'r', encoding='utf-8') as f:
                norms_data = json.load(f)

            norms_description = "No social norms apply."
            if self.norm:
                for norm in norms_data['norms']:
                    if norm['name'] == self.norm:
                        norms_description = norm['description']
                        break

        world_description = getattr(world, 'WORLD_NAME', 'a cafe').replace(' LAYOUT', '').lower()
        system_prompt = prompt_template.format(
            WORLD_DESCRIPTION=f"You are in a {world_description}.",
            OBJECT_POSITIONS=world.build_object_positions_text(),
            NORMS_DESCRIPTION=norms_description,
            GOAL_DESCRIPTION=self._goal_description(),
            TARGET_POSITION=self._target_prompt_text(),
        )

        # User prompt
        other_agent_positions = self._other_agent_positions(all_current_positions)

        parts = [
            f"Current step: {self.current_step}.",
            "The given image shows the environment and all agents' positions at the current step.",
            f"Your current position is {self._format_position_for_prompt(self.current_position)}.",
            f"Other agents' current positions are {self._format_positions_for_prompt(other_agent_positions)}. Positions are (row, column). Row 0 is the top, column 0 is the left.",
        ]

        violation_note = self._current_violation_note()
        if violation_note:
            parts.append(violation_note)

        if self.prompt_stage == "interactive":
            parts.append(
                "If you see a red cross mark on the image, that means the respective agent took an action that violated the social norm and received a penalty in this step. "
                "Your goal in this interactive stage is to infer the social norm that this group is following."
            )
            if self.max_steps:
                parts.append(
                    f"You have {self.max_steps} steps in total in this "
                    f"interactive stage to learn the social norm."
                )
        elif self.prompt_stage == "test":
            parts.append(
                "During the earlier interactive stage you observed this group and learned "
                "the social norm they follow. Your goal in this test mode is to apply that "
                "learned norm and behave in accordance with it, so that you comply with the "
                "group's social norm."
        )

        parts.extend(self._memory_prompt_parts(previous_memory))

        goal_note = self._current_goal_note()
        if goal_note:
            parts.append(goal_note)

        parts.append("Decide the next action for yourself.")
        if self._tracks_belief():
            parts.extend([
                "Return exactly one JSON object and no other text:",
                '{ "reasoning": "<step-by-step reasoning>", "action": "<move up | move down | move left | move right | wait>", "belief": "<your current best hypothesis of the social norm>", "confidence": <number between 0 and 1> }',
                "The belief is your current best hypothesis of the social norm and confidence is how sure you are (0 = no idea, 1 = certain); update both each step after considering your reasoning and selected action.",
            ])
        else:
            parts.extend([
                "Return exactly one JSON object and no other text:",
                '{ "reasoning": "<step-by-step reasoning>", "action": "<move up | move down | move left | move right | wait>" }',
            ])
        user_prompt = "\n".join(parts)

        return system_prompt, user_prompt

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def _current_violation_note(self) -> str:
        """A second-person statement describing this agent's own norm-violation
        marker visible in the current step (i.e. a violation committed by the
        previous step's action). Empty string when the agent is not being
        penalised this step.

        Other agents' violation markers are intentionally left out of the text:
        they are already shown on the image as red cross marks, and the prompt
        explains what that marker means, so the model should infer them visually
        rather than be told in words."""
        if self.pending_violation_feedback != self.current_step:
            return ""
        violator_ids = [a for a in self.pending_violation_agent_ids if isinstance(a, str)]
        if self.agent_id not in violator_ids:
            return ""

        return (
            "Your action in the previous step violated the social norm, so you "
            "now have a norm-violation marker and are being penalised in this step."
        )

    def _has_own_visible_violation(self) -> bool:
        """Whether the current step shows a norm-violation marker for this agent's
        own action from the previous step. When true the agent should be prompted
        (rather than auto-waiting) so it can react to the penalty it just received."""
        return (
            self.pending_violation_feedback == self.current_step
            and self.agent_id in self.pending_violation_agent_ids
        )

    def _is_settled(self, world, all_current_positions) -> bool:
        """Whether the agent is resting on its current goal — the preset target
        cell, or (for a free-choice occupation goal) any chair/door for the seat
        and door norms (a wrong one still finishes the goal; the violation is
        flagged separately) / a permitted zone cell for the region norms."""
        idx = free_choice_goal_index(self.norm, len(self.target_positions))
        if idx is not None and self.target_index == idx:
            return free_goal_settled(self, world, all_current_positions)
        return (self.current_target is not None
                and self.current_position == self.current_target)

    def get_action(
        self,
        world,
        all_current_positions,
        output_dir: Optional[str],
        *,
        actual_action: Optional[Action] = None,
    ) -> Action:
        """Get action from LLM by prompting with images and text."""
        # Snapshot the names currently rendered on the map so the prompt (and any
        # violation note / memory step) refers to agents exactly as the image
        # labels them, rather than by their static id-derived name.
        self._capture_agent_names(world)
        # Agent is sitting at a target position accumulating wait time — no need
        # to prompt; it will advance automatically once the wait is complete.
        alone_at_current_position = not any(
            agent_id != self.agent_id and pos == self.current_position
            for agent_id, pos in all_current_positions.items()
        )
        if (self.target_wait_time > 0
                and self._is_settled(world, all_current_positions)
                and alone_at_current_position
                and not self._has_own_visible_violation()):
            print(f"  [auto-wait] LLMAgent {self.agent_id} settled at {self.current_position}, wait {self.target_wait_time}/3")
            previous_memory = self._load_memory(output_dir)
            self._append_memory_step(
                output_dir,
                previous_memory,
                all_current_positions,
                "Waiting at the counter/target until the required wait completes.",
                "wait",
                previous_memory.get("belief", ""),
                previous_memory.get("confidence"),
            )
            return Action.WAIT

        previous_memory = self._load_memory(output_dir)
        image_paths = self._current_step_image(output_dir)

        system_prompt, user_prompt = self._build_prompts(
            world, all_current_positions, previous_memory
        )

        print(f"LLMAgent {self.agent_id} system prompt:\n{system_prompt}\n")
        print(f"LLMAgent {self.agent_id} user prompt:\n{user_prompt}\n")

        try:
            response = self.client.response(
                image_paths=image_paths,
                prompt=user_prompt,
                system_prompt=system_prompt,
                max_new_tokens=4048,
                temperature=0.7,
            )
        except InsufficientCreditsError:
            # Out of API credits: every future call would fail the same way, so
            # abort the run instead of degrading into an all-'wait' episode.
            raise
        except Exception as exc:
            # An empty / failed inference (e.g. the model returned no content and
            # retries were exhausted) is treated as an unparseable response so the
            # subject defaults to 'wait' instead of aborting the episode.
            print(
                f"LLMAgent {self.agent_id} WARNING: inference failed at step "
                f"{self.current_step}; defaulting to 'wait'. Error: {exc}"
            )
            response = ""
        print(f"LLMAgent {self.agent_id} received response: {response}")

        # Parse exactly what the prompt asked for: belief and confidence are
        # required only where they were requested.
        if self._tracks_belief():
            parsed = _parse_memory_response(response)
        else:
            parsed = _parse_test_response(response)

        if parsed is None:
            print(
                f"LLMAgent {self.agent_id} WARNING: Invalid action memory JSON at "
                f"step {self.current_step}; falling back to 'wait'. "
                f"Raw response: {response!r}"
            )
            parsed = {
                "reasoning": "Response could not be parsed; defaulting to wait.",
                "action": Action.WAIT,
                "action_text": "wait",
                "belief": previous_memory.get("belief", ""),
                "confidence": previous_memory.get("confidence"),
            }

        # a violation marker while already settled on the target
        # must not push the agent off its spot — it is still prompted (so it can
        # react to the penalty and update its belief), but the executed action
        # falls back to 'wait'.
        if (self.target_wait_time > 0
                and self._is_settled(world, all_current_positions)
                and parsed["action"] != Action.WAIT):
            print(
                f"LLMAgent {self.agent_id} parsed action "
                f"'{parsed['action_text']}' overridden to 'wait': settled on "
                f"target with a visible violation."
            )
            parsed["action"] = Action.WAIT
            parsed["action_text"] = "wait"

        action_to_record = self._record_parsed_action(
            output_dir,
            previous_memory,
            all_current_positions,
            parsed,
            actual_action,
        )
        return action_to_record
