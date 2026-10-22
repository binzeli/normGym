import random
from typing import Tuple, Optional, List, Dict

from src.env.actions import Action
from src.norms.lining import generate_queue_line, get_queue, is_line_norm
from src.norms.navigation import is_navigation_norm, is_movement_norm, is_spacing_norm, is_xcell_norm, xcell_requires_visit
from src.norms.norm_checker import check_norm_violation
from src.norms.occupation import (
    agent_may_enter_region,
    is_occupation_door_norm,
    is_occupation_region_norm,
    is_occupation_seat_norm,
    pick_exit_door,
    pick_seat,
    seat_available,
    waiting_zone_cells,
)

from src.agents.base_agent import (
    Agent,
    AgentType,
    AGENT_COLORS,
    agent_class_index,
)
from src.agents.planner import (
    predictive_planner,
)


class OracleAgent(Agent):
    """Oracle agent that plans its own path to a sequence of targets.

    Moves to its current target (handling queue and x-cell norms along the
    way), waits, then advances to the next target until all are exhausted.
    """

    is_oracle = True

    def get_action(
        self, world, all_current_positions, agent_classes, collision_actions=None,
        violation_log: Optional[Dict[str, List[str]]] = None,
        all_target_positions: Optional[Dict[str, Tuple[int, int]]] = None,
        all_target_indices: Optional[Dict[str, int]] = None,
    ) -> Optional[Action]:
        """
        Choose action based on current position and goal.

        Process flow:
        1. Done agents return None immediately.
        2. Check ordering condition — if at counter, return ORDER action.
        3. Otherwise plan a path and return a movement action.
        """
        if self.done:
            return None

        # Correct the randomly-assigned occupation goals so they comply with the
        # active norm before planning. Exactly one goal is the agent's to choose,
        # and which one depends on the norm family (occupation.free_choice_goal_index):
        #   - door norms: the exit door (last goal) is chosen — the door nearest
        #     the agent's seat among those the norm permits — then latched for
        #     later steps. The seat it was assigned stands.
        #   - seat norms: the seat (first goal) is chosen — a norm-compliant
        #     chair, re-picked while the agent is still on its way. The exit door
        #     it was assigned stands.
        #   - region norms: free_choice_goal_index calls the waiting-zone cell a
        #     free choice, but that is how an *interactive* agent's zone cell is
        #     graded (any cell it may occupy counts). An oracle keeps the cell it
        #     was assigned and instead times its entry to the capacity /
        #     single-class rule while planning, so nothing is re-picked here.
        #   - every other norm: both goals are assigned; neither is re-picked.
        # A norm-blind agent (norm=None) cannot read the norm family off itself,
        # so the pipeline flags which goal is its choice (free_choice_door /
        # free_choice_seat). pick_exit_door and pick_seat then run with norm=None,
        # leaving every door / every chair qualifying, and it takes the nearest —
        # the shortest path, compliant or not.
        door_norm_run = ((self.norm and is_occupation_door_norm(self.norm))
                         or getattr(self, 'free_choice_door', False))
        seat_norm_run = ((self.norm and is_occupation_seat_norm(self.norm))
                         or getattr(self, 'free_choice_seat', False))
        if (door_norm_run and len(self.target_positions) >= 2
                and not getattr(self, '_occupation_door_fixed', False)):
            door = pick_exit_door(self.norm, self.agent_class, world,
                                  from_pos=tuple(self.target_positions[0]))
            if door is not None:
                self.target_positions[-1] = door
            self._occupation_door_fixed = True

        # Seat norms only: the seat carries no preset assignment, so while it is
        # still the active goal the agent re-picks it every step — the nearest
        # allowed seat not actually occupied right now. Agents know nothing about
        # each other's intentions: a seat only reads as taken once someone is
        # sitting on it, so several agents may race for the same chair and the
        # losers divert to the next nearest the moment the winner sits down.
        # (Under occupation_seat_in_order the pick is the fill frontier instead.)
        # A norm-blind agent flagged free_choice_seat does the same over ALL
        # chairs — pick_seat with norm=None uses the full pool — so it sits at
        # the nearest free seat, compliant or not. Once the agent has actually
        # sat down on its seat goal (current_position == target_positions[0]) it
        # freezes — it must not give up the seat it just took.
        # Under every other norm, door norms included, the seat is an assigned
        # goal and stands as given. A pinned test-stage subject still gets the
        # region-restricted divert below if its chair proves unreachable.
        seat_goal_active = (
            self.target_index == 0 and bool(self.target_positions)
            and seat_norm_run
        )
        if seat_goal_active:
            already_seated = self.current_position == self.target_positions[0]
            if not already_seated:
                seat = pick_seat(
                    self.norm, self.agent_class, all_current_positions, world,
                    self.agent_id, agent_pos=self.current_position,
                )
                if seat is not None:
                    self.target_positions[0] = seat

        # A subject whose seat was pinned to a region for the test stage (the
        # door norms and xcell_must_visit — see occupation.subject_test_seat)
        # keeps that seat while it
        # can still be had, and moves to the next chair of the same region once
        # it cannot: someone is sitting on it, or every way in is blocked.
        # Without this the agent holds out for one specific chair it will never
        # reach and burns the episode waiting beside it.
        seat_pool = getattr(self, 'seat_region_pool', None)
        if seat_pool and self.target_index == 0 and self.target_positions:
            seat = tuple(self.target_positions[0])
            if self.current_position != seat:
                occupied = {pos for aid, pos in all_current_positions.items()
                            if aid != self.agent_id and pos is not None}
                if not seat_available(world, seat, occupied):
                    nxt = next((s for s in seat_pool
                                if seat_available(world, s, occupied)), None)
                    if nxt is not None:
                        self.target_positions[0] = nxt

        # Case 2: Plan path and choose movement action


        current_target = self.current_target
        if current_target is None or self.current_position == current_target:
            return Action.WAIT

        # Region norms: queue at the edge of a full zone instead of rerouting.
        # compute_a_star_path checks the norm on the immediate step only, so with
        # the zone full it plans a path that enters two steps out, finds that
        # entry barred on the next replan, and reroutes — the agent ping-pongs
        # outside the zone, burns the episode, and collides with the others doing
        # the same. Waiting one step from the zone is both what a compliant agent
        # does and the only way it is still standing there when a slot frees.
        # Applied only at the edge: further out the agent still walks in.
        if (self.norm and is_occupation_region_norm(self.norm)
                and self.target_index == 0):
            zone = waiting_zone_cells(world)
            at_edge = (self.current_position not in zone
                       and any(abs(self.current_position[0] - r)
                               + abs(self.current_position[1] - c) == 1
                               for r, c in zone))
            if at_edge and not agent_may_enter_region(
                    self.norm, self.agent_id, agent_classes,
                    all_current_positions, world):
                print(f"Agent {self.agent_id}: waiting zone full, holding at "
                      f"{self.current_position} until a slot frees.")
                return Action.WAIT

        if self.norm and is_xcell_norm(self.norm):
            x_cells = world.zones.get('affordance_x', [])
            nav_target = current_target
            if xcell_requires_visit(self.norm) and x_cells:
                if not self.visited_x_cells:
                    nav_target = min(
                        x_cells,
                        key=lambda c: abs(c[0] - self.current_position[0]) + abs(c[1] - self.current_position[1])
                    )

            action = predictive_planner(
                self.agent_id, self.current_position,
                self.norm,
                nav_target, world,
                all_current_positions, collision_actions,
                violation_log=violation_log,
                all_target_positions=all_target_positions,
                all_target_indices=all_target_indices,
            )


        elif self.norm and is_line_norm(self.norm) and self.target_index == 0:
            queue_cells = generate_queue_line(current_target, world, self.norm)
            eligible_agents = {a.agent_id for a in world.agents if a.target_index == 0}
            current_queue = get_queue(
                queue_cells, all_current_positions,
                previous_queue=world.previous_queue,
                eligible_agents=eligible_agents,
            )
            world.current_queue = current_queue

            current_index = next(
                (idx for idx, aid in current_queue.items() if aid == self.agent_id), None
            )

            print(f"Agent {self.agent_id} - Current queue: {current_queue}")
            print(f"Agent {self.agent_id} - Queue index: {current_index}")

            queue_target = current_target
            if (self.current_position != queue_cells[1] and self.agent_id not in world.current_queue.values()):
                next_index = max(current_queue.keys()) + 1 if len(current_queue) > 1 else 1
                if len(current_queue) > 1 and next_index < len(queue_cells):
                    queue_target = queue_cells[next_index]
                    print(
                        f"Agent {self.agent_id} not in queue, targeting position after"
                        f" last agent: {queue_target}"
                    )
                else:
                    queue_target = queue_cells[1]
                    print(
                        f"Agent {self.agent_id} not in queue, using sub-target:"
                        f" {queue_target} (queue_cells[1])"
                    )
            else:
                print(
                    f"Agent {self.agent_id} in queue at index {current_index},"
                    f" targeting: {queue_target}"
                )

            action = predictive_planner(
                self.agent_id, self.current_position,
                self.norm,
                queue_target, world,
                all_current_positions, collision_actions,
                violation_log=violation_log,
                all_target_positions=all_target_positions,
                all_target_indices=all_target_indices,
            )

        else:
            action = predictive_planner(
                self.agent_id, self.current_position,
                self.norm,
                current_target, world,
                all_current_positions, collision_actions,
                violation_log=violation_log,
                all_target_positions=all_target_positions,
                all_target_indices=all_target_indices,
            )

        print("target position:", current_target)
        print("action:", action)

        # Spacing and movement norms share the livelock: WAIT never violates
        # either, so pausing is always a compliant way to desynchronise.
        if self.norm and (is_spacing_norm(self.norm) or is_movement_norm(self.norm)):
            action = self._break_spacing_oscillation(action, world, all_current_positions)

        return action if action else Action.WAIT

    _MOVE_ACTIONS = (Action.MOVE_UP, Action.MOVE_DOWN, Action.MOVE_LEFT, Action.MOVE_RIGHT)
    # Oscillation events on the same goal before pausing escalates to retreating.
    _OSC_EVENTS_BEFORE_RETREAT = 3

    def _break_spacing_oscillation(self, action, world, all_current_positions) -> Optional[Action]:
        """
        Livelock breaker for spacing norms.

        Agents replan every step against predictions of each other, so agents
        blocking each other's adjacency zones can fall into a perfectly
        synchronized ping-pong (A->B->A->B forever) and never pass. When this
        agent is about to undo its own previous move for the second step in a
        row, pause for a random 1-3 steps instead: the random pause
        desynchronizes the dancers, and WAIT never violates a spacing norm.

        Pausing alone doesn't help when several agents are jammed around a
        chokepoint (pausing in place still blocks it), so after
        _OSC_EVENTS_BEFORE_RETREAT oscillation events without net progress
        toward the current goal, the agent retreats — takes a compliant step
        away from the nearest other agent — to open the jam up.
        """
        # Net progress toward the current goal resets the escalation.
        target = self.current_target
        if target is not None:
            h = abs(self.current_position[0] - target[0]) + abs(self.current_position[1] - target[1])
            key = (self.target_index, target)
            if (getattr(self, '_spacing_progress_key', None) != key
                    or h < getattr(self, '_spacing_best_h', float('inf'))):
                self._spacing_progress_key = key
                self._spacing_best_h = h
                self._spacing_osc_events = 0

        # Currently serving a sustained retreat: keep walking away from the
        # jam for the committed number of steps (a single retreat step isn't
        # enough — the next replan walks straight back into the jam).
        if getattr(self, '_spacing_retreat_steps', 0) > 0:
            self._spacing_retreat_steps -= 1
            retreat = self._spacing_retreat_action(world, all_current_positions)
            if retreat is not None:
                return retreat
            return Action.WAIT

        # Currently serving a pause.
        if getattr(self, '_spacing_pause_steps', 0) > 0:
            self._spacing_pause_steps -= 1
            return Action.WAIT

        if action not in self._MOVE_ACTIONS:
            return action

        next_pos = world.transition(self.current_position, action)
        # A shortest path never immediately backtracks, so returning to the
        # cell the agent's last move came from means the plan flip-flopped
        # between replans. Compared against the last chosen *move*, not the
        # last step: under the rhythm movement norms forced waits sit between
        # the moves of a bounce (up, down, wait, up, down, ...) and a
        # step-to-step comparison would never see it.
        last_move = getattr(self, '_last_planned_move', None)  # (from, to)
        self._last_planned_move = (self.current_position, next_pos)
        if (last_move is not None and last_move[1] == self.current_position
                and next_pos == last_move[0]):
            self._spacing_bounce_count = getattr(self, '_spacing_bounce_count', 0) + 1
        else:
            self._spacing_bounce_count = 0
            return action

        if self._spacing_bounce_count >= 2:
            self._spacing_bounce_count = 0
            self._spacing_osc_events = getattr(self, '_spacing_osc_events', 0) + 1
            if self._spacing_osc_events >= self._OSC_EVENTS_BEFORE_RETREAT:
                retreat = self._spacing_retreat_action(world, all_current_positions)
                if retreat is not None:
                    self._spacing_osc_events = 0
                    self._spacing_retreat_steps = random.randint(1, 3)
                    print(f"Agent {self.agent_id}: persistent jam, retreating "
                          f"{self._spacing_retreat_steps + 1} step(s) ({retreat}).")
                    return retreat
            self._spacing_pause_steps = random.randint(0, 2)
            print(f"Agent {self.agent_id}: oscillation detected, pausing "
                  f"{self._spacing_pause_steps + 1} step(s).")
            return Action.WAIT
        return action

    def _spacing_retreat_action(self, world, all_current_positions) -> Optional[Action]:
        """Compliant step that maximizes distance to the nearest other agent.
        Random tie-break so jammed agents don't mirror each other's retreats."""
        others = [p for aid, p in all_current_positions.items()
                  if aid != self.agent_id and p is not None]
        candidates = []
        for act in self._MOVE_ACTIONS:
            nxt = world.transition(self.current_position, act)
            if nxt is None or nxt == self.current_position:
                continue
            if world.is_sittable(nxt) and nxt != self.current_target:
                continue
            if nxt in others:
                continue
            if check_norm_violation(self.norm, self.current_position, nxt,
                                    self.current_target, all_current_positions,
                                    world, self.agent_id):
                continue
            candidates.append((act, nxt))
        if not candidates:
            return None

        def clearance(pos):
            return min((abs(pos[0] - p[0]) + abs(pos[1] - p[1]) for p in others),
                       default=99)

        best = max(clearance(nxt) for _, nxt in candidates)
        return random.choice([act for act, nxt in candidates if clearance(nxt) == best])
