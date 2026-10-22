"""Flat (top-down) renderer for normGym grids.

Produces PNG frames (or interactive windows) using matplotlib. Mirrors the
interface of :class:`~src.env.iso_renderer.IsometricRenderer` so the gym env
can delegate to either renderer behind a single `_visualize_step` call.
"""

from typing import Dict, List, Tuple, Optional, Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from src.env.actions import Action
from src.world.base_world import CellType


class FlatRenderer:
    """Top-down matplotlib renderer for a single simulation frame."""

    def __init__(self, sprite_cache, world):
        self.sprite_cache = sprite_cache
        self.world = world
        # Render-quality knobs. Defaults match the figures used as LLM eval input;
        # the web viewer lowers these for faster frames (less delay after input).
        self.save_dpi = 100
        self.tight_bbox = True

    def _is_queried_agent(self, agent_id: str, queried_agent_label: Optional[str]) -> bool:
        return queried_agent_label is not None and agent_id == queried_agent_label

    # ----------------------------- Helpers --------------------------------

    def _build_wood_floor_canvas(self, px_per_cell: int = 36) -> np.ndarray:
        """Floor background from tiled sprite texture."""
        return self.sprite_cache.build_floor_canvas(self.world.width, self.world.height)

    def _draw_cell_icon(self, ax, c: float, r: float, icon: str, z: int = 3) -> None:
        """Draw simple semantic icons for table/chair/counter."""
        if icon == "table":
            # tabletop
            ax.add_patch(patches.Rectangle(
                (c - 0.18, r - 0.12), 0.36, 0.16,
                facecolor="white", edgecolor="black", linewidth=1.0, zorder=z
            ))
            # legs
            ax.plot([c - 0.12, c - 0.12], [r + 0.04, r + 0.20], color="black", linewidth=1.2, zorder=z)
            ax.plot([c + 0.12, c + 0.12], [r + 0.04, r + 0.20], color="black", linewidth=1.2, zorder=z)

        elif icon == "chair":
            # backrest
            ax.add_patch(patches.Rectangle(
                (c - 0.16, r - 0.18), 0.32, 0.08,
                facecolor="white", edgecolor="black", linewidth=1.0, zorder=z
            ))
            # seat
            ax.add_patch(patches.Rectangle(
                (c - 0.16, r - 0.02), 0.32, 0.10,
                facecolor="white", edgecolor="black", linewidth=1.0, zorder=z
            ))
            # front legs
            ax.plot([c - 0.10, c - 0.10], [r + 0.08, r + 0.20], color="black", linewidth=1.2, zorder=z)
            ax.plot([c + 0.10, c + 0.10], [r + 0.08, r + 0.20], color="black", linewidth=1.2, zorder=z)

        elif icon == "counter":
            # top slab
            ax.add_patch(patches.Rectangle(
                (c - 0.24, r - 0.16), 0.48, 0.10,
                facecolor="#fff3df", edgecolor="black", linewidth=1.0, zorder=z
            ))
            # front face
            ax.add_patch(patches.Rectangle(
                (c - 0.20, r - 0.04), 0.40, 0.24,
                facecolor="#ffe5c4", edgecolor="black", linewidth=1.0, zorder=z
            ))
            # register
            ax.add_patch(patches.Rectangle(
                (c + 0.06, r - 0.22), 0.10, 0.08,
                facecolor="black", edgecolor="black", linewidth=0.8, zorder=z
            ))
        elif icon == "shelf":
            # wall shelf board
            ax.add_patch(patches.Rectangle(
                (c - 0.22, r - 0.12), 0.44, 0.08,
                facecolor="#fff6e3", edgecolor="black", linewidth=1.0, zorder=z
            ))
            # support brackets
            ax.plot([c - 0.14, c - 0.18], [r - 0.04, r + 0.12], color="black", linewidth=1.0, zorder=z)
            ax.plot([c + 0.14, c + 0.18], [r - 0.04, r + 0.12], color="black", linewidth=1.0, zorder=z)
            # small cup/box on shelf
            ax.add_patch(patches.Rectangle(
                (c - 0.05, r - 0.20), 0.10, 0.08,
                facecolor="#ffffff", edgecolor="black", linewidth=0.9, zorder=z
            ))

    def _get_display_entrance_cells(self) -> List[Tuple[int, int]]:
        """Return entrance cells used for rendering."""
        return list(self.world.entrances)

    def _draw_display_entrances(self, ax) -> None:
        """Overlay entrance visuals without changing the underlying world grid."""
        display_cells = self._get_display_entrance_cells()
        entrance_sprite = self.sprite_cache.get_cell_sprite(CellType.ENTRANCE.value)

        for r, c in display_cells:
            if entrance_sprite is not None:
                ax.imshow(
                    entrance_sprite,
                    origin='upper',
                    extent=(c - 0.5, c + 0.5, r + 0.5, r - 0.5),
                    interpolation='nearest',
                    zorder=2
                )
            else:
                ax.add_patch(
                    patches.Rectangle(
                        (c - 0.5, r - 0.5), 1, 1,
                        facecolor="#63b96a",
                        edgecolor="none",
                        alpha=0.95,
                        zorder=2
                    )
                )

    def _draw_waiting_region(self, ax) -> None:
        """Dashed yellow perimeter + footprints for the world's WAITING_ZONE."""
        region = frozenset(getattr(self.world, 'WAITING_ZONE', None) or ())
        if not region:
            return

        yellow      = (1.00, 0.82, 0.16)   # (255,210,40)  — border
        yellow_dark = (0.75, 0.59, 0.08)   # (190,150,20)  — footprint fill
        dash = (0, (4, 3))
        lw   = 2.0

        # Footprint proportions (foot_w = cell//7, foot_h = cell//4)
        foot_w = 1 / 7
        foot_h = 1 / 4
        gap    = 1 / 12

        for (r, c) in region:
            # Dashed border on exposed edges only — no background fill
            if (r - 1, c) not in region:
                ax.plot([c - 0.5, c + 0.5], [r - 0.5, r - 0.5],
                        color=yellow, lw=lw, ls=dash, zorder=4)
            if (r + 1, c) not in region:
                ax.plot([c - 0.5, c + 0.5], [r + 0.5, r + 0.5],
                        color=yellow, lw=lw, ls=dash, zorder=4)
            if (r, c - 1) not in region:
                ax.plot([c - 0.5, c - 0.5], [r - 0.5, r + 0.5],
                        color=yellow, lw=lw, ls=dash, zorder=4)
            if (r, c + 1) not in region:
                ax.plot([c + 0.5, c + 0.5], [r - 0.5, r + 0.5],
                        color=yellow, lw=lw, ls=dash, zorder=4)

            # Two footprint ovals centred on the cell, side by side
            for sign in (-1, 1):
                ox = sign * (foot_w / 2 + gap / 2)
                ax.add_patch(patches.Ellipse(
                    (c + ox, r),
                    width=foot_w, height=foot_h,
                    facecolor=yellow_dark, edgecolor="none",
                    zorder=4
                ))

    # ----------------------------- Frame ----------------------------------

    def save_frame(self, save_path: Optional[str], agents: List[Any], current_step: int,
                   step_data: Optional[Dict[str, Any]] = None,
                   queried_agent_label: Optional[str] = None,
                   show_queried_agent_star: bool = False,
                   interp_positions: Optional[Dict[str, Tuple[float, float]]] = None,
                   interp_frac: float = 1.0,
                   display_step: Optional[int] = None,
                   push_pairs_override: Optional[List[Dict]] = None):
        """Render a single top-down frame, saving to `save_path` or showing it."""
        fig, ax = plt.subplots(figsize=(12, 8))

        x_overlay = getattr(self.world, 'x_overlay', set())

        # Static background: floor + X-overlay + furniture, pre-composited into a
        # single cached RGBA image and drawn with one imshow. This collapses what
        # used to be three separate imshow() calls (each re-converting/resampling
        # the full canvas every frame) into one — the main per-frame render win.
        if self.sprite_cache is not None:
            static_bg = self.sprite_cache.build_static_background(self.world.grid, x_overlay)
            ax.imshow(
                static_bg,
                origin='upper',
                extent=(-0.5, self.world.width - 0.5, self.world.height - 0.5, -0.5),
                interpolation='nearest',
                zorder=0
            )

        # Non-floor blocks: tables/chairs/counter/door/etc.
        block_styles = {
            1: {"color": "#5f5f5f", "sign": "",   "icon": None},       # wall
            2: {"color": "#63b96a", "sign": "", "icon": None},       # door / entrance
            3: {"color": "#d57d3c", "sign": "",   "icon": "counter"},  # counter
            4: {"color": "#e8cb68", "sign": "",   "icon": None},       # queue
            6: {"color": "#c94f4f", "sign": "",   "icon": "chair"},    # chair
            7: {"color": "#8a5a3b", "sign": "",   "icon": "table"},    # table 2-top
            8: {"color": "#8a5a3b", "sign": "",   "icon": "table"},    # table 4-top
            10: {"color": "#cfd5db", "sign": "",  "icon": None},       # kitchen
            11: {"color": "#b56a3a", "sign": "",  "icon": None},       # laptop bar
            12: {"color": "#3b3b5c", "sign": "",  "icon": None},       # screen room
            13: {"color": "#c9a84c", "sign": "",  "icon": None},       # rope barrier
            14: {"color": "#7a4f8a", "sign": "",  "icon": None},       # theater seat
            15: {"color": "#e6a832", "sign": "",  "icon": None},       # concession stand
            16: {"color": "#e8c94a", "sign": "",  "icon": None},       # snack stand
            17: {"color": "#4a4a4a", "sign": "",  "icon": None},       # vault
            18: {"color": "#5a8faf", "sign": "",  "icon": None},       # ATM room
            19: {"color": "#c9a84c", "sign": "",  "icon": None},       # rope stanchion
            20: {"color": "#6b8e6b", "sign": "",  "icon": None},       # writing desk
            21: {"color": "#e6a832", "sign": "",  "icon": None},       # deposit slips
            22: {"color": "#8a6a5a", "sign": "",  "icon": None},       # waiting chair
            23: {"color": "#cfd5db", "sign": "",  "icon": None},       # stockroom
            24: {"color": "#4aa8c9", "sign": "",  "icon": None},       # medicine shelf
            25: {"color": "#7ec8a8", "sign": "",  "icon": None},       # display case
            26: {"color": "#c95a6a", "sign": "",  "icon": None},       # BP station
            27: {"color": "#e6a832", "sign": "",  "icon": None},       # Rx pickup
            28: {"color": "#cfd5db", "sign": "",  "icon": None},       # storage room
            29: {"color": "#5a9a5a", "sign": "",  "icon": None},       # product shelf
            30: {"color": "#8a8a8a", "sign": "",  "icon": None},       # cart corral
            31: {"color": "#e6a832", "sign": "",  "icon": None},       # bagging area
            32: {"color": "#dbaa82", "sign": "",  "icon": None},       # pastry display
            33: {"color": "#8b5a3c", "sign": "",  "icon": None},       # coffee grinder
            34: {"color": "#3c3c3c", "sign": "",  "icon": None},       # menu board
            35: {"color": "#64646e", "sign": "",  "icon": None},       # espresso machine
            36: {"color": "#4a8a4a", "sign": "",  "icon": None},       # flower pot
            # Clothing Store
            37: {"color": "#8a8a9a", "sign": "",  "icon": None},       # clothing stockroom
            38: {"color": "#c4a882", "sign": "",  "icon": None},       # clothing rack
            39: {"color": "#a07050", "sign": "",  "icon": None},       # fitting room
            40: {"color": "#e0c8b0", "sign": "",  "icon": None},       # mannequin
            41: {"color": "#6a4a3a", "sign": "",  "icon": None},       # shoe display
            # Library
            42: {"color": "#6b4226", "sign": "",  "icon": None},       # bookshelf
            43: {"color": "#8a7a5a", "sign": "",  "icon": None},       # reading desk
            44: {"color": "#5a7aaf", "sign": "",  "icon": None},       # computer station
            45: {"color": "#c94f4f", "sign": "",  "icon": None},       # return drop
            46: {"color": "#a0a0b0", "sign": "",  "icon": None},       # study room
            # Post Office
            47: {"color": "#3a5a8a", "sign": "",  "icon": None},       # mailbox
            48: {"color": "#8a6a4a", "sign": "",  "icon": None},       # package shelf
            49: {"color": "#cfd5db", "sign": "",  "icon": None},       # sorting room
            50: {"color": "#4a6a8a", "sign": "",  "icon": None},       # PO box
            51: {"color": "#8a8a5a", "sign": "",  "icon": None},       # scale station
            # Hospital
            52: {"color": "#5aaaca", "sign": "",  "icon": None},       # nurses station
            53: {"color": "#e0e0e0", "sign": "",  "icon": None},       # hospital bed
            54: {"color": "#8a6a5a", "sign": "",  "icon": None},       # waiting bench
            55: {"color": "#a0c0a0", "sign": "",  "icon": None},       # exam room
            56: {"color": "#c95a6a", "sign": "",  "icon": None},       # triage area
            # Barbershop
            57: {"color": "#c94f4f", "sign": "",  "icon": None},       # barber chair
            58: {"color": "#7a5a4a", "sign": "",  "icon": None},       # waiting couch
            59: {"color": "#b0c0d0", "sign": "",  "icon": None},       # mirror station
            60: {"color": "#5a5a6a", "sign": "",  "icon": None},       # supply cabinet
            61: {"color": "#a08a5a", "sign": "",  "icon": None},       # magazine rack
            # Affordance cells
            62: {"color": "#ffdc32", "sign": "",  "icon": None},       # affordance yellow
            63: {"color": "#c8c8cd", "sign": "X", "icon": None},       # affordance X
            # Color-tagged chairs
            66: {"color": "#dc3c3c", "sign": "",  "icon": "chair"},    # red chair
            67: {"color": "#f59632", "sign": "",  "icon": "chair"},    # orange chair
            68: {"color": "#3c82dc", "sign": "",  "icon": "chair"},    # blue chair
        }

        # Sprite-based cell rendering is handled by build_static_background above.
        if self.sprite_cache is None:
            # Fallback: matplotlib rectangle rendering
            # Draw X-overlay tints first (below furniture) then furniture on top
            for pos in x_overlay:
                r, c = pos
                rect = patches.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    facecolor="#c8c8cd",
                    edgecolor="none",
                    alpha=0.6,
                    zorder=1
                )
                ax.add_patch(rect)
                ax.text(
                    c, r, "X",
                    ha="center", va="center",
                    fontsize=6, fontweight="bold",
                    color="black", zorder=1,
                )

            for r in range(self.world.height):
                for c in range(self.world.width):
                    cell_type = int(self.world.grid[r, c])
                    if cell_type == 0:
                        continue

                    style = block_styles.get(cell_type, {"color": "#ffffff", "sign": ""})

                    rect = patches.Rectangle(
                        (c - 0.5, r - 0.5), 1, 1,
                        facecolor=style["color"],
                        edgecolor="none",
                        alpha=0.95,
                        zorder=2
                    )
                    ax.add_patch(rect)

                    if style.get("sign"):
                        badge = patches.Circle(
                            (c, r), radius=0.17,
                            facecolor="white", edgecolor="black",
                            linewidth=1.0, zorder=3
                        )
                        ax.add_patch(badge)
                        ax.text(
                            c, r, style["sign"],
                            ha="center", va="center",
                            fontsize=6, fontweight="bold",
                            color="black", zorder=3
                        )

        # Display entrance overlay from the active world entrance cells.
        self._draw_display_entrances(ax)

        # Waiting region overlay — yellow tint + dashed border + footprints.
        self._draw_waiting_region(ax)

        # Group agents by position
        position_to_agents = {}
        is_subframe = interp_positions is not None
        for agent in agents:
            show_agent = False

            # Show active agents (including ones that are done but still active).
            # In sub-frames (transition from step N-1 → N), hide agents that only
            # became active this step — they should first appear on the keyframe.
            if agent.active:
                if is_subframe and agent.arrival_time >= current_step:
                    show_agent = False
                else:
                    show_agent = True
            # Show agents arriving this step (keyframe only, not sub-frames)
            elif not agent.active and agent.arrival_time == current_step and not is_subframe:
                show_agent = True
            # Show agents that are still waiting at their target
            elif agent.target_wait_time != 0:
                show_agent = True

            if show_agent:
                pos = agent.current_position
                if pos not in position_to_agents:
                    position_to_agents[pos] = []
                position_to_agents[pos].append(agent)

        # Agent class is the agent's color ('orange' | 'green' | 'blue').
        def get_agent_color(agent_class):
            """Map an agent's color class to a matplotlib color."""
            if agent_class in ('orange', 'green', 'blue'):
                return agent_class
            return 'dodgerblue'  # fallback for unknown classes

        def get_agent_marker(agent_class):
            """All agents share one marker now; color encodes the class."""
            return 'o'

        # Plot agents
        agent_coords = {}

        for pos, agents_at_pos in position_to_agents.items():
            r, c = pos
            num_agents = len(agents_at_pos)

            # Calculate offsets for overlapping agents
            if num_agents == 1:
                offsets = [(0, 0)]
            elif num_agents == 2:
                offsets = [(-0.15, 0), (0.15, 0)]
            elif num_agents == 3:
                offsets = [(-0.2, 0), (0, 0), (0.2, 0)]
            else:
                offsets = []
                radius = 0.2
                for i in range(num_agents):
                    angle = 2 * np.pi * i / num_agents
                    offsets.append((radius * np.cos(angle), radius * np.sin(angle)))

            for i, agent in enumerate(agents_at_pos):
                offset_c, offset_r = offsets[i]

                # For sub-frames, use interpolated float position if provided
                if interp_positions is not None and agent.agent_id in interp_positions:
                    ir, ic = interp_positions[agent.agent_id]
                else:
                    ir, ic = r, c

                # Determine agent state for sprite lookup.
                # Keyframes show agent.action (what the agent is about to do).
                # Sub-frames show agent.previous_action (the action currently animating).
                is_subframe = interp_positions is not None
                # Push badge is deferred one step: it follows previous_action, not the
                # freshly-resolved action. So the step two agents first collide on one
                # cell shows no "!"; it appears the next step, when the pusher actually
                # displaces the other agent (matching the departing push animation).
                effective_push = agent.previous_action == Action.PUSH
                show_push_visuals = effective_push

                if is_subframe:
                    eff_done = getattr(agent, 'previous_done', agent.done)
                    eff_target_index = getattr(agent, 'previous_target_index', getattr(agent, 'target_index', 0))
                    eff_target_wait_time = getattr(agent, 'previous_target_wait_time', agent.target_wait_time)
                else:
                    eff_done = agent.done
                    eff_target_index = getattr(agent, 'target_index', 0)
                    eff_target_wait_time = agent.target_wait_time

                if show_push_visuals:
                    agent_state = 'push'
                elif eff_done or eff_target_index >= 1:
                    agent_state = 'done'
                elif eff_target_wait_time != 0:
                    agent_state = 'waiting'
                else:
                    agent_state = 'normal'

                # Use 'order' sprite for waiting state, which maps to <color>_order.png
                sprite_key = 'order' if agent_state == 'waiting' else agent_state
                # Walking agents face their movement direction. Same convention
                # as the action arrows: sub-frames show the animating action
                # (previous_action), keyframes the upcoming one (action).
                # 'done' agents (check badge) keep facing too, via the
                # synthesized 'done_<dir>' sprites.
                base_key = sprite_key
                if sprite_key in ('normal', 'done'):
                    facing_action = agent.previous_action if is_subframe else agent.action
                    direction = {
                        Action.MOVE_UP: 'up',
                        Action.MOVE_LEFT: 'left',
                        Action.MOVE_RIGHT: 'right',
                    }.get(facing_action)
                    if direction is not None:
                        sprite_key = direction if sprite_key == 'normal' else f'done_{direction}'
                # Sprites are keyed by the agent's color class.
                agent_sprite = self.sprite_cache.get_agent_sprite(agent.agent_class, sprite_key) if self.sprite_cache else None
                if agent_sprite is None and self.sprite_cache is not None and sprite_key != base_key:
                    # Directional sprite not available: fall back to front-facing.
                    agent_sprite = self.sprite_cache.get_agent_sprite(agent.agent_class, base_key)

                if agent_sprite is not None:

                    # Compute extent for the sprite (centered on agent position with offset)
                    sprite_half = 0.36  # a bit smaller than a full cell
                    cx = ic + offset_c
                    cy = ir + offset_r
                    ax.imshow(
                        agent_sprite,
                        extent=(cx - sprite_half, cx + sprite_half,
                                cy + sprite_half, cy - sprite_half),
                        interpolation='nearest',
                        zorder=5
                    )

                    # Name centered just above the agent.
                    ax.text(cx, cy - sprite_half - 0.05, agent.name,
                            ha='center', va='bottom', fontsize=8, fontweight='bold',
                            color='#111111', zorder=7)

                    # Newcomer star overlay (and optionally queried-agent star for ActionQA)
                    should_draw_star = not agent.is_oracle or (
                        show_queried_agent_star and self._is_queried_agent(agent.agent_id, queried_agent_label)
                    )
                    if should_draw_star:
                        llm_star = self.sprite_cache.get_llm_star()
                        if llm_star is not None:
                            star_half = 0.18
                            sx = cx + 0.25
                            sy = cy + 0.25
                            ax.imshow(
                                llm_star,
                                extent=(sx - star_half, sx + star_half,
                                        sy + star_half, sy - star_half),
                                interpolation='nearest',
                                zorder=6
                            )
                        else:
                            ax.plot(cx + 0.25, cy + 0.25, marker='*', markersize=12,
                                    color='gold', markeredgecolor='darkorange', markeredgewidth=1.5,
                                    zorder=6)
                else:
                    # Fallback: matplotlib marker rendering
                    marker = get_agent_marker(agent.agent_class)
                    color = get_agent_color(agent.agent_class)
                    edge_color = 'black'
                    edge_width = 2

                    if show_push_visuals:
                        color = 'purple'
                        edge_color = 'darkviolet'
                        edge_width = 4
                    elif eff_target_wait_time != 0:
                        color = 'orange'
                        edge_color = 'gold'
                        edge_width = 3

                    ax.plot(ic + offset_c, ir + offset_r, marker=marker, markersize=20,
                           color=color, markeredgecolor=edge_color, markeredgewidth=edge_width,
                           label=agent.agent_id)
                    ax.text(ic + offset_c, ir + offset_r - 0.4, agent.name,
                            ha='center', va='bottom', fontsize=8, fontweight='bold',
                            color='#111111', zorder=7)

                    should_draw_star = not agent.is_oracle or (
                        show_queried_agent_star and self._is_queried_agent(agent.agent_id, queried_agent_label)
                    )
                    if should_draw_star:
                        ax.plot(ic + offset_c + 0.25, ir + offset_r + 0.25, marker='*', markersize=12,
                               color='gold', markeredgecolor='darkorange', markeredgewidth=1.5,
                               zorder=6)

                # Track plot coords (violation marks are drawn on the agent)
                agent_coords[agent.agent_id] = (ic + offset_c, ir + offset_r)

        # Goal marker for human-controlled agents: outline + "GOAL n" badge on the
        # current target cell, so players can find it without translating (row, col)
        # text. Gated on the HumanAgent class flag show_goal_marker (not on
        # display_target, which every agent has) so oracle/LLM frames — including
        # the LLM eval figures — are unchanged. Free-choice goals -> no marker.
        for agent in agents:
            if agent.done or not getattr(agent, 'show_goal_marker', False):
                continue
            target = agent.display_target()
            if target is None:
                continue
            tr, tc = target
            goal_red = '#e63946'
            # zorder 4.5: above the floor/furniture/region overlays (<= 4) but
            # below agent sprites (5) and their name labels (7), so a player
            # standing on the goal cell is drawn on top of the marker.
            ax.add_patch(patches.Rectangle(
                (tc - 0.5, tr - 0.5), 1, 1,
                facecolor='none', edgecolor=goal_red, linewidth=3,
                linestyle=(0, (4, 2)), zorder=4.5
            ))
            goal_num = getattr(agent, 'target_index', 0) + 1
            ax.text(
                tc, tr - 0.58, f'GOAL {goal_num}',
                ha='center', va='bottom', fontsize=9, fontweight='bold',
                color='white', zorder=4.5,
                bbox=dict(boxstyle='round,pad=0.25', facecolor=goal_red,
                          edgecolor='none'),
            )

        x_size = 0.3

        # Violations are stored in the current step's data and rendered at the keyframe only (sub0).
        if step_data and interp_positions is None:
            for violation in step_data.get('norm_violations', []):
                agent_id = violation[0]
                if agent_id not in agent_coords:
                    continue
                vc, vr = agent_coords[agent_id]
                ax.plot([vc - x_size, vc + x_size], [vr - x_size, vr + x_size],
                       'r-', linewidth=4, zorder=20)
                ax.plot([vc - x_size, vc + x_size], [vr + x_size, vr - x_size],
                       'r-', linewidth=4, zorder=20)

        ax.set_title(f'Step {display_step if display_step is not None else current_step}', fontsize=16, weight='bold')
        ax.set_xlabel('Column', fontsize=12)
        ax.set_ylabel('Row', fontsize=12)
        ax.set_xlim(-0.5, self.world.width - 0.5)
        ax.set_ylim(self.world.height - 0.5, -0.5)
        ax.set_xticks(range(self.world.width))
        ax.set_yticks(range(self.world.height))

        plt.tight_layout()

        if save_path:
            savefig_kwargs = {'dpi': self.save_dpi}
            if self.tight_bbox:
                savefig_kwargs['bbox_inches'] = 'tight'
            plt.savefig(save_path, **savefig_kwargs)
            print(f"  [Figure saved to {save_path}]")
        else:
            plt.show()

        plt.close()
