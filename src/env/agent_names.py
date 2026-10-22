"""Display-name mapping for agents.

Agents keep their internal ``agent_id`` of the form ``"C1"``, ``"C2"`` ...
(relied on throughout queueing, collision, and norm logic). For rendering we
map that 1-indexed id to a short unique name so labels read as people rather
than ``C1``/``C2``/``C3``.
"""

AGENT_NAMES = [
    "Ash", "Bob", "Cal", "Drew", "Eden", "Finn", "Grey", "Hart", "Iris", "Jude",
    "Kai", "Lane", "Mars", "Noel", "Oryn", "Page", "Quin", "Rory", "Sage", "Tate",
    "Uri", "Vale", "Wren", "Xen", "Yael", "Zion", "Arlo", "Brae", "Cleo", "Dex",
    "Emre", "Fox", "Glen", "Haze", "Iden", "Jory", "Kael", "Lux", "Milo", "Nash",
    "Odin", "Pax", "Reed", "Sky", "True", "Aven", "Bleu", "Crew", "Dani", "Ezra",
    "Faye", "Gus", "Hollis", "Ivo", "Juno", "Kip", "Lior", "Moss", "Nova", "Oak",
    "Pia", "Quill", "Rune", "Silas", "Thea", "Ulla", "Vera", "Wade", "Xander", "Yara",
    "Zane", "Ada", "Beck", "Cove", "Dove", "Echo", "Fern", "Gale", "Hugo", "Iona",
    "Joss", "Knox", "Lark", "Mira", "Nero", "Onyx", "Perry", "Remy", "Soren", "Toby",
    "Otto", "Vin", "Wells", "Xia", "Yves", "Zia", "Alma", "Bran", "Cira", "Dorn",
]


def agent_display_name(agent_id: str) -> str:
    """Map an internal ``agent_id`` (e.g. ``"C3"``) to its display name.

    Falls back to the raw id if it is not in the expected ``C<n>`` form or the
    index is out of range.
    """
    if isinstance(agent_id, str) and agent_id.startswith("C"):
        suffix = agent_id[1:]
        if suffix.isdigit():
            idx = int(suffix) - 1
            if 0 <= idx < len(AGENT_NAMES):
                return AGENT_NAMES[idx]
    return agent_id
