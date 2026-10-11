"""Label taxonomy.

The original project used one 4-way softmax over folder names. Here a sample is described
by two independent factors (which machine, is it occupied) so the model gets a dedicated
occupancy head that is shared across machines and can be extended to new equipment.
"""

from __future__ import annotations

from dataclasses import dataclass

EQUIPMENT: tuple[str, ...] = ("leg_extension", "pec_fly")
OCCUPANCY: tuple[str, ...] = ("free", "in_use")

_FOLDER_EQUIPMENT = {"LEG-EXTENSION": "leg_extension", "PEC-FLY": "pec_fly"}


@dataclass(frozen=True)
class Label:
    equipment: str
    occupancy: str

    @property
    def equipment_idx(self) -> int:
        return EQUIPMENT.index(self.equipment)

    @property
    def occupancy_idx(self) -> int:
        return OCCUPANCY.index(self.occupancy)

    @property
    def joint(self) -> str:
        return f"{self.equipment}/{self.occupancy}"


def parse_folder(name: str) -> Label:
    """Map dataset folder names like ``NOT-IN-USE-PEC-FLY`` to a :class:`Label`."""
    upper = name.upper()
    if upper.startswith("NOT-IN-USE-"):
        occupancy, rest = "free", upper.removeprefix("NOT-IN-USE-")
    elif upper.startswith("IN-USE-"):
        occupancy, rest = "in_use", upper.removeprefix("IN-USE-")
    else:
        raise ValueError(f"unrecognised class folder: {name!r}")
    if rest not in _FOLDER_EQUIPMENT:
        raise ValueError(f"unknown equipment in folder name: {name!r}")
    return Label(_FOLDER_EQUIPMENT[rest], occupancy)


JOINT_CLASSES: tuple[str, ...] = tuple(f"{e}/{o}" for e in EQUIPMENT for o in OCCUPANCY)
