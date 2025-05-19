from typing import Protocol, Optional, overload, Iterable, Tuple

from ck.pgm import Indicator, ParamId

# Type of a slot map key.
SlotKey = Indicator | ParamId


class SlotMap(Protocol):
    """
    A slotmap is a protocol for mapping keys (indicators and
    parameter ids) to slots in a ProgramBuffer.

    A Python dict[SlotKey, int] implements the protocol.
    """

    def __len__(self) -> int:
        ...

    @overload
    def get(self, slot_key: SlotKey, default: None) -> Optional[int]:
        ...

    @overload
    def get(self, slot_key: SlotKey, default: int) -> int:
        ...

    def get(self, slot_key: SlotKey, default: Optional[int]) -> Optional[int]:
        ...

    def __getitem__(self, slot_key: SlotKey) -> int:
        ...

    def items(self) -> Iterable[Tuple[SlotKey, int]]:
        ...
