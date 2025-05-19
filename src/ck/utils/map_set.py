"""
This module defines a class "MapSet" for mapping keys to sets.
"""
from __future__ import annotations

from typing import TypeVar, Generic, Set, Dict, MutableMapping, Iterator, KeysView, ValuesView, ItemsView, Iterable

_K = TypeVar('_K')
_V = TypeVar('_V')


class MapSet(Generic[_K, _V], MutableMapping[_K, Set[_V]]):
    """
    A MapSet keeps a set for each key, unlike a dict which keeps only
    a single element for each key.
    """
    __slots__ = ('_map',)

    def __init__(self, *args, **kwargs):
        self._map: Dict[_K, Set[_V]] = {}
        self.update(*args, add_all=False, **kwargs)

    def __str__(self) -> str:
        return str(self._map)

    def __repr__(self) -> str:
        args = ', '.join(f'{key!r}:{val!r}' for key, val in self.items())
        class_name = self.__class__.__name__
        return f'{class_name}({args})'

    def __len__(self) -> int:
        return len(self._map)

    def __bool__(self) -> bool:
        return len(self) > 0

    def __getitem__(self, key: _K) -> Set[_V]:
        return self._map[key]

    def __setitem__(self, key: _K, val: Set[_V]):
        if not isinstance(val, set):
            class_name = self.__class__.__name__
            raise RuntimeError(f'every {class_name} value must be a set')
        self._map[key] = val

    def __delitem__(self, key: _K):
        del self._map[key]

    def __iter__(self) -> Iterator[_K]:
        return iter(self._map)

    def __contains__(self, key: _K) -> bool:
        return key in self._map

    def update(self, *args, add_all=True, **kwargs):
        k: _K
        v: Set[_V]
        if add_all:
            for k, v in dict(*args, **kwargs).items():
                self.add_all(k, v)
        else:
            for k, v in dict(*args, **kwargs).items():
                self[k] = v

    def keys(self) -> KeysView[_K]:
        return self._map.keys()

    def values(self) -> ValuesView[Set[_V]]:
        return self._map.values()

    def items(self) -> ItemsView[_K, Set[_V]]:
        return self._map.items()

    def get(self, key: _K, default=None):
        """
        Get the set corresponding to the given key.
        If the key is not yet in the MapSet then the
        supplied default will be returned.
        """
        return self._map.get(key, default)

    def get_set(self, key: _K) -> Set[_V]:
        """
        Get the set corresponding to the given key.

        This method will always return a set in the MapSet, even if
        it requires a new set being created.

        Modifying the returned set affects this MapSet object.
        """
        the_set = self._map.get(key)
        if the_set is None:
            the_set = set()
            self._map[key] = the_set
        return the_set

    def add(self, key: _K, item: _V):
        """
        Add the given item to the set identified by the given key.
        """
        self.get_set(key).add(item)

    def add_all(self, key: _K, items: Iterable[_V]):
        """
        Add all the given items to the set identified by the given key.
        """
        return self.get_set(key).update(items)

    def add_map_set(self, map_set: MapSet[_K, _V]):
        """
        Add all the keyed given items to the set identified by each key.
        """
        for key, items in map_set.items():
            self.add_all(key, items)

    def clear(self):
        """
        Remove all items.
        """
        return self._map.clear()

    def clear_empty(self):
        """
        Remove all empty values.
        """
        keys_to_remove = [key for key, value in self._map.items() if len(value) == 0]
        for key in keys_to_remove:
            del self._map[key]
