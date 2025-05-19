"""
This module defines a class "MapList" for mapping keys to lists.
"""
from __future__ import annotations

from typing import TypeVar, Generic, List, Dict, MutableMapping, KeysView, ValuesView, ItemsView, Iterable, Iterator

_K = TypeVar('_K')
_V = TypeVar('_V')


class MapList(Generic[_K, _V], MutableMapping[_K, List[_V]]):
    """
    A MapList keeps a list for each key, unlike a dict which keeps only
    a single element for each key.
    """
    __slots__ = ('_map', )

    def __init__(self, *args, **kwargs):
        self._map: Dict[_K, List[_V]] = {}
        self.update(*args, extend=False, **kwargs)

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

    def __getitem__(self, key) -> List[_V]:
        return self._map[key]

    def __setitem__(self, key: _K, val: List[_V]):
        if not isinstance(val, list):
            class_name = self.__class__.__name__
            raise RuntimeError(f'every {class_name} value must be a list')
        self._map[key] = val

    def __delitem__(self, key: _K):
        del self._map[key]

    def __iter__(self) -> Iterator[_K]:
        return iter(self._map)

    def __contains__(self, key: _K) -> bool:
        return key in self._map

    def update(self, *args, extend=False, **kwargs):
        k: _K
        v: List[_V]
        if extend:
            for k, v in dict(*args, **kwargs).items():
                self.extend(k, v)
        else:
            for k, v in dict(*args, **kwargs).items():
                self[k] = v

    def keys(self) -> KeysView[_K]:
        return self._map.keys()

    def values(self) -> ValuesView[List[_V]]:
        return self._map.values()

    def items(self) -> ItemsView[_K, List[_V]]:
        return self._map.items()

    def get(self, key: _K, default=None):
        """
        Get the list corresponding to the given key.
        If the key is not yet in the MapList then the
        supplied default will be returned.
        """
        return self._map.get(key, default)

    def get_list(self, key: _K) -> List[_V]:
        """
        Get the list corresponding to the given key.

        This method will always return a list in the MapList, even if
        it requires a new list being created.

        Modifying the returned list affects this MapList object.
        """
        the_list = self._map.get(key)
        if the_list is None:
            the_list = []
            self._map[key] = the_list
        return the_list

    def append(self, key: _K, item: _V):
        """
        Append the given item to the list identified by the given key.
        """
        self.get_list(key).append(item)

    def extend(self, key: _K, items: Iterable[_V]):
        """
        Extend the given item to the list identified by the given key.
        """
        return self.get_list(key).extend(items)

    def extend_map_list(self, map_list: MapList[_K, _V]):
        """
        Add all the keyed given items to the list identified by each key.
        """
        for key, items in map_list.items():
            self.extend(key, items)

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
