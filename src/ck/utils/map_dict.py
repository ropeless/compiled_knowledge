"""
This module defines a class "MapDict" for mapping keys to dicts.
"""

from typing import TypeVar, Generic, Dict, MutableMapping, Iterator, KeysView, ValuesView, ItemsView

_K = TypeVar('_K')
_KV = TypeVar('_KV')
_V = TypeVar('_V')


class MapDict(Generic[_K, _KV, _V], MutableMapping[_K, Dict[_KV, _V]]):
    """
    A MapDict keeps a dict for each key.
    """
    __slots__ = ('_map',)

    def __init__(self):
        self._map: Dict[_K, Dict[_KV, _V]] = {}

    def __str__(self) -> str:
        return str(self._map)

    def __repr__(self) -> str:
        args = ', '.join(f'{key!r}:{key!r}' for key, val in self.items())
        class_name = self.__class__.__name__
        return f'{class_name}({args})'

    def __len__(self) -> int:
        return len(self._map)

    def __bool__(self) -> bool:
        return len(self) > 0

    def __getitem__(self, key: _K) -> Dict[_KV, _V]:
        return self._map[key]

    def __setitem__(self, key: _K, val: Dict[_KV, _V]):
        if not isinstance(val, dict):
            class_name = self.__class__.__name__
            raise RuntimeError(f'every {class_name} value must be a dict')
        self._map[key] = val

    def __delitem__(self, key: _K):
        del self._map[key]

    def __iter__(self) -> Iterator[_K]:
        return iter(self._map)

    def __contains__(self, key: _K) -> bool:
        return key in self._map

    def keys(self) -> KeysView[_K]:
        return self._map.keys()

    def values(self) -> ValuesView[Dict[_KV, _V]]:
        return self._map.values()

    def items(self) -> ItemsView[_K, Dict[_KV, _V]]:
        return self._map.items()

    def get(self, key: _K, default=None):
        """
        Get the list corresponding to the given key.
        If the key is not yet in the MapList then the
        supplied default will be returned.
        """
        return self._map.get(key, default)

    def get_dict(self, key: _K) -> Dict[_KV, _V]:
        """
        Get the dict corresponding to the given key.

        This method will always return a dict in the MapDict, even if
        it requires a new dict being created.

        Modifying the returned dict affects this MapDict object.
        """
        the_dict = self._map.get(key)
        if the_dict is None:
            the_dict = {}
            self._map[key] = the_dict
        return the_dict

    def clear(self):
        """
        Remove all items.
        """
        return self._map.clear()
