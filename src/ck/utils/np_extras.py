# Type used by numpy and ctypes for data type definition

import numpy as np

# A numpy/ctypes data type.
DType = np.typing.DTypeLike

# A numpy data type for holding state indexes.
DTypeStates = np.dtypes.UInt8DType | np.dtypes.UInt16DType | np.dtypes.UInt32DType | np.dtypes.UInt64DType

DTypeNumeric = (
        np.dtypes.Float64DType | np.dtypes.Float32DType
        | np.dtypes.Int8DType | np.dtypes.Int16DType | np.dtypes.Int32DType | np.dtypes.Int64DType
        | np.dtypes.UInt8DType | np.dtypes.UInt16DType | np.dtypes.UInt32DType | np.dtypes.UInt64DType
)

# A numpy array data type.
NDArray = np.typing.NDArray

NDArrayUInt8 = NDArray[np.dtypes.UInt8DType]
NDArrayUInt16 = NDArray[np.dtypes.UInt16DType]
NDArrayUInt32 = NDArray[np.dtypes.UInt32DType]
NDArrayUInt64 = NDArray[np.dtypes.UInt64DType]

NDArrayFloat64 = NDArray[np.dtypes.Float64DType]

NDArrayStates = NDArray[DTypeStates]
NDArrayNumeric = NDArray[DTypeNumeric]


# Constants for maximum number of states.
_MAX_STATES_8: int = 2 ** 8
_MAX_STATES_16: int = 2 ** 16
_MAX_STATES_32: int = 2 ** 32
_MAX_STATES_64: int = 2 ** 64


def dtype_for_number_of_states(number_of_states: int) -> DTypeStates:
    """
    Infer the numpy dtype required to store any state index of the given PGM.
    """
    # Infer the needed dtype required to hold a number of states
    if number_of_states <= _MAX_STATES_8:
        return np.uint8
    if number_of_states <= _MAX_STATES_16:
        return np.uint16
    if number_of_states <= _MAX_STATES_32:
        return np.uint32
    if number_of_states <= _MAX_STATES_64:
        return np.uint64
    raise ValueError(f'cannot determine dtype for the given number of states: {number_of_states!r}')
