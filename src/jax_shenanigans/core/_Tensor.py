from functools import wraps

from jax.numpy import ndarray
from jaxlib.xla_extension import DeviceArray

from ._DAGNode import DAGNode
from .const import BINARY_OPERATORS, UNARY_OPERATORS


class Tensor(DAGNode):
    def __init__(self, x: ndarray, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._x = x

    def __repr__(self) -> str:
        return f"jax_shenanigans.Tensor({self._x})"


def _binary_op_wrap(op_name):
    op = getattr(DeviceArray, op_name)

    @wraps(op)
    def opwrap(self, other: Tensor):
        return Tensor(op(self._x, other._x), parents=[self, other])

    return opwrap


def _unary_op_wrap(op_name):
    op = getattr(DeviceArray, op_name)

    @wraps(op)
    def opwrap(self):
        return Tensor(op(self._x), parents=[self])

    return opwrap


for operator in BINARY_OPERATORS:
    setattr(Tensor, operator, _binary_op_wrap(operator))
for operator in UNARY_OPERATORS:
    setattr(Tensor, operator, _unary_op_wrap(operator))
