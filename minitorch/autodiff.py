from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vals1 = [v for v in vals]
    vals2 = [v for v in vals]
    vals1[arg] = vals1[arg] + epsilon
    vals2[arg] = vals2[arg] - epsilon
    delta = f(*vals1) - f(*vals2)
    return delta / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """
        Accumulates the derivative (gradient) for this Variable.

        Args:
            x (Any): The gradient value to be accumulated.
        """
        pass

    @property
    def unique_id(self) -> int:
        """
        Returns:
            int: The unique identifier of this Variable.
        """
        pass

    def is_leaf(self) -> bool:
        """
        Returns whether this Variable is a leaf node in the computation graph.

        Returns:
            bool: True if this Variable is a leaf node, False otherwise.
        """
        pass

    def is_constant(self) -> bool:
        """
        Returns whether this Variable represents a constant value.

        Returns:
            bool: True if this Variable is constant, False otherwise.
        """
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        """
        Returns the parent Variables of this Variable in the computation graph.

        Returns:
            Iterable[Variable]: The parent Variables of this Variable.
        """
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        """
        Implements the chain rule to compute the gradient contributions of this Variable.

        Args:
            d_output (Any): The gradient of the output with respect to the Variable.

        Returns:
            Iterable[Tuple[Variable, Any]]: An iterable of tuples, where each tuple
                contains a parent Variable and the corresponding gradient contribution.
        """
        pass

def _var_parent_ids(variable: Variable) -> Iterable(int):
    """Get the ids of this variable's immediate parents."""
    return (p.unique_id for p in variable.parents if !p.is_constant())


def _count_in_edges(variable: Variable, counts: Counter, var_index: dict[int, Variable]) -> None:
    """
    Counts the number of parent edges into each
    Mutates the counts dict in-place.
    """
    if variable.is_constant():
        return

    var_index[variable.unique_id] = variable

    if variable.is_leaf():
        return

    counts.update(_var_parent_ids(variable))

    for p in variable.parents:
        _count_in_edges(p, counts, var_index)


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # BEGIN ASSIGN1_1
    # compute the in-edges for each node in the graph
    in_edges = Counter()
    var_index = {}
    _count_in_edges(variable, in_edges, var_index)

    while var_index:
        # get a var with no in edges
        var = var_index[min(var_index.keys, key=lambda vid: in_edges[vid])]
        if in_edges[var]:
            raise Exception(f"Error: minimum node {var.unique_id} has {in_edges[var]} in edges!!")

        # remove var from the graph
        yield var
        del in_edges[var.unique_id]
        del var_index[var.unique_id]
        in_edges.subtract(_var_parent_ids(var))

        # check for empty graph
        if !var_index:
            if in_edges:
                raise Exception(f"Error: edges left in graph after processing!")
            break

    # END ASSIGN1_1


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # BEGIN ASSIGN1_1

    # base case - variable is a leaf - accumulate deriv here
    if variable.is_leaf():
        variable.accumulate_derivative(deriv)
        return

    # recursive case - propagate deriv using the chain rule
    for (var, d_part) in variable.chain_rule(deriv):
        backpropagate(var, d_part)

    # END ASSIGN1_1


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
