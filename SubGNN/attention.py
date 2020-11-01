import torch
from torch.nn.parameter import Parameter

# All of the below code is taken from AllenAI's AllenNLP library

def tiny_value_of_dtype(dtype: torch.dtype):
    """
    Returns a moderately tiny value for a given PyTorch data type that is used to avoid numerical
    issues such as division by zero.
    This is different from `info_value_of_dtype(dtype).tiny` because it causes some NaN bugs.
    Only supports floating point dtypes.
    """
    if not dtype.is_floating_point:
        raise TypeError("Only supports floating point dtypes.")
    if dtype == torch.float or dtype == torch.double:
        return 1e-13
    elif dtype == torch.half:
        return 1e-4
    else:
        raise TypeError("Does not support dtype " + str(dtype))

def masked_softmax(
    vector: torch.Tensor, mask: torch.BoolTensor, dim: int = -1, memory_efficient: bool = False,
) -> torch.Tensor:
    """
    `torch.nn.functional.softmax(vector)` does not work if some elements of `vector` should be
    masked.  This performs a softmax on just the non-masked portions of `vector`.  Passing
    `None` in for the mask is also acceptable; you'll just get a regular softmax.
    `vector` can have an arbitrary number of dimensions; the only requirement is that `mask` is
    broadcastable to `vector's` shape.  If `mask` has fewer dimensions than `vector`, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    If `memory_efficient` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.
    In the case that the input vector is completely masked and `memory_efficient` is false, this function
    returns an array of `0.0`. This behavior may cause `NaN` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if `memory_efficient` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (
                result.sum(dim=dim, keepdim=True) + tiny_value_of_dtype(result.dtype)
            )
        else:
            masked_vector = vector.masked_fill(~mask, min_value_of_dtype(vector.dtype))
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result



class Attention(torch.nn.Module):
    """
    An `Attention` takes two inputs: a (batched) vector and a matrix, plus an optional mask on the
    rows of the matrix.  We compute the similarity between the vector and each row in the matrix,
    and then (optionally) perform a softmax over rows using those computed similarities.
    Inputs:
    - vector: shape `(batch_size, embedding_dim)`
    - matrix: shape `(batch_size, num_rows, embedding_dim)`
    - matrix_mask: shape `(batch_size, num_rows)`, specifying which rows are just padding.
    Output:
    - attention: shape `(batch_size, num_rows)`.
    # Parameters
    normalize : `bool`, optional (default = `True`)
        If true, we normalize the computed similarities with a softmax, to return a probability
        distribution for your attention.  If false, this is just computing a similarity score.
    """

    def __init__(self, normalize: bool = True) -> None:
        super().__init__()
        self._normalize = normalize

    def forward(
        self, vector: torch.Tensor, matrix: torch.Tensor, matrix_mask: torch.BoolTensor = None
    ) -> torch.Tensor:
        similarities = self._forward_internal(vector, matrix)
        if self._normalize:
            return masked_softmax(similarities, matrix_mask)
        else:
            return similarities

    def _forward_internal(self, vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class DotProductAttention(Attention):
    """
    Computes attention between a vector and a matrix using dot product.
    Registered as an `Attention` with name "dot_product".
    """

    def _forward_internal(self, vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        return matrix.bmm(vector.unsqueeze(-1)).squeeze(-1)

class AdditiveAttention(Attention):
    """
    Computes attention between a vector and a matrix using an additive attention function.  This
    function has two matrices `W`, `U` and a vector `V`. The similarity between the vector
    `x` and the matrix `y` is computed as `V tanh(Wx + Uy)`.
    This attention is often referred as concat or additive attention. It was introduced in
    <https://arxiv.org/abs/1409.0473> by Bahdanau et al.
    Registered as an `Attention` with name "additive".
    # Parameters
    vector_dim : `int`, required
        The dimension of the vector, `x`, described above.  This is `x.size()[-1]` - the length
        of the vector that will go into the similarity computation.  We need this so we can build
        the weight matrix correctly.
    matrix_dim : `int`, required
        The dimension of the matrix, `y`, described above.  This is `y.size()[-1]` - the length
        of the vector that will go into the similarity computation.  We need this so we can build
        the weight matrix correctly.
    normalize : `bool`, optional (default : `True`)
        If true, we normalize the computed similarities with a softmax, to return a probability
        distribution for your attention.  If false, this is just computing a similarity score.
    """

    def __init__(self, vector_dim: int, matrix_dim: int, normalize: bool = True) -> None:
        super().__init__(normalize)
        self._w_matrix = Parameter(torch.Tensor(vector_dim, vector_dim))
        self._u_matrix = Parameter(torch.Tensor(matrix_dim, vector_dim))
        self._v_vector = Parameter(torch.Tensor(vector_dim, 1))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self._w_matrix)
        torch.nn.init.xavier_uniform_(self._u_matrix)
        torch.nn.init.xavier_uniform_(self._v_vector)

    def _forward_internal(self, vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        intermediate = vector.matmul(self._w_matrix).unsqueeze(1) + matrix.matmul(self._u_matrix)
        intermediate = torch.tanh(intermediate)
        return intermediate.matmul(self._v_vector).squeeze(2)

