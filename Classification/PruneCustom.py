import torch
import torch.nn.utils.prune as prune
from torch import nn
import torch.nn.functional as F
from scipy.spatial import distance
import numpy as np

class GMStructured(prune.BasePruningMethod):
    """Prune every other entry in a tensor
    """
    PRUNING_TYPE = 'structured'

    def __init__(self, amount, dim=-1):
        prune._validate_pruning_amount_init(amount)
        self.amount = amount
        self.dim = dim

    def compute_mask(self, t, default_mask):
        r"""Computes and returns a mask for the input tensor ``t``.
                Starting from a base ``default_mask`` (which should be a mask of ones
                if the tensor has not been pruned yet), generate a mask to apply on
                top of the ``default_mask`` by zeroing out the channels along the
                specified dim with the lowest L\ ``n``-norm.

                Args:
                    t (torch.Tensor): tensor representing the parameter to prune
                    default_mask (torch.Tensor): Base mask from previous pruning
                        iterations, that need to be respected after the new mask is
                        applied.  Same dims as ``t``.

                Returns:
                    mask (torch.Tensor): mask to apply to ``t``, of same dims as ``t``

                Raises:
                    IndexError: if ``self.dim >= len(t.shape)``
                """
        # Check that tensor has structure (i.e. more than 1 dimension) such
        # that the concept of "channels" makes sense
        prune._validate_structured_pruning(t)
        # Check that self.dim is a valid dim to index t, else raise IndexError
        prune._validate_pruning_dim(t, self.dim)

        # Check that the amount of channels to prune is not > than the number of
        # channels in t along the dim to prune
        tensor_size = t.shape[self.dim]
        # Compute number of units to prune: amount if int,
        # else amount * tensor_size
        nparams_toprune = prune._compute_nparams_toprune(self.amount, tensor_size)
        nparams_tokeep = tensor_size - nparams_toprune
        # This should raise an error if the number of units to prune is larger
        # than the number of units in the tensor
        prune._validate_pruning_amount(nparams_toprune, tensor_size)

        # Structured pruning prunes entire channels so we need to know the
        # L_n norm along each channel to then find the topk based on this
        # metric
        # norm = prune._compute_norm(t, self.n, self.dim)
        t_2d = t.view(t.size()[0], -1)
        similar_matrix = distance.cdist(t_2d.detach().numpy(), t_2d.detach().numpy(), 'euclidean')
        similar_sum = np.sum(np.abs(similar_matrix), axis=0)
        tensor_sort = torch.tensor(similar_sum.argsort())

        # largest=True --> top k; largest=False --> bottom k
        # Keep the largest k channels along dim=self.dim
        topk = torch.topk(tensor_sort, k=nparams_tokeep, largest=True)

        # topk will have .indices and .values

        # Compute binary mask by initializing it to all 0s and then filling in
        # 1s wherever topk.indices indicates, along self.dim.
        # mask has the same shape as tensor t
        def make_mask(t, dim, indices):
            # init mask to 0
            mask = torch.zeros_like(t)
            # e.g.: slc = [None, None, None], if len(t.shape) = 3
            slc = [slice(None)] * len(t.shape)
            # replace a None at position=dim with indices
            # e.g.: slc = [None, None, [0, 2, 3]] if dim=2 & indices=[0,2,3]
            slc[dim] = indices
            # use slc to slice mask and replace all its entries with 1s
            # e.g.: mask[:, :, [0, 2, 3]] = 1
            mask[slc] = 1
            return mask

        if nparams_toprune == 0:  # k=0 not supported by torch.kthvalue
            mask = default_mask
        else:
            mask = make_mask(t, self.dim, topk.indices)
            mask *= default_mask.to(dtype=mask.dtype)
        return mask



def gm_structured(module, name, amount, dim, importance_scores=None):
    """Prunes tensor corresponding to parameter called `name` in `module`
    by removing every other entry in the tensors.
    Modifies module in place (and also return the modified module)
    by:
    1) adding a named buffer called `name+'_mask'` corresponding to the
    binary mask applied to the parameter `name` by the pruning method.
    The parameter `name` is replaced by its pruned version, while the
    original (unpruned) parameter is stored in a new parameter named
    `name+'_orig'`.

    Args:
        module (nn.Module): module containing the tensor to prune
        name (string): parameter name within `module` on which pruning
                will act.

    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input
            module

    Examples:
        >>> m = nn.Linear(3, 4)
        >>> foobar_unstructured(m, name='bias')
    """
    GMStructured.apply(module, name, amount, dim, importance_scores=importance_scores)
    return module
