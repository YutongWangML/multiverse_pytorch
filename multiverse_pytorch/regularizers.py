import torch

class GFL1:
    """
    A class that implements the Group fused L1 regularization regularization function for a given value of k.

    The GFL1 regularization is calculated based on the weight differences between pairs
    of classes and the L1 norm of the weight.

    Attributes:
        num_classes (int): The number of classes.
        A (torch.Tensor): A helper matrix used to compute the pairwise weight differences.

    Methods:
        __call__(weight): Computes the GFL1 regularization for the given weight.
    """

    def __init__(self, num_classes):
        km1 = num_classes-1
        num_combinations = km1 * (km1 - 1) // 2
        self.A = torch.zeros((num_combinations, km1))

        idx = 0
        for i in range(km1):
            for j in range(i + 1, km1):
                self.A[idx, i] = 1
                self.A[idx, j] = -1
                idx += 1

    def __call__(self, weight):
        diff_weight = self.A @ weight
        return torch.norm(diff_weight, p=1) + torch.norm(weight, p=1)
