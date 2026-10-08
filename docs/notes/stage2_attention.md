1) Why scale by 1/√d_k?
We do this in order to prevent gradients from vanishing,stalling learning because the dot product sometimes becomes very large so dividing by root dk helps to bring down dot product roughly to unit variance,

2) Why mask before softmax, not after?
It ensures the position which are forbidden never participate in computing the prob. distribution as their contribution is taken up as e(-inf) which is 0.Also if they are not masked then they would influence normalization and thus the prob. computed would not be right.
