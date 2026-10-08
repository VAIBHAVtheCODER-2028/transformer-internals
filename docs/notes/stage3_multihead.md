1) Why must d_model be divisible by n_heads?
okay if d_model will not be divisible by n_heads(integer division) then the value d_k=d_model//n_heads will lead to dropping of one dimension or the head size would need to be different.


2) What's the purpose of self.proj after concatenation — why not just return the concatenated heads directly?
It is necessary in order to mix information across multiple heads , map the dimensions back to d_model for residual connections, and provide additional learnable capacity.
