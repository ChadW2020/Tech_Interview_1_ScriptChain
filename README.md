# Tech_Interview_1_ScriptChain
Answer to the first interview for the ML internship position.

Q1: Suppose that we design a deep architecture to represent a sequence by stacking self-attention layers with positional encoding. What could be issues? (paragraph format)

Ans:

Since self-attention mechanisms will not take account of the sequence information from the input sentence, we need to addin such information with a positional encoding vector.

The positional encoding vector is added to the input token sequence before input to the first layer. Encoding vectors are no longer neccesary for the rest layers, as the position information will already be merged into the learned attention values that output from the first layer.

Positional encoding can be either absolute or relative. The difference is, absolute positional encoding of each token will not be effected by the location of other tokens, while it is not true in relative positional encoding.

Original Transformers model use an un-learnable absolute positional encoding vector to store the sequence information, which is a combination of sine and cosine functions with different period. This encoding method limit the input length of sequence - if your model is trained with a fixed K length sequence, it will not be able to take input sequence longer than K. Besides, this method is theoretically less efficient than relative positional encoding, on capturing the information from a token sequence. For example, it would be more reasonable to have two sentences "it turns dark" and "now it turns dark" share a common positional information between token "it" and "dark".

Relative positional encoding stores the pairwised distance information among positions. For K length of input sequence, we can simply use a length K vector to store the absolute position. However, it will require a K by (2K - 1) matrix to store the relative position information. This memory consumption can be improved by a "skewing" machenism, introduced by Huang et al.

Rotory positional encoding is another approach to overcome the disadvantage of absolute positional encoding. It introduced an rotation matrix, with each of its 2 x 2 diagnal block equals a 2D rotation matrix of angle theta_i. As a result, each pair of 2D vectors will be rotated by different angle, according to their positions. Relative positional information is therefore stored in the rotation angle.  

Other issues:

Use multi-head attention to parallize the information processing. This can simply be done by slicing the embeding dimension of attention vectors.

As a language model preducing token sequences, Transformers use label smoothing cross-entropy loss as loss function. However, other loss functions can also be applied based on the task of the model.

Use higher precision data type, like 32 bit floating to calculate the loss value to reduce the unstability.

Traditional machine learning optimization tricks can be applied - like dropout, weight decay, etc. We can also keep track on the loss value, and decrease the learning rate when it is not dropping over past several steps. 

Locality-sensitive hashing (LSH) attention is an algorithm used in Reformer model. It can significantly lower the computational cost / reduce memory usage. In LSH attention, projection weights of key and query is tied. Therefore, we can hash key query vectors of similar tokens into a same bucket, where the similarity can be either spacial or in common characters among tokens. When calculating attention matrix P from Q and K, the model will only need to search the most likely bucket chunk by chunk.


Q2: Can you design a learnable positional encoding method using pytorch? (Create dummy dataset)

Ans:

Please refer to learnable_positional_encoding.py
