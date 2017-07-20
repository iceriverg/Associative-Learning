# Associative-Learning
It is a TF implementation of the paper : "Learning by Association : A versatile semi-supervised training method for neural networks"  (https://arxiv.org/pdf/1706.00909.pdf)

### Learning by association : Core idea

"We feed a batch of labeled and a batch of unlabeled data through a network, producing embeddings for both batches.  Then, an imaginary walker is sent from samples in the labeled batch to sampled in the unlabeled batch.  The transition follows a probability distribution obtained from the similarity of the respective embeddings which we refer to as an _association_"

In other words, given a batch A of labeled data and a batch B of unlabeled data, we first use an arbitrary network to find the embedding of A and B.  For any _a_ in emb(A), we find _b_ in emb(B) "similar to" _a_.  Analogously, we find _a'_ in emb(A) that is "similar to" _b_.  We penalize if class(_a_) and class(_a'_) differ.  The paper likens this concept to an "imaginary walker" from batch emb(A) to emb(B) and back to emb(A), according to the probability distriution obtained from the similarity matrix.

![](https://ai2-s2-public.s3.amazonaws.com/figures/2016-11-08/220ef8fa2f2a3bc148df0769a340124c57d6f11d/0-Figure1-1.png)

The red arrow is the traveling path of an "imaginary walker"



