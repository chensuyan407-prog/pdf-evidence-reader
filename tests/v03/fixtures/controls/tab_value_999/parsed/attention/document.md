<!-- source: C:\Users\12117\Desktop\bohrium-pdf-evidence-reader\v0.3\tests\pdfs\attention.pdf | status: success | parser: pdf-evidence parse_pdf.py 0.3.0 -->


<!-- ===== page 1 ===== -->

<!-- p1-b1 body col=0 bbox=[124.31, 72.9, 487.89, 98.8] -->
Provided proper attribution is provided, Google hereby grants permission to reproduce the tables and figures in this paper solely for use in journalistic or

<!-- p1-b2 body col=0 bbox=[267.59, 100.79, 346.5, 112.75] -->
scholarly works.

<!-- p1-b3 title col=0 bbox=[211.49, 148.83, 399.89, 166.05] -->
# Attention Is All You Need

<!-- T1 table page=1 bbox=[108.0, 176.15, 504.0, 320.17] -->
| Ashish Vaswani^∗ | Noam Shazeer^∗ Niki Parmar^∗ Jakob Uszkoreit^∗ |
|---|---|
| Google Brain | Google Brain Google Research Google Research |
| avaswani@google.com | noam@google.com nikip@google.com usz@google.com |
| Llion Jones^∗ | Aidan N. Gomez^∗ † Łukasz Kaiser^∗ |
| Google Research | University of Toronto Google Brain |
| llion@google.com | aidan@cs.toronto.edu lukaszkaiser@google.com |

<!-- p1-b14 author_affiliation col=0 bbox=[238.02, 333.53, 374.0, 356.25] -->
Illia Polosukhin^{∗‡} illia.polosukhin@gmail.com

<!-- p1-b15 heading col=0 bbox=[283.76, 385.61, 328.24, 397.56] -->
## Abstract

<!-- p1-b16 body col=0 bbox=[143.56, 413.33, 469.79, 576.1] -->
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 Englishto-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.

<!-- p1-b17 body col=0 bbox=[108.0, 597.3, 504.31, 687.72] -->
^{∗}Equal contribution. Listing order is random. Jakob proposed replacing RNNs with self-attention and started the effort to evaluate this idea. Ashish, with Illia, designed and implemented the first Transformer models and has been crucially involved in every aspect of this work. Noam proposed scaled dot-product attention, multi-head attention and the parameter-free position representation and became the other person involved in nearly every detail. Niki designed, implemented, tuned and evaluated countless model variants in our original codebase and tensor2tensor. Llion also experimented with novel model variants, was responsible for our initial codebase, and efficient inference and visualizations. Lukasz and Aidan spent countless long days designing various parts of and implementing tensor2tensor, replacing our earlier codebase, greatly improving results and massively accelerating our research.

<!-- p1-b18 footnote col=0 bbox=[120.21, 687.95, 280.29, 709.62] -->
[^fn]: ^{†}Work performed while at Google Brain. ^{‡}Work performed while at Google Research.

<!-- p1-b19 body col=0 bbox=[108.0, 733.08, 460.14, 742.04] -->
31st Conference on Neural Information Processing Systems (NIPS 2017), Long Beach, CA, USA.


<!-- ===== page 2 ===== -->

<!-- p2-b1 heading col=0 bbox=[108.0, 72.79, 190.81, 84.74] -->
## 1 Introduction

<!-- p2-b2 body col=0 bbox=[108.0, 99.02, 504.17, 152.69] -->
Recurrent neural networks, long short-term memory [13] and gated recurrent [7] neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation [35, 2, 5]. Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures [38, 24, 15].

<!-- p2-b3 body col=0 bbox=[108.0, 159.04, 504.35, 245.44] -->
Recurrent models typically factor computation along the symbol positions of the input and output sequences. Aligning the positions to steps in computation time, they generate a sequence of hidden states h_{t}, as a function of the previous hidden state h_{t}_{−}_{1} and the input for position t. This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples. Recent work has achieved significant improvements in computational efficiency through factorization tricks [21] and conditional computation [32], while also improving model performance in case of the latter. The fundamental constraint of sequential computation, however, remains.

<!-- p2-b4 body col=0 bbox=[107.64, 251.89, 505.66, 294.56] -->
Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences [2, 19]. In all but a few cases [27], however, such attention mechanisms are used in conjunction with a recurrent network.

<!-- p2-b5 body col=0 bbox=[107.69, 300.91, 505.74, 343.67] -->
In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.

<!-- p2-b6 heading col=0 bbox=[108.0, 363.1, 188.83, 375.06] -->
## 2 Background

<!-- p2-b7 body col=0 bbox=[107.69, 389.45, 505.24, 486.64] -->
The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU [16], ByteNet [18] and ConvS2S [9], all of which use convolutional neural networks as basic building block, computing hidden representations in parallel for all input and output positions. In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes it more difficult to learn dependencies between distant positions [12]. In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as described in section 3.2.

<!-- p2-b8 body col=0 bbox=[108.0, 493.12, 505.25, 535.76] -->
Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. Self-attention has been used successfully in a variety of tasks including reading comprehension, abstractive summarization, textual entailment and learning task-independent sentence representations [4, 27, 28, 22].

<!-- p2-b9 body col=0 bbox=[108.0, 542.11, 505.65, 573.96] -->
End-to-end memory networks are based on a recurrent attention mechanism instead of sequencealigned recurrence and have been shown to perform well on simple-language question answering and language modeling tasks [34].

<!-- p2-b10 body col=0 bbox=[107.69, 580.31, 505.66, 623.08] -->
To the best of our knowledge, however, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequencealigned RNNs or convolution. In the following sections, we will describe the Transformer, motivate self-attention and discuss its advantages over models such as [17, 18] and [9].

<!-- p2-b11 heading col=0 bbox=[108.0, 642.5, 226.09, 654.46] -->
## 3 Model Architecture

<!-- p2-b12 body col=0 bbox=[108.0, 668.81, 505.74, 722.41] -->
Most competitive neural sequence transduction models have an encoder-decoder structure [5, 2, 35]. Here, the encoder maps an input sequence of symbol representations (x_{1}, ..., x_{n}) to a sequence of continuous representations z = (z_{1}, ..., z_{n}). Given z, the decoder then generates an output sequence (y_{1}, ..., y_{m}) of symbols one element at a time. At each step the model is auto-regressive [10], consuming the previously generated symbols as additional input when generating the next.


<!-- ===== page 3 ===== -->

<!-- p3-b1 caption col=0 bbox=[210.01, 404.75, 401.99, 414.71] -->
> **Figure 1: The Transformer - model architecture.**

<!-- p3-b2 body col=0 bbox=[107.69, 436.75, 505.25, 468.58] -->
The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1, respectively.

<!-- p3-b3 heading col=0 bbox=[108.0, 482.99, 253.01, 492.95] -->
### 3.1 Encoder and Decoder Stacks

<!-- p3-b4 body col=0 bbox=[107.64, 502.83, 505.66, 579.74] -->
Encoder: The encoder is composed of a stack of N = 6 identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, positionwise fully connected feed-forward network. We employ a residual connection [11] around each of the two sub-layers, followed by layer normalization [1]. That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension d_{model} = 512.

<!-- p3-b5 body col=0 bbox=[108.0, 591.41, 504.0, 667.05] -->
Decoder: The decoder is also composed of a stack of N = 6 identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization. We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position i can depend only on the known outputs at positions less than i.

<!-- p3-b6 heading col=0 bbox=[108.0, 681.47, 170.81, 691.43] -->
### 3.2 Attention

<!-- p3-b7 body col=0 bbox=[107.64, 701.54, 505.25, 722.4] -->
An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum


<!-- ===== page 4 ===== -->

<!-- p4-b1 body col=0 bbox=[147.78, 71.2, 450.2, 81.16] -->
Scaled Dot-Product Attention Multi-Head Attention

<!-- p4-b2 caption col=0 bbox=[108.0, 274.39, 504.0, 295.34] -->
> **Figure 2: (left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention layers running in parallel.**

<!-- p4-b3 body col=0 bbox=[108.0, 317.25, 504.0, 338.05] -->
of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

<!-- p4-b4 heading col=0 bbox=[108.0, 351.06, 263.88, 361.02] -->
#### 3.2.1 Scaled Dot-Product Attention

<!-- p4-b5 body col=0 bbox=[107.53, 369.7, 504.0, 412.47] -->
We call our particular attention "Scaled Dot-Product Attention" (Figure 2). The input consists of queries and keys of dimension d_{k}, and values of dimension d_{v}. We compute the dot products of the query with all keys, divide each by ^{√}d_{k}, and apply a softmax function to obtain the weights on the values.

<!-- p4-b6 body col=0 bbox=[108.0, 418.82, 504.17, 450.67] -->
In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix Q. The keys and values are also packed together into matrices K and V . We compute the matrix of outputs as:

<!-- p4-b7 equation col=0 bbox=[219.97, 464.97, 377.37, 482.95] -->
$$ Attention(Q, K, V ) = softmax(^{QK}^{T} $$  (eq. 1)

<!-- p4-b8 equation col=0 bbox=[358.08, 472.56, 375.8, 491.23] -->
$$ √d_{k} $$  (eq. 1)

<!-- p4-b9 equation col=0 bbox=[380.13, 472.99, 389.81, 482.95] -->
$$ )V $$  (eq. 1)

<!-- p4-b10 equation col=0 bbox=[493.05, 473.22, 504.67, 483.18] -->
$$ (1) $$  (eq. 1)

<!-- p4-b11 body col=0 bbox=[107.69, 499.5, 505.65, 531.31] -->
The two most commonly used attention functions are additive attention [2], and dot-product (multiplicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor of

<!-- p4-b12 body col=0 bbox=[125.6, 519.52, 129.57, 526.49] -->
1

<!-- p4-b13 body col=0 bbox=[108.0, 521.3, 504.0, 566.29] -->
^{√}_{d}_{k} . Additive attention computes the compatibility function using a feed-forward network with a single hidden layer. While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.

<!-- p4-b14 body col=0 bbox=[107.53, 572.48, 504.0, 615.4] -->
While for small values of d_{k} the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of d_{k} [3]. We suspect that for large values of d_{k}, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients ^{4}. To counteract this effect, we scale the dot products by

<!-- p4-b15 footnote col=0 bbox=[440.73, 603.62, 459.46, 620.67] -->
[^fn]: 1 √d_{k} .

<!-- p4-b16 heading col=0 bbox=[108.0, 630.78, 230.59, 640.74] -->
#### 3.2.2 Multi-Head Attention

<!-- p4-b17 body col=0 bbox=[107.64, 649.27, 505.24, 692.8] -->
Instead of performing a single attention function with d_{model}-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values h times with different, learned linear projections to d_{k}, d_{k} and d_{v} dimensions, respectively. On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding d_{v}-dimensional

<!-- p4-b18 footnote col=0 bbox=[107.78, 700.31, 504.0, 728.38] -->
[^fn]: ^{4}To illustrate why the dot products get large, assume that the components of q and k are independent random variables with mean 0 and variance 1. Then their dot product, q · k = ^{�}^{d}^{k}

<!-- p4-b19 body col=0 bbox=[368.58, 712.99, 503.88, 724.7] -->
_{i}_{=1} ^{q}^{i}^{k}^{i}^{, has mean}^{0}^{and variance}^{d}^{k}^{.}


<!-- ===== page 5 ===== -->

<!-- p5-b1 body col=0 bbox=[108.0, 74.33, 504.0, 95.28] -->
output values. These are concatenated and once again projected, resulting in the final values, as depicted in Figure 2.

<!-- p5-b2 body col=0 bbox=[108.0, 101.69, 504.0, 122.58] -->
Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.

<!-- p5-b3 body col=0 bbox=[186.94, 145.82, 410.92, 158.4] -->
MultiHead(Q, K, V ) = Concat(head_{1}, ..., head_{h})W ^{O}

<!-- p5-b4 body col=0 bbox=[224.5, 161.96, 358.45, 175.86] -->
where head_{i} = Attention(QW ^{Q}

<!-- p5-b5 body col=0 bbox=[350.79, 162.63, 390.03, 177.15] -->
_{i} ^{, KW}^{K}

<!-- p5-b6 body col=0 bbox=[381.96, 162.63, 418.97, 176.83] -->
_{i} ^{, V W}^{V}

<!-- p5-b7 body col=0 bbox=[412.88, 164.41, 425.06, 176.83] -->
_{i} ^{)}

<!-- p5-b8 body col=0 bbox=[107.53, 203.07, 302.96, 215.69] -->
Where the projections are parameter matrices W ^{Q}

<!-- p5-b9 body col=0 bbox=[306.23, 203.35, 377.32, 215.69] -->
∈R^{d}^{model}^{×}^{d}^{k}, W ^{K}

<!-- p5-b10 body col=0 bbox=[381.07, 203.35, 450.17, 215.69] -->
∈R^{d}^{model}^{×}^{d}^{k}, W ^{V}

<!-- p5-b11 body col=0 bbox=[455.15, 203.35, 502.82, 215.36] -->
∈R^{d}^{model}^{×}^{d}^{v}

<!-- p5-b12 body col=0 bbox=[369.24, 210.45, 372.06, 217.42] -->
i

<!-- p5-b13 body col=0 bbox=[444.08, 210.45, 446.9, 217.42] -->
i

<!-- p5-b14 body col=0 bbox=[295.3, 210.63, 298.12, 217.61] -->
i

<!-- p5-b15 body col=0 bbox=[108.0, 214.01, 201.21, 227.95] -->
and W ^{O} ∈R^{hd}^{v}^{×}^{d}^{model}.

<!-- p5-b16 body col=0 bbox=[108.0, 234.14, 504.0, 266.15] -->
In this work we employ h = 8 parallel attention layers, or heads. For each of these we use d_{k} = d_{v} = d_{model}/h = 64. Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality.

<!-- p5-b17 heading col=0 bbox=[108.0, 280.27, 302.86, 290.24] -->
#### 3.2.3 Applications of Attention in our Model

<!-- p5-b18 body col=0 bbox=[107.69, 299.42, 372.61, 309.38] -->
The Transformer uses multi-head attention in three different ways:

<!-- p5-b19 body col=0 bbox=[135.4, 320.19, 505.24, 373.86] -->
• In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence. This mimics the typical encoder-decoder attention mechanisms in sequence-to-sequence models such as [38, 2, 9].

<!-- p5-b20 body col=0 bbox=[135.4, 379.66, 504.0, 422.42] -->
• The encoder contains self-attention layers. In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder. Each position in the encoder can attend to all positions in the previous layer of the encoder.

<!-- p5-b21 body col=0 bbox=[135.4, 428.3, 504.0, 481.9] -->
• Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position. We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot-product attention by masking out (setting to −∞) all values in the input of the softmax which correspond to illegal connections. See Figure 2.

<!-- p5-b22 heading col=0 bbox=[108.0, 497.36, 292.87, 507.32] -->
### 3.3 Position-wise Feed-Forward Networks

<!-- p5-b23 body col=0 bbox=[108.0, 517.78, 504.34, 549.63] -->
In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between.

<!-- p5-b24 equation col=0 bbox=[226.9, 568.15, 384.61, 579.6] -->
$$ FFN(x) = max(0, xW_{1} + b_{1})W_{2} + b_{2} $$  (eq. 2)

<!-- p5-b25 equation col=0 bbox=[493.05, 568.38, 504.67, 578.34] -->
$$ (2) $$  (eq. 2)

<!-- p5-b26 body col=0 bbox=[107.53, 590.24, 505.75, 634.12] -->
While the linear transformations are the same across different positions, they use different parameters from layer to layer. Another way of describing this is as two convolutions with kernel size 1. The dimensionality of input and output is d_{model} = 512, and the inner-layer has dimensionality d_{ff} = 2048.

<!-- p5-b27 heading col=0 bbox=[108.0, 648.32, 240.02, 658.28] -->
### 3.4 Embeddings and Softmax

<!-- p5-b28 body col=0 bbox=[108.0, 668.73, 505.74, 723.18] -->
Similarly to other sequence transduction models, we use learned embeddings to convert the input tokens and output tokens to vectors of dimension d_{model}. We also use the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities. In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation, similar to [30]. In the embedding layers, we multiply those weights by ^{√}d_{model}.


<!-- ===== page 6 ===== -->

<!-- p6-b1 caption col=0 bbox=[107.69, 71.28, 504.0, 102.98] -->
> **Table 1: Maximum path lengths, per-layer complexity and minimum number of sequential operations for different layer types. n is the sequence length, d is the representation dimension, k is the kernel size of convolutions and r the size of the neighborhood in restricted self-attention.**

<!-- T2 table page=6 bbox=[118.57, 111.24, 493.43, 188.88] -->
| Layer Type | Complexity | per Layer | Sequential | Maximum Path Length |
|---|---|---|---|---|
|  |  |  | Operations |  |
| Self-Attention | O(n^2 · | d) | O(1) | O(1) |
| Recurrent | O(n · d^2 | ) | O(n) | O(n) |
| Convolutional | O(k · n · | d^2 ) | O(1) | O(log k (n)) |
| Self-Attention (restricted) | O(r · n · | d) | O(1) | O(n/r) |

<!-- p6-b3 heading col=0 bbox=[108.0, 213.79, 215.2, 223.75] -->
### 3.5 Positional Encoding

<!-- p6-b4 body col=0 bbox=[108.0, 233.93, 505.25, 298.36] -->
Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence. To this end, we add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks. The positional encodings have the same dimension d_{model} as the embeddings, so that the two can be summed. There are many choices of positional encodings, learned and fixed [9].

<!-- p6-b5 body col=0 bbox=[108.0, 304.79, 389.86, 314.75] -->
In this work, we use sine and cosine functions of different frequencies:

<!-- p6-b6 body col=0 bbox=[235.51, 336.17, 386.31, 349.71] -->
PE_{(}_{pos,}_{2}_{i}_{)} = sin(pos/10000^{2}^{i/d}^{model})

<!-- p6-b7 body col=0 bbox=[225.69, 353.04, 386.31, 366.58] -->
PE_{(}_{pos,}_{2}_{i}_{+1)} = cos(pos/10000^{2}^{i/d}^{model})

<!-- p6-b8 body col=0 bbox=[107.64, 377.29, 504.34, 431.73] -->
where pos is the position and i is the dimension. That is, each dimension of the positional encoding corresponds to a sinusoid. The wavelengths form a geometric progression from 2π to 10000 · 2π. We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset k, PE_{pos}_{+}_{k} can be represented as a linear function of PE_{pos}.

<!-- p6-b9 body col=0 bbox=[107.53, 437.54, 504.0, 480.24] -->
We also experimented with using learned positional embeddings [9] instead, and found that the two versions produced nearly identical results (see Table 3 row (E)). We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training.

<!-- p6-b10 heading col=0 bbox=[108.0, 496.89, 225.04, 508.84] -->
## 4 Why Self-Attention

<!-- p6-b11 body col=0 bbox=[106.83, 521.39, 505.65, 554.38] -->
In this section we compare various aspects of self-attention layers to the recurrent and convolutional layers commonly used for mapping one variable-length sequence of symbol representations (x_{1}, ..., x_{n}) to another sequence of equal length (z_{1}, ..., z_{n}), with x_{i}, z_{i} ∈R^{d}, such as a hidden

<!-- p6-b12 body col=0 bbox=[108.0, 554.2, 504.0, 575.06] -->
layer in a typical sequence transduction encoder or decoder. Motivating our use of self-attention we consider three desiderata.

<!-- p6-b13 body col=0 bbox=[108.0, 581.48, 504.0, 602.36] -->
One is the total computational complexity per layer. Another is the amount of computation that can be parallelized, as measured by the minimum number of sequential operations required.

<!-- p6-b14 body col=0 bbox=[107.69, 608.74, 504.0, 684.2] -->
The third is the path length between long-range dependencies in the network. Learning long-range dependencies is a key challenge in many sequence transduction tasks. One key factor affecting the ability to learn such dependencies is the length of the paths forward and backward signals have to traverse in the network. The shorter these paths between any combination of positions in the input and output sequences, the easier it is to learn long-range dependencies [12]. Hence we also compare the maximum path length between any two input and output positions in networks composed of the different layer types.

<!-- p6-b15 body col=0 bbox=[107.64, 690.7, 504.35, 722.43] -->
As noted in Table 1, a self-attention layer connects all positions with a constant number of sequentially executed operations, whereas a recurrent layer requires O(n) sequential operations. In terms of computational complexity, self-attention layers are faster than recurrent layers when the sequence


<!-- ===== page 7 ===== -->

<!-- p7-b1 body col=0 bbox=[107.75, 74.18, 504.0, 138.92] -->
length n is smaller than the representation dimensionality d, which is most often the case with sentence representations used by state-of-the-art models in machine translations, such as word-piece [38] and byte-pair [31] representations. To improve computational performance for tasks involving very long sequences, self-attention could be restricted to considering only a neighborhood of size r in the input sequence centered around the respective output position. This would increase the maximum path length to O(n/r). We plan to investigate this approach further in future work.

<!-- p7-b2 body col=0 bbox=[107.64, 145.11, 505.25, 231.67] -->
A single convolutional layer with kernel width k < n does not connect all pairs of input and output positions. Doing so requires a stack of O(n/k) convolutional layers in the case of contiguous kernels, or O(log_{k}(n)) in the case of dilated convolutions [18], increasing the length of the longest paths between any two positions in the network. Convolutional layers are generally more expensive than recurrent layers, by a factor of k. Separable convolutions [6], however, decrease the complexity considerably, to O(k · n · d + n · d^{2}). Even with k = n, however, the complexity of a separable convolution is equal to the combination of a self-attention layer and a point-wise feed-forward layer, the approach we take in our model.

<!-- p7-b3 body col=0 bbox=[107.64, 238.17, 504.0, 280.78] -->
As side benefit, self-attention could yield more interpretable models. We inspect attention distributions from our models and present and discuss examples in the appendix. Not only do individual attention heads clearly learn to perform different tasks, many appear to exhibit behavior related to the syntactic and semantic structure of the sentences.

<!-- p7-b4 heading col=0 bbox=[108.0, 299.91, 170.23, 311.87] -->
## 5 Training

<!-- p7-b5 body col=0 bbox=[107.69, 326.04, 337.48, 336.0] -->
This section describes the training regime for our models.

<!-- p7-b6 heading col=0 bbox=[108.0, 352.73, 249.53, 362.69] -->
### 5.1 Training Data and Batching

<!-- p7-b7 body col=0 bbox=[107.53, 373.65, 505.65, 449.14] -->
We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs. Sentences were encoded using byte-pair encoding [3], which has a shared sourcetarget vocabulary of about 37000 tokens. For English-French, we used the significantly larger WMT 2014 English-French dataset consisting of 36M sentences and split tokens into a 32000 word-piece vocabulary [38]. Sentence pairs were batched together by approximate sequence length. Each training batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000 target tokens.

<!-- p7-b8 heading col=0 bbox=[108.0, 465.86, 233.04, 475.83] -->
### 5.2 Hardware and Schedule

<!-- p7-b9 body col=0 bbox=[107.53, 486.78, 504.0, 540.46] -->
We trained our models on one machine with 8 NVIDIA P100 GPUs. For our base models using the hyperparameters described throughout the paper, each training step took about 0.4 seconds. We trained the base models for a total of 100,000 steps or 12 hours. For our big models,(described on the bottom line of table 3), step time was 1.0 seconds. The big models were trained for 300,000 steps (3.5 days).

<!-- p7-b10 heading col=0 bbox=[108.0, 557.18, 174.13, 567.15] -->
### 5.3 Optimizer

<!-- p7-b11 body col=0 bbox=[107.53, 576.58, 504.0, 599.05] -->
We used the Adam optimizer [20] with β_{1} = 0.9, β_{2} = 0.98 and ϵ = 10^{−}^{9}. We varied the learning rate over the course of training, according to the formula:

<!-- p7-b12 equation col=0 bbox=[162.89, 617.23, 219.34, 629.21] -->
$$ lrate = d^{−}^{0}^{.}^{5} $$  (eq. 3)

<!-- p7-b13 equation col=0 bbox=[202.8, 617.38, 449.11, 631.97] -->
$$ _{model} ^{·}^{min(}^{step}^{_}^{num}^{−}^{0}^{.}^{5}^{, step}^{_}^{num}^{·}^{warmup}^{_}^{steps}^{−}^{1}^{.}^{5}^{)} $$  (eq. 3)

<!-- p7-b14 equation col=0 bbox=[493.05, 619.48, 504.67, 629.44] -->
$$ (3) $$  (eq. 3)

<!-- p7-b15 body col=0 bbox=[107.69, 642.71, 505.24, 674.73] -->
This corresponds to increasing the learning rate linearly for the first warmup_steps training steps, and decreasing it thereafter proportionally to the inverse square root of the step number. We used warmup_steps = 4000.

<!-- p7-b16 heading col=0 bbox=[108.0, 691.45, 193.51, 701.41] -->
### 5.4 Regularization

<!-- p7-b17 body col=0 bbox=[107.53, 712.44, 331.99, 722.41] -->
We employ three types of regularization during training:


<!-- ===== page 8 ===== -->

<!-- p8-b1 caption col=0 bbox=[107.69, 71.2, 504.0, 92.07] -->
> **Table 2: The Transformer achieves better BLEU scores than previous state-of-the-art models on the English-to-German and English-to-French newstest2014 tests at a fraction of the training cost.**

<!-- T3 table page=8 bbox=[130.69, 92.52, 481.31, 245.61] -->
|  | BLEU |  | Training Cost (FLOPs) |  |
|---|---|---|---|---|
| Model |  |  |  |  |
|  | EN-DE | EN-FR | EN-DE | EN-FR |
| ByteNet [18] | 23.75 |  |  |  |
| Deep-Att + PosUnk [39] |  | 39.2 |  | 1.0 · 10^20 |
| GNMT + RL [38] | 24.6 | 39.92 | 2.3 · 10^19 | 1.4 · 10^20 |
| ConvS2S [9] | 25.16 | 40.46 | 9.6 · 10^18 | 1.5 · 10^20 |
| MoE [32] | 26.03 | 40.56 | 2.0 · 10^19 | 1.2 · 10^20 |
| Deep-Att + PosUnk Ensemble [39] |  | 40.4 |  | 8.0 · 10^20 |
| GNMT + RL Ensemble [38] | 26.30 | 41.16 | 1.8 · 10^20 | 1.1 · 10^21 |
| ConvS2S Ensemble [9] | 26.36 | 41.29 | 7.7 · 10^19 | 1.2 · 10^21 |
| Transformer (base model) | 27.3 | 38.1 | 3.3 · 10^18 |  |
| Transformer (big) | 28.4 | 41.8 | 2.3 · 10^19 |  |

<!-- p8-b13 body col=0 bbox=[108.0, 273.23, 504.0, 317.25] -->
Residual Dropout We apply dropout [33] to the output of each sub-layer, before it is added to the sub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks. For the base model, we use a rate of P_{drop} = 0.1.

<!-- p8-b14 body col=0 bbox=[108.0, 331.14, 504.0, 352.24] -->
Label Smoothing During training, we employed label smoothing of value ϵ_{ls} = 0.1 [36]. This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score.

<!-- p8-b15 heading col=0 bbox=[108.0, 371.29, 163.13, 383.24] -->
## 6 Results

<!-- p8-b16 heading col=0 bbox=[108.0, 397.27, 219.07, 407.23] -->
### 6.1 Machine Translation

<!-- p8-b17 body col=0 bbox=[108.0, 418.3, 504.67, 482.73] -->
On the WMT 2014 English-to-German translation task, the big transformer model (Transformer (big) in Table 2) outperforms the best previously reported models (including ensembles) by more than 2.0 BLEU, establishing a new state-of-the-art BLEU score of 28.4. The configuration of this model is listed in the bottom line of Table 3. Training took 3.5 days on 8 P100 GPUs. Even our base model surpasses all previously published models and ensembles, at a fraction of the training cost of any of the competitive models.

<!-- p8-b18 body col=0 bbox=[108.0, 488.93, 505.25, 533.11] -->
On the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of 41.0, outperforming all of the previously published single models, at less than 1/4 the training cost of the previous state-of-the-art model. The Transformer (big) model trained for English-to-French used dropout rate P_{drop} = 0.1, instead of 0.3.

<!-- p8-b19 body col=0 bbox=[107.64, 538.2, 504.0, 591.87] -->
For the base models, we used a single model obtained by averaging the last 5 checkpoints, which were written at 10-minute intervals. For the big models, we averaged the last 20 checkpoints. We used beam search with a beam size of 4 and length penalty α = 0.6 [38]. These hyperparameters were chosen after experimentation on the development set. We set the maximum output length during inference to input length + 50, but terminate early when possible [38].

<!-- p8-b20 body col=0 bbox=[107.69, 598.38, 504.0, 640.99] -->
Table 2 summarizes our results and compares our translation quality and training costs to other model architectures from the literature. We estimate the number of floating point operations used to train a model by multiplying the training time, the number of GPUs used, and an estimate of the sustained single-precision floating-point capacity of each GPU ^{5}.

<!-- p8-b21 heading col=0 bbox=[108.0, 657.63, 203.94, 667.59] -->
### 6.2 Model Variations

<!-- p8-b22 body col=0 bbox=[107.69, 678.51, 504.0, 699.48] -->
To evaluate the importance of different components of the Transformer, we varied our base model in different ways, measuring the change in performance on English-to-German translation on the

<!-- p8-b23 footnote col=0 bbox=[120.65, 711.66, 453.56, 722.17] -->
[^fn]: ^{5}We used values of 2.8, 3.7, 6.0 and 9.5 TFLOPS for K80, K40, M40 and P100, respectively.


<!-- ===== page 9 ===== -->

<!-- p9-b1 caption col=0 bbox=[107.69, 71.26, 504.0, 113.89] -->
> **Table 3: Variations on the Transformer architecture. Unlisted values are identical to those of the base model. All metrics are on the English-to-German translation development set, newstest2013. Listed perplexities are per-wordpiece, according to our byte-pair encoding, and should not be compared to per-word perplexities.**

<!-- T4 table page=9 bbox=[108.0, 127.87, 508.74, 386.17] -->
|  | train | PPL BLEU params |
|---|---|---|
|  | N dmodel dff h d k d v Pdrop ϵls |  |
|  | steps | (dev) (dev) ×10^6 |
| base | 6 512 2048 8 64 64 0.1 0.1 100K | 4.92 25.8 65 |
|  | 1 512 512 | 5.29 24.9 |
|  | 4 128 128 | 5.00 25.5 |
| (A) |  |  |
|  | 16 32 32 | 4.91 25.8 |
|  | 32 16 16 | 5.01 25.4 |
|  | 16 | 5.16 25.1 58 |
| (B) |  |  |
|  | 32 | 5.01 25.4 60 |
|  | 2 | 6.11 23.7 36 |
|  | 4 | 5.19 25.3 50 |
|  | 8 | 4.88 25.5 80 |
| (C) | 256 32 32 | 5.75 24.5 28 |
|  | 1024 128 128 | 4.66 26.0 168 |
|  | 1024 | 5.12 25.4 53 |
|  | 4096 | 4.75 26.2 90 |
|  | 0.0 | 5.77 24.6 |
|  | 0.2 | 4.95 25.5 |
| (D) |  |  |
|  | 0.0 | 4.67 25.3 |
|  | 0.2 | 5.47 25.7 |
| (E) | positional embedding instead of sinusoids | 4.92 25.7 |
| big | 6 1024 4096 16 0.3 300K | 4.33 26.4 213 |

<!-- p9-b15 body col=0 bbox=[108.0, 413.68, 504.0, 434.62] -->
development set, newstest2013. We used beam search as described in the previous section, but no checkpoint averaging. We present these results in Table 3.

<!-- p9-b16 body col=0 bbox=[108.0, 441.12, 505.24, 472.83] -->
In Table 3 rows (A), we vary the number of attention heads and the attention key and value dimensions, keeping the amount of computation constant, as described in Section 3.2.2. While single-head attention is 0.9 BLEU worse than the best setting, quality also drops off with too many heads.

<!-- p9-b17 body col=0 bbox=[108.0, 479.02, 505.24, 543.76] -->
In Table 3 rows (B), we observe that reducing the attention key size d_{k} hurts model quality. This suggests that determining compatibility is not easy and that a more sophisticated compatibility function than dot product may be beneficial. We further observe in rows (C) and (D) that, as expected, bigger models are better, and dropout is very helpful in avoiding over-fitting. In row (E) we replace our sinusoidal positional encoding with learned positional embeddings [9], and observe nearly identical results to the base model.

<!-- p9-b18 heading col=0 bbox=[108.0, 560.49, 255.97, 570.45] -->
### 6.3 English Constituency Parsing

<!-- p9-b19 body col=0 bbox=[107.69, 581.43, 504.0, 624.18] -->
To evaluate if the Transformer can generalize to other tasks we performed experiments on English constituency parsing. This task presents specific challenges: the output is subject to strong structural constraints and is significantly longer than the input. Furthermore, RNN sequence-to-sequence models have not been able to attain state-of-the-art results in small-data regimes [37].

<!-- p9-b20 body col=0 bbox=[107.53, 630.37, 505.24, 684.2] -->
We trained a 4-layer transformer with d_{model} = 1024 on the Wall Street Journal (WSJ) portion of the Penn Treebank [25], about 40K training sentences. We also trained it in a semi-supervised setting, using the larger high-confidence and BerkleyParser corpora from with approximately 17M sentences [37]. We used a vocabulary of 16K tokens for the WSJ only setting and a vocabulary of 32K tokens for the semi-supervised setting.

<!-- p9-b21 body col=0 bbox=[107.53, 690.64, 504.0, 722.43] -->
We performed only a small number of experiments to select the dropout, both attention and residual (section 5.4), learning rates and beam size on the Section 22 development set, all other parameters remained unchanged from the English-to-German base translation model. During inference, we


<!-- ===== page 10 ===== -->

<!-- p10-b1 caption col=0 bbox=[107.69, 71.28, 504.0, 92.07] -->
> **Table 4: The Transformer generalizes well to English constituency parsing (Results are on Section 23 of WSJ)**

<!-- T5 table page=10 bbox=[145.05, 91.95, 466.95, 238.56] -->
| Parser | Training | WSJ 23 F1 |
|---|---|---|
| Vinyals & Kaiser el al. (2014) [37] | WSJ only, discriminative | 88.3 |
| Petrov et al. (2006) [29] | WSJ only, discriminative | 90.4 |
| Zhu et al. (2013) [40] | WSJ only, discriminative | 90.4 |
| Dyer et al. (2016) [8] | WSJ only, discriminative | 91.7 |
| Transformer (4 layers) | WSJ only, discriminative | 91.3 |
| Zhu et al. (2013) [40] | semi-supervised | 91.3 |
| Huang & Harper (2009) [14] | semi-supervised | 91.3 |
| McClosky et al. (2006) [26] | semi-supervised | 92.1 |
| Vinyals & Kaiser el al. (2014) [37] | semi-supervised | 92.1 |
| Transformer (4 layers) | semi-supervised | 92.7 |
| Luong et al. (2015) [23] | multi-task | 93.0 |
| Dyer et al. (2016) [8] | generative | 93.3 |

<!-- p10-b3 body col=0 bbox=[108.0, 261.85, 504.25, 282.95] -->
increased the maximum output length to input length + 300. We used a beam size of 21 and α = 0.3 for both WSJ only and the semi-supervised setting.

<!-- p10-b4 body col=0 bbox=[108.0, 289.31, 505.65, 321.16] -->
Our results in Table 4 show that despite the lack of task-specific tuning our model performs surprisingly well, yielding better results than all previously reported models with the exception of the Recurrent Neural Network Grammar [8].

<!-- p10-b5 body col=0 bbox=[108.0, 327.53, 505.66, 348.46] -->
In contrast to RNN sequence-to-sequence models [37], the Transformer outperforms the Berkeley- Parser [29] even when training only on the WSJ training set of 40K sentences.

<!-- p10-b6 heading col=0 bbox=[108.0, 364.82, 183.07, 376.78] -->
## 7 Conclusion

<!-- p10-b7 body col=0 bbox=[108.0, 389.1, 504.0, 420.89] -->
In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention.

<!-- p10-b8 body col=0 bbox=[108.0, 427.24, 504.0, 470.0] -->
For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers. On both WMT 2014 English-to-German and WMT 2014 English-to-French translation tasks, we achieve a new state of the art. In the former task our best model outperforms even all previously reported ensembles.

<!-- p10-b9 body col=0 bbox=[107.53, 476.44, 505.61, 519.1] -->
We are excited about the future of attention-based models and plan to apply them to other tasks. We plan to extend the Transformer to problems involving input and output modalities other than text and to investigate local, restricted attention mechanisms to efficiently handle large inputs and outputs such as images, audio and video. Making generation less sequential is another research goals of ours.

<!-- p10-b10 body col=0 bbox=[107.69, 525.47, 505.07, 546.66] -->
The code we used to train and evaluate our models is available at https://github.com/ tensorflow/tensor2tensor.

<!-- p10-b11 heading col=0 bbox=[108.0, 558.95, 192.01, 569.01] -->
## Acknowledgements

<!-- p10-b12 body col=0 bbox=[108.0, 559.04, 504.0, 579.99] -->
We are grateful to Nal Kalchbrenner and Stephan Gouws for their fruitful comments, corrections and inspiration.

<!-- p10-b13 heading col=0 bbox=[108.0, 596.35, 163.54, 608.31] -->
## References

<!-- p10-b14 reference col=0 bbox=[112.98, 615.04, 504.0, 625.12] -->
- [1] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint

<!-- p10-b15 reference col=0 bbox=[129.58, 625.89, 229.75, 636.03] -->
- arXiv:1607.06450, 2016.

<!-- p10-b16 reference col=0 bbox=[112.98, 643.95, 504.35, 653.91] -->
- [2] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly

<!-- p10-b17 reference col=0 bbox=[129.58, 654.68, 368.65, 664.82] -->
- learning to align and translate. CoRR, abs/1409.0473, 2014.

<!-- p10-b18 reference col=0 bbox=[112.98, 672.74, 504.0, 682.71] -->
- [3] Denny Britz, Anna Goldie, Minh-Thang Luong, and Quoc V. Le. Massive exploration of neural

<!-- p10-b19 reference col=0 bbox=[129.58, 683.48, 386.9, 693.62] -->
- machine translation architectures. CoRR, abs/1703.03906, 2017.

<!-- p10-b20 reference col=0 bbox=[112.98, 701.54, 504.0, 711.5] -->
- [4] Jianpeng Cheng, Li Dong, and Mirella Lapata. Long short-term memory-networks for machine

<!-- p10-b21 reference col=0 bbox=[129.58, 712.27, 325.11, 722.41] -->
- reading. arXiv preprint arXiv:1601.06733, 2016.


<!-- ===== page 11 ===== -->

<!-- p11-b1 reference col=0 bbox=[112.98, 74.33, 505.24, 84.39] -->
- [5] Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Fethi Bougares, Holger Schwenk,

<!-- p11-b2 reference col=0 bbox=[129.58, 85.31, 504.0, 106.19] -->
- and Yoshua Bengio. Learning phrase representations using rnn encoder-decoder for statistical machine translation. CoRR, abs/1406.1078, 2014.

<!-- p11-b3 reference col=0 bbox=[112.98, 115.77, 504.0, 126.01] -->
- [6] Francois Chollet. Xception: Deep learning with depthwise separable convolutions. arXiv

<!-- p11-b4 reference col=0 bbox=[129.58, 126.76, 264.53, 136.9] -->
- preprint arXiv:1610.02357, 2016.

<!-- p11-b5 reference col=0 bbox=[112.98, 146.74, 504.0, 156.7] -->
- [7] Junyoung Chung, Çaglar Gülçehre, Kyunghyun Cho, and Yoshua Bengio. Empirical evaluation

<!-- p11-b6 reference col=0 bbox=[129.58, 157.47, 478.88, 167.61] -->
- of gated recurrent neural networks on sequence modeling. CoRR, abs/1412.3555, 2014.

<!-- p11-b7 reference col=0 bbox=[112.98, 177.37, 504.0, 187.43] -->
- [8] Chris Dyer, Adhiguna Kuncoro, Miguel Ballesteros, and Noah A. Smith. Recurrent neural

<!-- p11-b8 reference col=0 bbox=[129.58, 188.18, 313.3, 198.32] -->
- network grammars. In Proc. of NAACL, 2016.

<!-- p11-b9 reference col=0 bbox=[112.98, 208.11, 505.65, 218.13] -->
- [9] Jonas Gehring, Michael Auli, David Grangier, Denis Yarats, and Yann N. Dauphin. Convolu-

<!-- p11-b10 reference col=0 bbox=[129.58, 218.89, 450.72, 229.03] -->
- tional sequence to sequence learning. arXiv preprint arXiv:1705.03122v2, 2017.

<!-- p11-b11 reference col=0 bbox=[108.0, 238.61, 504.0, 259.74] -->
- [10] Alex Graves. Generating sequences with recurrent neural networks. arXiv preprint arXiv:1308.0850, 2013.

<!-- p11-b12 reference col=0 bbox=[108.0, 269.5, 505.65, 279.56] -->
- [11] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for im-

<!-- p11-b13 reference col=0 bbox=[129.27, 280.23, 504.0, 301.36] -->
- age recognition. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 770–778, 2016.

<!-- p11-b14 reference col=0 bbox=[108.0, 311.14, 504.0, 321.18] -->
- [12] Sepp Hochreiter, Yoshua Bengio, Paolo Frasconi, and Jürgen Schmidhuber. Gradient flow in

<!-- p11-b15 reference col=0 bbox=[129.58, 322.11, 412.15, 332.07] -->
- recurrent nets: the difficulty of learning long-term dependencies, 2001.

<!-- p11-b16 reference col=0 bbox=[108.0, 341.65, 505.25, 351.89] -->
- [13] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation,

<!-- p11-b17 reference col=0 bbox=[129.58, 352.82, 221.18, 362.78] -->
- 9(8):1735–1780, 1997.

<!-- p11-b18 reference col=0 bbox=[108.0, 372.54, 504.0, 382.6] -->
- [14] Zhongqiang Huang and Mary Harper. Self-training PCFG grammars with latent annotations

<!-- p11-b19 reference col=0 bbox=[129.3, 383.27, 504.0, 404.4] -->
- across languages. In Proceedings of the 2009 Conference on Empirical Methods in Natural Language Processing, pages 832–841. ACL, August 2009.

<!-- p11-b20 reference col=0 bbox=[108.0, 414.19, 504.0, 424.21] -->
- [15] Rafal Jozefowicz, Oriol Vinyals, Mike Schuster, Noam Shazeer, and Yonghui Wu. Exploring

<!-- p11-b21 reference col=0 bbox=[129.58, 424.97, 421.98, 435.11] -->
- the limits of language modeling. arXiv preprint arXiv:1602.02410, 2016.

<!-- p11-b22 reference col=0 bbox=[108.0, 444.84, 504.0, 454.91] -->
- [16] Łukasz Kaiser and Samy Bengio. Can active memory replace attention? In Advances in Neural

<!-- p11-b23 reference col=0 bbox=[129.41, 455.68, 317.4, 465.82] -->
- Information Processing Systems, (NIPS), 2016.

<!-- p11-b24 reference col=0 bbox=[108.0, 475.51, 504.0, 485.62] -->
- [17] Łukasz Kaiser and Ilya Sutskever. Neural GPUs learn algorithms. In International Conference

<!-- p11-b25 reference col=0 bbox=[129.58, 486.39, 303.0, 496.53] -->
- on Learning Representations (ICLR), 2016.

<!-- p11-b26 reference col=0 bbox=[108.0, 506.37, 505.65, 516.33] -->
- [18] Nal Kalchbrenner, Lasse Espeholt, Karen Simonyan, Aaron van den Oord, Alex Graves, and Ko-

<!-- p11-b27 reference col=0 bbox=[129.58, 517.18, 505.24, 538.15] -->
- ray Kavukcuoglu. Neural machine translation in linear time. arXiv preprint arXiv:1610.10099v2, 2017.

<!-- p11-b28 reference col=0 bbox=[108.0, 547.98, 505.75, 557.95] -->
- [19] Yoon Kim, Carl Denton, Luong Hoang, and Alexander M. Rush. Structured attention networks.

<!-- p11-b29 reference col=0 bbox=[129.58, 558.72, 386.19, 568.86] -->
- In International Conference on Learning Representations, 2017.

<!-- p11-b30 reference col=0 bbox=[108.0, 578.56, 505.75, 588.66] -->
- [20] Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2015.

<!-- p11-b31 reference col=0 bbox=[108.0, 598.35, 504.0, 608.46] -->
- [21] Oleksii Kuchaiev and Boris Ginsburg. Factorization tricks for LSTM networks. arXiv preprint

<!-- p11-b32 reference col=0 bbox=[129.58, 609.23, 229.75, 619.37] -->
- arXiv:1703.10722, 2017.

<!-- p11-b33 reference col=0 bbox=[108.0, 629.13, 504.0, 639.19] -->
- [22] Zhouhan Lin, Minwei Feng, Cicero Nogueira dos Santos, Mo Yu, Bing Xiang, Bowen

<!-- p11-b34 reference col=0 bbox=[129.58, 639.86, 504.0, 660.99] -->
- Zhou, and Yoshua Bengio. A structured self-attentive sentence embedding. arXiv preprint arXiv:1703.03130, 2017.

<!-- p11-b35 reference col=0 bbox=[108.0, 670.83, 504.25, 680.79] -->
- [23] Minh-Thang Luong, Quoc V. Le, Ilya Sutskever, Oriol Vinyals, and Lukasz Kaiser. Multi-task

<!-- p11-b36 reference col=0 bbox=[129.58, 681.56, 416.13, 691.7] -->
- sequence to sequence learning. arXiv preprint arXiv:1511.06114, 2015.

<!-- p11-b37 reference col=0 bbox=[108.0, 701.54, 505.65, 711.5] -->
- [24] Minh-Thang Luong, Hieu Pham, and Christopher D Manning. Effective approaches to attention-

<!-- p11-b38 reference col=0 bbox=[129.58, 712.27, 426.1, 722.41] -->
- based neural machine translation. arXiv preprint arXiv:1508.04025, 2015.


<!-- ===== page 12 ===== -->

<!-- p12-b1 reference col=0 bbox=[108.0, 74.41, 504.0, 84.37] -->
- [25] Mitchell P Marcus, Mary Ann Marcinkiewicz, and Beatrice Santorini. Building a large annotated

<!-- p12-b2 reference col=0 bbox=[129.58, 85.14, 479.12, 95.28] -->
- corpus of english: The penn treebank. Computational linguistics, 19(2):313–330, 1993.

<!-- p12-b3 reference col=0 bbox=[108.0, 108.79, 504.0, 118.77] -->
- [26] David McClosky, Eugene Charniak, and Mark Johnson. Effective self-training for parsing. In

<!-- p12-b4 reference col=0 bbox=[129.27, 119.56, 505.25, 140.58] -->
- Proceedings of the Human Language Technology Conference of the NAACL, Main Conference, pages 152–159. ACL, June 2006.

<!-- p12-b5 reference col=0 bbox=[108.0, 154.1, 504.0, 164.06] -->
- [27] Ankur Parikh, Oscar Täckström, Dipanjan Das, and Jakob Uszkoreit. A decomposable attention

<!-- p12-b6 reference col=0 bbox=[129.58, 164.83, 407.45, 174.97] -->
- model. In Empirical Methods in Natural Language Processing, 2016.

<!-- p12-b7 reference col=0 bbox=[108.0, 188.49, 504.0, 198.46] -->
- [28] Romain Paulus, Caiming Xiong, and Richard Socher. A deep reinforced model for abstractive

<!-- p12-b8 reference col=0 bbox=[129.58, 199.22, 354.45, 209.36] -->
- summarization. arXiv preprint arXiv:1705.04304, 2017.

<!-- p12-b9 reference col=0 bbox=[108.0, 222.81, 505.24, 232.87] -->
- [29] Slav Petrov, Leon Barrett, Romain Thibaux, and Dan Klein. Learning accurate, compact,

<!-- p12-b10 reference col=0 bbox=[129.25, 233.54, 504.35, 265.57] -->
- and interpretable tree annotation. In Proceedings of the 21st International Conference on Computational Linguistics and 44th Annual Meeting of the ACL, pages 433–440. ACL, July 2006.

<!-- p12-b11 reference col=0 bbox=[108.0, 278.84, 504.0, 289.08] -->
- [30] Ofir Press and Lior Wolf. Using the output embedding to improve language models. arXiv

<!-- p12-b12 reference col=0 bbox=[129.58, 289.83, 264.53, 299.97] -->
- preprint arXiv:1608.05859, 2016.

<!-- p12-b13 reference col=0 bbox=[108.0, 313.49, 504.0, 323.45] -->
- [31] Rico Sennrich, Barry Haddow, and Alexandra Birch. Neural machine translation of rare words

<!-- p12-b14 reference col=0 bbox=[129.22, 324.22, 371.16, 334.36] -->
- with subword units. arXiv preprint arXiv:1508.07909, 2015.

<!-- p12-b15 reference col=0 bbox=[108.0, 347.88, 505.25, 357.84] -->
- [32] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton,

<!-- p12-b16 reference col=0 bbox=[129.58, 358.71, 504.0, 379.66] -->
- and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. arXiv preprint arXiv:1701.06538, 2017.

<!-- p12-b17 reference col=0 bbox=[108.0, 393.18, 505.65, 403.14] -->
- [33] Nitish Srivastava, Geoffrey E Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdi-

<!-- p12-b18 reference col=0 bbox=[129.3, 403.86, 504.0, 424.96] -->
- nov. Dropout: a simple way to prevent neural networks from overfitting. Journal of Machine Learning Research, 15(1):1929–1958, 2014.

<!-- p12-b19 reference col=0 bbox=[108.0, 438.41, 504.35, 448.47] -->
- [34] Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, and Rob Fergus. End-to-end memory

<!-- p12-b20 reference col=0 bbox=[128.97, 449.32, 505.25, 470.26] -->
- networks. In C. Cortes, N. D. Lawrence, D. D. Lee, M. Sugiyama, and R. Garnett, editors, Advances in Neural Information Processing Systems 28, pages 2440–2448. Curran Associates,

<!-- p12-b21 reference col=0 bbox=[129.58, 471.21, 172.19, 481.17] -->
- Inc., 2015.

<!-- p12-b22 reference col=0 bbox=[108.0, 494.62, 504.0, 504.68] -->
- [35] Ilya Sutskever, Oriol Vinyals, and Quoc VV Le. Sequence to sequence learning with neural

<!-- p12-b23 reference col=0 bbox=[129.58, 505.42, 494.13, 515.56] -->
- networks. In Advances in Neural Information Processing Systems, pages 3104–3112, 2014.

<!-- p12-b24 reference col=0 bbox=[108.0, 529.01, 505.74, 539.07] -->
- [36] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, and Zbigniew Wojna.

<!-- p12-b25 reference col=0 bbox=[129.58, 539.82, 484.32, 549.96] -->
- Rethinking the inception architecture for computer vision. CoRR, abs/1512.00567, 2015.

<!-- p12-b26 reference col=0 bbox=[108.0, 563.4, 504.0, 573.46] -->
- [37] Vinyals & Kaiser, Koo, Petrov, Sutskever, and Hinton. Grammar as a foreign language. In

<!-- p12-b27 reference col=0 bbox=[128.97, 574.21, 365.24, 584.35] -->
- Advances in Neural Information Processing Systems, 2015.

<!-- p12-b28 reference col=0 bbox=[108.0, 597.79, 504.0, 607.85] -->
- [38] Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang

<!-- p12-b29 reference col=0 bbox=[129.58, 608.85, 504.0, 640.56] -->
- Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. Google’s neural machine translation system: Bridging the gap between human and machine translation. arXiv preprint arXiv:1609.08144, 2016.

<!-- p12-b30 reference col=0 bbox=[108.0, 654.0, 504.0, 664.06] -->
- [39] Jie Zhou, Ying Cao, Xuguang Wang, Peng Li, and Wei Xu. Deep recurrent models with

<!-- p12-b31 reference col=0 bbox=[129.58, 664.81, 476.61, 674.95] -->
- fast-forward connections for neural machine translation. CoRR, abs/1606.04199, 2016.

<!-- p12-b32 reference col=0 bbox=[108.0, 688.39, 504.0, 698.46] -->
- [40] Muhua Zhu, Yue Zhang, Wenliang Chen, Min Zhang, and Jingbo Zhu. Fast and accurate

<!-- p12-b33 reference col=0 bbox=[128.83, 699.28, 504.0, 720.25] -->
- shift-reduce constituent parsing. In Proceedings of the 51st Annual Meeting of the ACL (Volume 1: Long Papers), pages 434–443. ACL, August 2013.


<!-- ===== page 13 ===== -->

<!-- p13-b1 reference col=0 bbox=[108.0, 312.31, 504.17, 355.08] -->
- Figure 3: An example of the attention mechanism following long-distance dependencies in the encoder self-attention in layer 5 of 6. Many of the attention heads attend to a distant dependency of the verb ‘making’, completing the phrase ‘making...more difficult’. Attentions here shown only for the word ‘making’. Different colors represent different heads. Best viewed in color.


<!-- ===== page 14 ===== -->

<!-- p14-b1 reference col=0 bbox=[108.0, 614.28, 505.39, 646.06] -->
- Figure 4: Two attention heads, also in layer 5 of 6, apparently involved in anaphora resolution. Top: Full attentions for head 5. Bottom: Isolated attentions from just the word ‘its’ for attention heads 5 and 6. Note that the attentions are very sharp for this word.


<!-- ===== page 15 ===== -->

<!-- p15-b1 reference col=0 bbox=[108.0, 602.32, 504.0, 634.18] -->
- Figure 5: Many of the attention heads exhibit behaviour that seems related to the structure of the sentence. We give two such examples above, from two different heads from the encoder self-attention at layer 5 of 6. The heads clearly learned to perform different tasks.


<!-- ===== removed running headers / footers / margin notes ===== -->

<!-- p1-m1 margin_note: arXiv:1706.03762v7 [cs.CL] 2 Aug 2023 -->

<!-- p2-m1 footer: 2 -->

<!-- p3-m1 footer: 3 -->

<!-- p4-m1 footer: 4 -->

<!-- p5-m1 footer: 5 -->

<!-- p6-m1 footer: 6 -->

<!-- p7-m1 footer: 7 -->

<!-- p8-m1 footer: 8 -->

<!-- p9-m1 footer: 9 -->

<!-- p10-m1 footer: 10 -->

<!-- p11-m1 footer: 11 -->

<!-- p12-m1 footer: 12 -->

<!-- p13-g1 figure_text: Attention Visualizations -->

<!-- p13-g2 figure_text: It -->

<!-- p13-g3 figure_text: is -->

<!-- p13-g4 figure_text: in -->

<!-- p13-g5 figure_text: this -->

<!-- p13-g6 figure_text: spirit -->

<!-- p13-g7 figure_text: that -->

<!-- p13-g8 figure_text: a -->

<!-- p13-g9 figure_text: majority -->

<!-- p13-g10 figure_text: of -->

<!-- p13-g11 figure_text: American -->

<!-- p13-g12 figure_text: governments -->

<!-- p13-g13 figure_text: have -->

<!-- p13-g14 figure_text: passed -->

<!-- p13-g15 figure_text: new -->

<!-- p13-g16 figure_text: laws -->

<!-- p13-g17 figure_text: since -->

<!-- p13-g18 figure_text: 2009 -->

<!-- p13-g19 figure_text: making -->

<!-- p13-g20 figure_text: the -->

<!-- p13-g21 figure_text: registration -->

<!-- p13-g22 figure_text: or -->

<!-- p13-g23 figure_text: voting -->

<!-- p13-g24 figure_text: process -->

<!-- p13-g25 figure_text: more -->

<!-- p13-g26 figure_text: difficult -->

<!-- p13-g27 figure_text: . -->

<!-- p13-g28 figure_text: <EOS> -->

<!-- p13-g29 figure_text: <pad> -->

<!-- p13-g30 figure_text: <pad> -->

<!-- p13-g31 figure_text: <pad> -->

<!-- p13-g32 figure_text: <pad> -->

<!-- p13-g33 figure_text: <pad> -->

<!-- p13-g34 figure_text: <pad> -->

<!-- p13-g35 figure_text: It -->

<!-- p13-g36 figure_text: is -->

<!-- p13-g37 figure_text: in -->

<!-- p13-g38 figure_text: this -->

<!-- p13-g39 figure_text: spirit -->

<!-- p13-g40 figure_text: that -->

<!-- p13-g41 figure_text: a -->

<!-- p13-g42 figure_text: majority -->

<!-- p13-g43 figure_text: of -->

<!-- p13-g44 figure_text: American -->

<!-- p13-g45 figure_text: governments -->

<!-- p13-g46 figure_text: have -->

<!-- p13-g47 figure_text: passed -->

<!-- p13-g48 figure_text: new -->

<!-- p13-g49 figure_text: laws -->

<!-- p13-g50 figure_text: since -->

<!-- p13-g51 figure_text: 2009 -->

<!-- p13-g52 figure_text: making -->

<!-- p13-g53 figure_text: the -->

<!-- p13-g54 figure_text: registration -->

<!-- p13-g55 figure_text: or -->

<!-- p13-g56 figure_text: voting -->

<!-- p13-g57 figure_text: process -->

<!-- p13-g58 figure_text: more -->

<!-- p13-g59 figure_text: difficult -->

<!-- p13-g60 figure_text: . -->

<!-- p13-g61 figure_text: <EOS> -->

<!-- p13-g62 figure_text: <pad> -->

<!-- p13-g63 figure_text: <pad> -->

<!-- p13-g64 figure_text: <pad> -->

<!-- p13-g65 figure_text: <pad> -->

<!-- p13-g66 figure_text: <pad> -->

<!-- p13-g67 figure_text: <pad> -->

<!-- p13-m1 footer: 13 -->

<!-- p14-g1 figure_text: The -->

<!-- p14-g2 figure_text: Law -->

<!-- p14-g3 figure_text: will -->

<!-- p14-g4 figure_text: never -->

<!-- p14-g5 figure_text: be -->

<!-- p14-g6 figure_text: perfect -->

<!-- p14-g7 figure_text: , -->

<!-- p14-g8 figure_text: but -->

<!-- p14-g9 figure_text: its -->

<!-- p14-g10 figure_text: application -->

<!-- p14-g11 figure_text: should -->

<!-- p14-g12 figure_text: be -->

<!-- p14-g13 figure_text: just -->

<!-- p14-g14 figure_text: - -->

<!-- p14-g15 figure_text: this -->

<!-- p14-g16 figure_text: is -->

<!-- p14-g17 figure_text: what -->

<!-- p14-g18 figure_text: we -->

<!-- p14-g19 figure_text: are -->

<!-- p14-g20 figure_text: missing -->

<!-- p14-g21 figure_text: , -->

<!-- p14-g22 figure_text: in -->

<!-- p14-g23 figure_text: my -->

<!-- p14-g24 figure_text: opinion -->

<!-- p14-g25 figure_text: . -->

<!-- p14-g26 figure_text: <EOS> -->

<!-- p14-g27 figure_text: <pad> -->

<!-- p14-g28 figure_text: The -->

<!-- p14-g29 figure_text: Law -->

<!-- p14-g30 figure_text: will -->

<!-- p14-g31 figure_text: never -->

<!-- p14-g32 figure_text: be -->

<!-- p14-g33 figure_text: perfect -->

<!-- p14-g34 figure_text: , -->

<!-- p14-g35 figure_text: but -->

<!-- p14-g36 figure_text: its -->

<!-- p14-g37 figure_text: application -->

<!-- p14-g38 figure_text: should -->

<!-- p14-g39 figure_text: be -->

<!-- p14-g40 figure_text: just -->

<!-- p14-g41 figure_text: - -->

<!-- p14-g42 figure_text: this -->

<!-- p14-g43 figure_text: is -->

<!-- p14-g44 figure_text: what -->

<!-- p14-g45 figure_text: we -->

<!-- p14-g46 figure_text: are -->

<!-- p14-g47 figure_text: missing -->

<!-- p14-g48 figure_text: , -->

<!-- p14-g49 figure_text: in -->

<!-- p14-g50 figure_text: my -->

<!-- p14-g51 figure_text: opinion -->

<!-- p14-g52 figure_text: . -->

<!-- p14-g53 figure_text: <EOS> -->

<!-- p14-g54 figure_text: <pad> -->

<!-- p14-g55 figure_text: The -->

<!-- p14-g56 figure_text: Law -->

<!-- p14-g57 figure_text: will -->

<!-- p14-g58 figure_text: never -->

<!-- p14-g59 figure_text: be -->

<!-- p14-g60 figure_text: perfect -->

<!-- p14-g61 figure_text: , -->

<!-- p14-g62 figure_text: but -->

<!-- p14-g63 figure_text: its -->

<!-- p14-g64 figure_text: application -->

<!-- p14-g65 figure_text: should -->

<!-- p14-g66 figure_text: be -->

<!-- p14-g67 figure_text: just -->

<!-- p14-g68 figure_text: - -->

<!-- p14-g69 figure_text: this -->

<!-- p14-g70 figure_text: is -->

<!-- p14-g71 figure_text: what -->

<!-- p14-g72 figure_text: we -->

<!-- p14-g73 figure_text: are -->

<!-- p14-g74 figure_text: missing -->

<!-- p14-g75 figure_text: , -->

<!-- p14-g76 figure_text: in -->

<!-- p14-g77 figure_text: my -->

<!-- p14-g78 figure_text: opinion -->

<!-- p14-g79 figure_text: . -->

<!-- p14-g80 figure_text: <EOS> -->

<!-- p14-g81 figure_text: <pad> -->

<!-- p14-g82 figure_text: The -->

<!-- p14-g83 figure_text: Law -->

<!-- p14-g84 figure_text: will -->

<!-- p14-g85 figure_text: never -->

<!-- p14-g86 figure_text: be -->

<!-- p14-g87 figure_text: perfect -->

<!-- p14-g88 figure_text: , -->

<!-- p14-g89 figure_text: but -->

<!-- p14-g90 figure_text: its -->

<!-- p14-g91 figure_text: application -->

<!-- p14-g92 figure_text: should -->

<!-- p14-g93 figure_text: be -->

<!-- p14-g94 figure_text: just -->

<!-- p14-g95 figure_text: - -->

<!-- p14-g96 figure_text: this -->

<!-- p14-g97 figure_text: is -->

<!-- p14-g98 figure_text: what -->

<!-- p14-g99 figure_text: we -->

<!-- p14-g100 figure_text: are -->

<!-- p14-g101 figure_text: missing -->

<!-- p14-g102 figure_text: , -->

<!-- p14-g103 figure_text: in -->

<!-- p14-g104 figure_text: my -->

<!-- p14-g105 figure_text: opinion -->

<!-- p14-g106 figure_text: . -->

<!-- p14-g107 figure_text: <EOS> -->

<!-- p14-g108 figure_text: <pad> -->

<!-- p14-m1 footer: 14 -->

<!-- p15-g1 figure_text: The -->

<!-- p15-g2 figure_text: Law -->

<!-- p15-g3 figure_text: will -->

<!-- p15-g4 figure_text: never -->

<!-- p15-g5 figure_text: be -->

<!-- p15-g6 figure_text: perfect -->

<!-- p15-g7 figure_text: , -->

<!-- p15-g8 figure_text: but -->

<!-- p15-g9 figure_text: its -->

<!-- p15-g10 figure_text: application -->

<!-- p15-g11 figure_text: should -->

<!-- p15-g12 figure_text: be -->

<!-- p15-g13 figure_text: just -->

<!-- p15-g14 figure_text: - -->

<!-- p15-g15 figure_text: this -->

<!-- p15-g16 figure_text: is -->

<!-- p15-g17 figure_text: what -->

<!-- p15-g18 figure_text: we -->

<!-- p15-g19 figure_text: are -->

<!-- p15-g20 figure_text: missing -->

<!-- p15-g21 figure_text: , -->

<!-- p15-g22 figure_text: in -->

<!-- p15-g23 figure_text: my -->

<!-- p15-g24 figure_text: opinion -->

<!-- p15-g25 figure_text: . -->

<!-- p15-g26 figure_text: <EOS> -->

<!-- p15-g27 figure_text: <pad> -->

<!-- p15-g28 figure_text: The -->

<!-- p15-g29 figure_text: Law -->

<!-- p15-g30 figure_text: will -->

<!-- p15-g31 figure_text: never -->

<!-- p15-g32 figure_text: be -->

<!-- p15-g33 figure_text: perfect -->

<!-- p15-g34 figure_text: , -->

<!-- p15-g35 figure_text: but -->

<!-- p15-g36 figure_text: its -->

<!-- p15-g37 figure_text: application -->

<!-- p15-g38 figure_text: should -->

<!-- p15-g39 figure_text: be -->

<!-- p15-g40 figure_text: just -->

<!-- p15-g41 figure_text: - -->

<!-- p15-g42 figure_text: this -->

<!-- p15-g43 figure_text: is -->

<!-- p15-g44 figure_text: what -->

<!-- p15-g45 figure_text: we -->

<!-- p15-g46 figure_text: are -->

<!-- p15-g47 figure_text: missing -->

<!-- p15-g48 figure_text: , -->

<!-- p15-g49 figure_text: in -->

<!-- p15-g50 figure_text: my -->

<!-- p15-g51 figure_text: opinion -->

<!-- p15-g52 figure_text: . -->

<!-- p15-g53 figure_text: <EOS> -->

<!-- p15-g54 figure_text: <pad> -->

<!-- p15-g55 figure_text: The -->

<!-- p15-g56 figure_text: Law -->

<!-- p15-g57 figure_text: will -->

<!-- p15-g58 figure_text: never -->

<!-- p15-g59 figure_text: be -->

<!-- p15-g60 figure_text: perfect -->

<!-- p15-g61 figure_text: , -->

<!-- p15-g62 figure_text: but -->

<!-- p15-g63 figure_text: its -->

<!-- p15-g64 figure_text: application -->

<!-- p15-g65 figure_text: should -->

<!-- p15-g66 figure_text: be -->

<!-- p15-g67 figure_text: just -->

<!-- p15-g68 figure_text: - -->

<!-- p15-g69 figure_text: this -->

<!-- p15-g70 figure_text: is -->

<!-- p15-g71 figure_text: what -->

<!-- p15-g72 figure_text: we -->

<!-- p15-g73 figure_text: are -->

<!-- p15-g74 figure_text: missing -->

<!-- p15-g75 figure_text: , -->

<!-- p15-g76 figure_text: in -->

<!-- p15-g77 figure_text: my -->

<!-- p15-g78 figure_text: opinion -->

<!-- p15-g79 figure_text: . -->

<!-- p15-g80 figure_text: <EOS> -->

<!-- p15-g81 figure_text: <pad> -->

<!-- p15-g82 figure_text: The -->

<!-- p15-g83 figure_text: Law -->

<!-- p15-g84 figure_text: will -->

<!-- p15-g85 figure_text: never -->

<!-- p15-g86 figure_text: be -->

<!-- p15-g87 figure_text: perfect -->

<!-- p15-g88 figure_text: , -->

<!-- p15-g89 figure_text: but -->

<!-- p15-g90 figure_text: its -->

<!-- p15-g91 figure_text: application -->

<!-- p15-g92 figure_text: should -->

<!-- p15-g93 figure_text: be -->

<!-- p15-g94 figure_text: just -->

<!-- p15-g95 figure_text: - -->

<!-- p15-g96 figure_text: this -->

<!-- p15-g97 figure_text: is -->

<!-- p15-g98 figure_text: what -->

<!-- p15-g99 figure_text: we -->

<!-- p15-g100 figure_text: are -->

<!-- p15-g101 figure_text: missing -->

<!-- p15-g102 figure_text: , -->

<!-- p15-g103 figure_text: in -->

<!-- p15-g104 figure_text: my -->

<!-- p15-g105 figure_text: opinion -->

<!-- p15-g106 figure_text: . -->

<!-- p15-g107 figure_text: <EOS> -->

<!-- p15-g108 figure_text: <pad> -->

<!-- p15-m1 footer: 15 -->
