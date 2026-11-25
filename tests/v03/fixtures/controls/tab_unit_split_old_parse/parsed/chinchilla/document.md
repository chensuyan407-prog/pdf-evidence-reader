<!-- source: C:\Users\12117\Desktop\bohrium-pdf-evidence-reader\v0.3\tests\holdout_pdfs\2203.15556.pdf | status: success | parser: pdf-evidence parse_pdf.py 0.3.0 -->


<!-- ===== page 1 ===== -->

<!-- p1-b1 title col=0 bbox=[72.78, 84.65, 498.59, 102.58] -->
# Training Compute-Optimal Large Language Models

<!-- p1-b2 author_affiliation col=0 bbox=[63.11, 119.34, 509.14, 131.23] -->
Jordan Hoffmann^{★}, Sebastian Borgeaud^{★}, Arthur Mensch^{★}, Elena Buchatskaya, Trevor Cai, Eliza Rutherford,

<!-- p1-b3 author_affiliation col=0 bbox=[67.59, 135.22, 505.07, 157.14] -->
Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan,

<!-- p1-b4 author_affiliation col=0 bbox=[166.66, 158.19, 404.49, 170.09] -->
Erich Elsen, Jack W. Rae, Oriol Vinyals and Laurent Sifre^{★}

<!-- p1-b5 author_affiliation col=0 bbox=[249.89, 171.68, 321.23, 181.78] -->
^{★}Equal contributions

<!-- p1-b6 body col=0 bbox=[61.78, 209.39, 534.64, 391.51] -->
We investigate the optimal model size and number of tokens for training a transformer language model under a given compute budget. We find that current large language models are significantly undertrained, a consequence of the recent focus on scaling language models whilst keeping the amount of training data constant. By training over 400 language models ranging from 70 million to over 16 billion parameters on 5 to 500 billion tokens, we find that for compute-optimal training, the model size and the number of training tokens should be scaled equally: for every doubling of model size the number of training tokens should also be doubled. We test this hypothesis by training a predicted computeoptimal model, Chinchilla, that uses the same compute budget as Gopher but with 70B parameters and 4× more more data. Chinchilla uniformly and significantly outperforms Gopher (280B), GPT-3 (175B), Jurassic-1 (178B), and Megatron-Turing NLG (530B) on a large range of downstream evaluation tasks. This also means that Chinchilla uses substantially less compute for fine-tuning and inference, greatly facilitating downstream usage. As a highlight, Chinchilla reaches a state-of-the-art average accuracy of 67.5% on the MMLU benchmark, greater than a 7% improvement over Gopher.

<!-- p1-b7 heading col=0 bbox=[62.36, 416.67, 154.68, 429.62] -->
## 1. Introduction

<!-- p1-b8 author_affiliation col=0 bbox=[61.46, 443.27, 533.12, 494.82] -->
Recently a series of Large Language Models (LLMs) have been introduced (Brown et al., 2020; Lieber et al., 2021; Rae et al., 2021; Smith et al., 2022; Thoppilan et al., 2022), with the largest dense language models now having over 500 billion parameters. These large autoregressive transformers (Vaswani et al., 2017) have demonstrated impressive performance on many tasks using a variety of

<!-- p1-b9 author_affiliation col=0 bbox=[62.36, 497.46, 371.38, 508.37] -->
evaluation protocols such as zero-shot, few-shot, and fine-tuning.

<!-- p1-b10 body col=0 bbox=[61.94, 517.79, 534.73, 582.89] -->
The compute and energy cost for training large language models is substantial (Rae et al., 2021; Thoppilan et al., 2022) and rises with increasing model size. In practice, the allocated training compute budget is often known in advance: how many accelerators are available and for how long we want to use them. Since it is typically only feasible to train these large models once, accurately estimating the best model hyperparameters for a given compute budget is critical (Tay et al., 2021).

<!-- p1-b11 body col=0 bbox=[62.36, 592.31, 532.92, 711.61] -->
Kaplan et al. (2020) showed that there is a power law relationship between the number of parameters in an autoregressive language model (LM) and its performance. As a result, the field has been training larger and larger models, expecting performance improvements. One notable conclusion in Kaplan et al. (2020) is that large models should not be trained to their lowest possible loss to be compute optimal. Whilst we reach the same conclusion, we estimate that large models should be trained for many more training tokens than recommended by the authors. Specifically, given a 10× increase computational budget, they suggests that the size of the model should increase 5.5× while the number of training tokens should only increase 1.8×. Instead, we find that model size and the number of training tokens should be scaled in equal proportions.

<!-- p1-b12 body col=0 bbox=[62.36, 721.02, 532.92, 759.03] -->
Following Kaplan et al. (2020) and the training setup of GPT-3 (Brown et al., 2020), many of the recently trained large models have been trained for approximately 300 billion tokens (Table 1), in line with the approach of predominantly increasing model size when increasing compute.

<!-- p1-b13 body col=0 bbox=[61.65, 786.66, 352.82, 804.34] -->
Corresponding authors: {jordanhoffmann|sborgeaud|amensch|sifre}@deepmind.com © 2023 DeepMind. All rights reserved


<!-- ===== page 2 ===== -->

<!-- p2-b1 caption col=0 bbox=[62.36, 294.89, 534.43, 374.08] -->
> **Figure 1 | Overlaid predictions. We overlay the predictions from our three different approaches, along with projections from Kaplan et al. (2020). We find that all three methods predict that current large models should be substantially smaller and therefore trained much longer than is currently done. In Figure A3, we show the results with the predicted optimal tokens plotted against the optimal number of parameters for fixed FLOP budgets. Chinchilla outperforms Gopher and the other large models (see Section 4.2).**

<!-- p2-b2 body col=0 bbox=[62.36, 394.22, 532.91, 422.62] -->
In this work, we revisit the question: Given a fixed FLOPs budget,^{1} how should one trade-off model size and the number of training tokens? To answer this question, we model the final pre-training loss^{2}

<!-- p2-b3 body col=0 bbox=[62.36, 423.74, 534.74, 476.56] -->
𝐿(𝑁, 𝐷) as a function of the number of model parameters 𝑁, and the number of training tokens, 𝐷. Since the computational budget 𝐶is a deterministic function FLOPs(𝑁, 𝐷) of the number of seen training tokens and model parameters, we are interested in minimizing 𝐿under the constraint FLOPs(𝑁, 𝐷) = 𝐶:

<!-- p2-b4 equation col=0 bbox=[195.95, 477.94, 337.42, 490.11] -->
$$ 𝑁_{𝑜𝑝𝑡}(𝐶), 𝐷_{𝑜𝑝𝑡}(𝐶) = argmin $$  (eq. 1)

<!-- p2-b5 equation col=0 bbox=[362.63, 477.94, 399.58, 489.78] -->
$$ 𝐿(𝑁, 𝐷). $$  (eq. 1)

<!-- p2-b6 equation col=0 bbox=[517.75, 479.13, 532.91, 490.04] -->
$$ (1) $$  (eq. 1)

<!-- p2-b7 equation col=0 bbox=[280.93, 490.48, 360.0, 499.37] -->
$$ 𝑁,𝐷s.t. FLOPs(𝑁,𝐷)=𝐶 $$  (eq. 1)

<!-- p2-b8 body col=0 bbox=[62.03, 506.05, 533.12, 585.89] -->
The functions 𝑁_{𝑜𝑝𝑡}(𝐶), and 𝐷_{𝑜𝑝𝑡}(𝐶) describe the optimal allocation of a computational budget 𝐶. We empirically estimate these functions based on the losses of over 400 models, ranging from under 70M to over 16B parameters, and trained on 5B to over 400B tokens – with each model configuration trained for several different training horizons. Our approach leads to considerably different results than that of Kaplan et al. (2020). We highlight our results in Figure 1 and how our approaches differ in Section 2.

<!-- p2-b9 body col=0 bbox=[62.36, 595.3, 533.13, 701.06] -->
Based on our estimated compute-optimal frontier, we predict that for the compute budget used to train Gopher, an optimal model should be 4 times smaller, while being training on 4 times more tokens. We verify this by training a more compute-optimal 70B model, called Chinchilla, on 1.4 trillion tokens. Not only does Chinchilla outperform its much larger counterpart, Gopher, but its reduced model size reduces inference cost considerably and greatly facilitates downstream uses on smaller hardware. The energy cost of a large language model is amortized through its usage for inference an fine-tuning. The benefits of a more optimally trained smaller model, therefore, extend beyond the immediate benefits of its improved performance.

<!-- p2-b10 footnote col=0 bbox=[62.01, 725.33, 532.91, 758.64] -->
[^fn]: ^{1}For example, knowing the number of accelerators and a target training duration. ^{2}For simplicity, we perform our analysis on the smoothed training loss which is an unbiased estimate of the test loss, as we are in the infinite data regime (the number of training tokens is less than the number of tokens in the entire corpus).


<!-- ===== page 3 ===== -->

<!-- p3-b1 caption col=0 bbox=[62.03, 85.55, 534.43, 137.35] -->
> **Table 1 | Current LLMs. We show five of the current largest dense transformer models, their size, and the number of training tokens. Other than LaMDA (Thoppilan et al., 2022), most models are trained for approximately 300 billion tokens. We introduce Chinchilla, a substantially smaller model, trained for much longer than 300B tokens.**

<!-- T1 table page=3 bbox=[111.46, 147.63, 483.82, 263.22] -->
| Model |  | Size (# | Parameters) | Training | Tokens |
|---|---|---|---|---|---|
| LaMDA (Thoppilan et 2022) | al., | 137 | Billion | 168 | Billion |
| GPT-3 (Brown et al., | 2020) | 175 | Billion | 300 | Billion |
| Jurassic (Lieber et al., | 2021) | 178 | Billion | 300 | Billion |
| Gopher (Rae et al., | 2021) | 280 | Billion | 300 | Billion |
| MT-NLG 530B (Smith | et al., 2022) | 530 | Billion | 270 | Billion |
| Chinchilla |  | 70 | Billion | 1.4 | Trillion |

<!-- p3-b5 heading col=0 bbox=[62.36, 282.52, 160.6, 295.47] -->
## 2. Related Work

<!-- p3-b6 body col=0 bbox=[62.36, 309.12, 533.34, 455.52] -->
Large language models. A variety of large language models have been introduced in the last few years. These include both dense transformer models (Brown et al., 2020; Lieber et al., 2021; Rae et al., 2021; Smith et al., 2022; Thoppilan et al., 2022) and mixture-of-expert (MoE) models (Du et al., 2021; Fedus et al., 2021; Zoph et al., 2022). The largest dense transformers have passed 500 billion parameters (Smith et al., 2022). The drive to train larger and larger models is clear—so far increasing the size of language models has been responsible for improving the state-of-the-art in many language modelling tasks. Nonetheless, large language models face several challenges, including their overwhelming computational requirements (the cost of training and inference increase with model size) (Rae et al., 2021; Thoppilan et al., 2022) and the need for acquiring more high-quality training data. In fact, in this work we find that larger, high quality datasets will play a key role in any further scaling of language models.

<!-- p3-b7 body col=0 bbox=[61.46, 480.19, 534.74, 734.99] -->
Modelling the scaling behavior. Understanding the scaling behaviour of language models and their transfer properties has been important in the development of recent large models (Hernandez et al., 2021; Kaplan et al., 2020). Kaplan et al. (2020) first showed a predictable relationship between model size and loss over many orders of magnitude. The authors investigate the question of choosing the optimal model size to train for a given compute budget. Similar to us, they address this question by training various models. Our work differs from Kaplan et al. (2020) in several important ways. First, the authors use a fixed number of training tokens and learning rate schedule for all models; this prevents them from modelling the impact of these hyperparameters on the loss. In contrast, we find that setting the learning rate schedule to approximately match the number of training tokens results in the best final loss regardless of model size—see Figure A1. For a fixed learning rate cosine schedule to 130B tokens, the intermediate loss estimates (for 𝐷^{′} << 130B) are therefore overestimates of the loss of a model trained with a schedule length matching 𝐷^{′}. Using these intermediate losses results in underestimating the effectiveness of training models on less data than 130B tokens, and eventually contributes to the conclusion that model size should increase faster than training data size as compute budget increases. In contrast, our analysis predicts that both quantities should scale at roughly the same rate. Secondly, we include models with up to 16B parameters, as we observe that there is slight curvature in the FLOP-loss frontier (see Appendix E)—in fact, the majority of the models used in our analysis have more than 500 million parameters, in contrast the majority of runs in Kaplan et al. (2020) are significantly smaller—many being less than 100M parameters.

<!-- p3-b8 body col=0 bbox=[79.3, 744.4, 533.1, 755.31] -->
Recently, Clark et al. (2022) specifically looked in to the scaling properties of Mixture of Expert


<!-- ===== page 4 ===== -->

<!-- p4-b1 body col=0 bbox=[62.36, 87.27, 532.91, 138.83] -->
language models, showing that the scaling with number of experts diminishes as the model size increases—their approach models the loss as a function of two variables: the model size and the number of experts. However, the analysis is done with a fixed number of training tokens, as in Kaplan et al. (2020), potentially underestimating the improvements of branching.

<!-- p4-b2 body col=0 bbox=[61.46, 163.5, 534.74, 282.81] -->
Estimating hyperparameters for large models. The model size and the number of training tokens are not the only two parameters to chose when selecting a language model and a procedure to train it. Other important factors include learning rate, learning rate schedule, batch size, optimiser, and width-to-depth ratio. In this work, we focus on model size and the number of training steps, and we rely on existing work and provided experimental heuristics to determine the other necessary hyperparameters. Yang et al. (2021) investigates how to choose a variety of these parameters for training an autoregressive transformer, including the learning rate and batch size. McCandlish et al. (2018) finds only a weak dependence between optimal batch size and model size. Shallue et al. (2018); Zhang et al. (2019) suggest that using larger batch-sizes than those we use is possible. Levine

<!-- p4-b3 body col=0 bbox=[62.36, 285.45, 533.13, 323.45] -->
et al. (2020) investigates the optimal depth-to-width ratio for a variety of standard model sizes. We use slightly less deep models than proposed as this translates to better wall-clock performance on our hardware.

<!-- p4-b4 body col=0 bbox=[61.46, 348.13, 534.74, 467.43] -->
Improved model architectures. Recently, various promising alternatives to traditional dense transformers have been proposed. For example, through the use of conditional computation large MoE models like the 1.7 trillion parameter Switch transformer (Fedus et al., 2021), the 1.2 Trillion parameter GLaM model (Du et al., 2021), and others (Artetxe et al., 2021; Zoph et al., 2022) are able to provide a large effective model size despite using relatively fewer training and inference FLOPs. However, for very large models the computational benefits of routed models seems to diminish (Clark et al., 2022). An orthogonal approach to improving language models is to augment transformers with explicit retrieval mechanisms, as done by Borgeaud et al. (2021); Guu et al. (2020); Lewis et al. (2020). This approach effectively increases the number of data tokens seen during training (by a

<!-- p4-b5 body col=0 bbox=[62.36, 469.82, 532.91, 494.53] -->
factor of ∼10 in Borgeaud et al. (2021)). This suggests that the performance of language models may be more dependant on the size of the training data than previously thought.

<!-- p4-b6 heading col=0 bbox=[62.36, 518.8, 443.3, 531.75] -->
## 3. Estimating the optimal parameter/training tokens allocation

<!-- p4-b7 body col=0 bbox=[61.86, 545.39, 532.91, 556.67] -->
We present three different approaches to answer the question driving our research: Given a fixed

<!-- p4-b8 body col=0 bbox=[61.86, 558.94, 534.73, 610.5] -->
FLOPs budget, how should one trade-off model size and the number of training tokens? In all three cases we start by training a range of models varying both model size and the number of training tokens and use the resulting training curves to fit an empirical estimator of how they should scale. We assume a power-law relationship between compute and model size as done in Clark et al. (2022);

<!-- p4-b9 body col=0 bbox=[61.94, 613.14, 535.64, 678.24] -->
Kaplan et al. (2020), though future work may want to include potential curvature in this relationship for large model sizes. The resulting predictions are similar for all three methods and suggest that parameter count and number of training tokens should be increased equally with more compute^{3}— with proportions reported in Table 2. This is in clear contrast to previous work on this topic and warrants further investigation.

<!-- p4-b10 footnote col=0 bbox=[77.35, 747.78, 266.62, 758.64] -->
[^fn]: ^{3}We compute FLOPs as described in Appendix F.


<!-- ===== page 5 ===== -->

<!-- p5-b1 caption col=0 bbox=[61.46, 211.68, 532.92, 277.33] -->
> **Figure 2 | Training curve envelope. On the left we show all of our different runs. We launched a range of model sizes going from 70M to 10B, each for four different cosine cycle lengths. From these curves, we extracted the envelope of minimal loss per FLOP, and we used these points to estimate the optimal model size (center) for a given compute budget and the optimal number of training tokens (right). In green, we show projections of optimal model size and training token count based on the**

<!-- p5-b2 body col=0 bbox=[62.36, 278.07, 312.99, 290.94] -->
number of FLOPs used to train Gopher (5.76 × 10^{23}).

<!-- p5-b3 heading col=0 bbox=[62.36, 314.92, 411.78, 325.83] -->
### 3.1. Approach 1: Fix model sizes and vary number of training tokens

<!-- p5-b4 body col=0 bbox=[62.06, 336.95, 534.73, 388.5] -->
In our first approach we vary the number of training steps for a fixed family of models (ranging from 70M to over 10B parameters), training each model for 4 different number of training sequences. From these runs, we are able to directly extract an estimate of the minimum loss achieved for a given number of training FLOPs. Training details for this approach can be found in Appendix D.

<!-- p5-b5 body col=0 bbox=[61.46, 396.73, 533.12, 530.77] -->
For each parameter count 𝑁we train 4 different models, decaying the learning rate by a factor of 10× over a horizon (measured in number of training tokens) that ranges by a factor of 16×. Then, for each run, we smooth and then interpolate the training loss curve. From this, we obtain a continuous mapping from FLOP count to training loss for each run. Then, for each FLOP count, we determine which run achieves the lowest loss. Using these interpolants, we obtain a mapping from any FLOP count 𝐶, to the most efficient choice of model size 𝑁and number of training tokens 𝐷such that FLOPs(𝑁, 𝐷) = 𝐶.^{4} At 1500 logarithmically spaced FLOP values, we find which model size achieves the lowest loss of all models along with the required number of training tokens. Finally, we fit power laws to estimate the optimal model size and number of training tokens for any given amount of compute (see the center and right panels of Figure 2), obtaining a relationship 𝑁_{𝑜𝑝𝑡}∝𝐶^{𝑎}and 𝐷_{𝑜𝑝𝑡}∝𝐶^{𝑏}. We

<!-- p5-b6 body col=0 bbox=[62.36, 532.22, 532.91, 571.42] -->
find that 𝑎= 0.50 and 𝑏= 0.50—as summarized in Table 2. In Section D.4, we show a head-to-head comparison at 10^{21} FLOPs, using the model size recommended by our analysis and by the analysis of Kaplan et al. (2020)—using the model size we predict has a clear advantage.

<!-- p5-b7 heading col=0 bbox=[62.36, 594.24, 234.18, 605.15] -->
### 3.2. Approach 2: IsoFLOP profiles

<!-- p5-b8 body col=0 bbox=[62.36, 613.41, 532.42, 628.28] -->
In our second approach we vary the model size^{5} for a fixed set of 9 different training FLOP counts^{6}

<!-- p5-b9 body col=0 bbox=[61.46, 626.96, 532.91, 641.83] -->
(ranging from 6 × 10^{18} to 3 × 10^{21} FLOPs), and consider the final training loss for each point^{7}. in

<!-- p5-b10 body col=0 bbox=[62.36, 643.28, 532.92, 668.93] -->
contrast with Approach 1 that considered points (𝑁, 𝐷, 𝐿) along the entire training runs. This allows us to directly answer the question: For a given FLOP budget, what is the optimal parameter count?

<!-- p5-b11 footnote col=0 bbox=[62.01, 691.2, 534.16, 713.03] -->
[^fn]: ^{4}Note that all selected points are within the last 15% of training. This suggests that when training a model over 𝐷tokens, we should pick a cosine cycle length that decays 10× over approximately 𝐷tokens—see further details in Appendix B.

<!-- p5-b12 footnote col=0 bbox=[62.36, 713.65, 532.91, 758.64] -->
[^fn]: ^{5}In approach 2, model size varies up to 16B as opposed to approach 1 where we only used models up to 10B. ^{6}The number of training tokens is determined by the model size and training FLOPs. ^{7}We set the cosine schedule length to match the number of tokens, which is optimal according to the analysis presented in Appendix B.


<!-- ===== page 6 ===== -->

<!-- p6-b1 caption col=0 bbox=[61.46, 224.32, 533.12, 276.41] -->
> **Figure 3 | IsoFLOP curves. For various model sizes, we choose the number of training tokens such that the final FLOPs is a constant. The cosine cycle length is set to match the target FLOP count. We find a clear valley in loss, meaning that for a given FLOP budget there is an optimal model to train (left). Using the location of these valleys, we project optimal model size and number of tokens for**

<!-- p6-b2 body col=0 bbox=[62.36, 278.76, 532.91, 303.57] -->
larger models (center and right). In green, we show the estimated number of parameters and tokens for an optimal model trained with the compute budget of Gopher.

<!-- p6-b3 body col=0 bbox=[79.3, 327.26, 532.91, 338.17] -->
For each FLOP budget, we plot the final loss (after smoothing) against the parameter count in

<!-- p6-b4 body col=0 bbox=[62.36, 340.81, 533.1, 419.54] -->
Figure 3 (left). In all cases, we ensure that we have trained a diverse enough set of model sizes to see a clear minimum in the loss. We fit a parabola to each IsoFLOPs curve to directly estimate at what model size the minimum loss is achieved (Figure 3 (left)). As with the previous approach, we then fit a power law between FLOPs and loss-optimal model size and number of training tokens, shown in Figure 3 (center, right). Again, we fit exponents of the form 𝑁_{𝑜𝑝𝑡}∝𝐶^{𝑎}and 𝐷_{𝑜𝑝𝑡}∝𝐶^{𝑏}and we find that 𝑎= 0.49 and 𝑏= 0.51—as summarized in Table 2.

<!-- p6-b5 heading col=0 bbox=[62.36, 442.29, 320.33, 453.2] -->
### 3.3. Approach 3: Fitting a parametric loss function

<!-- p6-b6 body col=0 bbox=[62.36, 464.32, 532.92, 502.32] -->
Lastly, we model all final losses from experiments in Approach 1 & 2 as a parametric function of model parameter count and the number of seen tokens. Following a classical risk decomposition (see Section D.2), we propose the following functional form

<!-- p6-b7 equation col=0 bbox=[243.11, 511.37, 319.07, 530.6] -->
$$ ˆ𝐿(𝑁, 𝐷) ≜𝐸+ ^{𝐴} $$  (eq. 2)

<!-- p6-b8 equation col=0 bbox=[309.57, 511.37, 345.13, 536.27] -->
$$ 𝑁^{𝛼}^{+} ^{𝐵} $$  (eq. 2)

<!-- p6-b9 equation col=0 bbox=[336.21, 518.75, 352.42, 536.67] -->
$$ 𝐷^{𝛽}^{.} $$  (eq. 2)

<!-- p6-b10 equation col=0 bbox=[517.75, 519.94, 532.91, 530.85] -->
$$ (2) $$  (eq. 2)

<!-- p6-b11 body col=0 bbox=[62.03, 547.88, 532.92, 612.98] -->
The first term captures the loss for an ideal generative process on the data distribution, and should correspond to the entropy of natural text. The second term captures the fact that a perfectly trained transformer with 𝑁parameters underperforms the ideal generative process. The final term captures the fact that the transformer is not trained to convergence, as we only make a finite number of optimisation steps, on a sample of the dataset distribution.

<!-- p6-b12 body col=0 bbox=[62.36, 636.47, 532.92, 662.12] -->
Model fitting. To estimate (𝐴, 𝐵, 𝐸, 𝛼, 𝛽), we minimize the Huber loss (Huber, 1964) between the predicted and observed log loss using the L-BFGS algorithm (Nocedal, 1980):

<!-- p6-b13 equation col=0 bbox=[297.95, 673.95, 395.27, 689.1] -->
$$ � log ^{ˆ}𝐿(𝑁_{𝑖}, 𝐷_{𝑖}) −log 𝐿_{𝑖} $$  (eq. 3)

<!-- p6-b14 equation col=0 bbox=[396.07, 673.95, 399.95, 684.86] -->
$$ � $$  (eq. 3)

<!-- p6-b15 equation col=0 bbox=[242.93, 674.85, 257.36, 685.75] -->
$$ ∑︁ $$  (eq. 3)

<!-- p6-b16 equation col=0 bbox=[201.3, 678.2, 219.74, 689.1] -->
$$ min $$  (eq. 3)

<!-- p6-b17 equation col=0 bbox=[262.79, 678.2, 296.58, 689.1] -->
$$ Huber_{𝛿} $$  (eq. 3)

<!-- p6-b18 equation col=0 bbox=[517.75, 678.2, 532.91, 689.1] -->
$$ (3) $$  (eq. 3)

<!-- p6-b19 equation col=0 bbox=[195.01, 686.89, 226.26, 694.22] -->
$$ 𝐴,𝐵,𝐸,𝛼,𝛽 $$  (eq. 3)

<!-- p6-b20 body col=0 bbox=[239.31, 692.48, 260.9, 701.31] -->
Runs 𝑖

<!-- p6-b21 body col=0 bbox=[61.86, 712.21, 533.13, 723.12] -->
We account for possible local minima by selecting the best fit from a grid of initialisations. The Huber

<!-- p6-b22 body col=0 bbox=[62.36, 723.97, 533.12, 750.22] -->
loss (𝛿= 10^{−}^{3}) is robust to outliers, which we find important for good predictive performance over held-out data points. Section D.2 details the fitting procedure and the loss decomposition.


<!-- ===== page 7 ===== -->

<!-- p7-b1 caption col=0 bbox=[62.36, 298.24, 534.43, 365.26] -->
> **Figure 4 | Parametric fit. We fit a parametric modelling of the loss ^{ˆ}𝐿(𝑁, 𝐷) and display contour (left) and isoFLOP slices (right). For each isoFLOP slice, we include a corresponding dashed line in the left plot. In the left plot, we show the efficient frontier in blue, which is a line in log-log space. Specifically, the curve goes through each iso-loss contour at the point with the fewest FLOPs. We project the optimal model size given the Gopher FLOP budget to be 40B parameters.**

<!-- p7-b2 body col=0 bbox=[62.36, 387.3, 533.18, 440.04] -->
Efficient frontier. We can approximate the functions 𝑁_{𝑜𝑝𝑡}and 𝐷_{𝑜𝑝𝑡}by minimizing the parametric loss ^{ˆ}𝐿under the constraint FLOPs(𝑁, 𝐷) ≈6𝑁𝐷(Kaplan et al., 2020). The resulting 𝑁_{𝑜𝑝𝑡}and 𝐷_{𝑜𝑝𝑡} balance the two terms in Equation (3) that depend on model size and data. By construction, they have a power-law form:

<!-- p7-b3 equation col=0 bbox=[360.33, 450.79, 370.57, 462.61] -->
$$ 1 𝛼+𝛽 $$  (eq. 4)

<!-- p7-b4 equation col=0 bbox=[139.93, 451.55, 227.89, 473.97] -->
$$ �_{𝑎} , 𝐷_{𝑜𝑝𝑡}(𝐶) = 𝐺^{−}^{1} $$  (eq. 4)

<!-- p7-b5 equation col=0 bbox=[242.58, 451.55, 330.29, 473.97] -->
$$ �_{𝑏} , where 𝐺= $$  (eq. 4)

<!-- p7-b6 equation col=0 bbox=[126.6, 454.42, 138.31, 481.38] -->
$$ �𝐶 6 $$  (eq. 4)

<!-- p7-b7 equation col=0 bbox=[229.26, 454.42, 240.97, 481.38] -->
$$ �𝐶 6 $$  (eq. 4)

<!-- p7-b8 equation col=0 bbox=[334.19, 454.42, 352.71, 479.32] -->
$$ �𝛼𝐴 𝛽𝐵 $$  (eq. 4)

<!-- p7-b9 equation col=0 bbox=[372.52, 454.42, 507.64, 481.13] -->
$$ , 𝑎= 𝛽 𝛼+ 𝛽^{,} ^{and} ^{𝑏}^{=} 𝛼 𝛼+ 𝛽^{.} $$  (eq. 4)

<!-- p7-b10 equation col=0 bbox=[354.24, 455.41, 358.26, 466.32] -->
$$ � $$  (eq. 4)

<!-- p7-b11 equation col=0 bbox=[72.72, 461.8, 125.29, 473.97] -->
$$ 𝑁_{𝑜𝑝𝑡}(𝐶) = 𝐺 $$  (eq. 4)

<!-- p7-b12 equation col=0 bbox=[517.75, 462.99, 532.91, 473.9] -->
$$ (4) $$  (eq. 4)

<!-- p7-b13 body col=0 bbox=[61.86, 492.72, 532.91, 505.2] -->
We show contours of the fitted function ^{ˆ}𝐿in Figure 4 (left), and the closed-form efficient computational

<!-- p7-b14 body col=0 bbox=[62.36, 506.65, 534.54, 518.82] -->
frontier in blue. From this approach, we find that 𝑎= 0.46 and 𝑏= 0.54—as summarized in Table 2.

<!-- p7-b15 heading col=0 bbox=[62.36, 541.33, 200.14, 552.24] -->
### 3.4. Optimal model scaling

<!-- p7-b16 body col=0 bbox=[61.86, 563.36, 532.91, 574.27] -->
We find that the three approaches, despite using different fitting methodologies and different trained

<!-- p7-b17 body col=0 bbox=[61.46, 576.91, 532.92, 601.37] -->
models, yield comparable predictions for the optimal scaling in parameters and tokens with FLOPs (shown in Table 2). All three approaches suggest that as compute budget increases, model size and

<!-- p7-b18 body col=0 bbox=[62.36, 604.01, 532.92, 628.47] -->
the amount of training data should be increased in approximately equal proportions. The first and second approaches yield very similar predictions for optimal model sizes, as shown in Figure 1 and

<!-- p7-b19 caption col=0 bbox=[61.86, 631.11, 534.74, 655.56] -->
> **Figure A3. The third approach predicts even smaller models being optimal at larger compute budgets. We note that the observed points (𝐿, 𝑁, 𝐷) for low training FLOPs (𝐶⩽1𝑒21) have larger residuals**

<!-- p7-b20 body col=0 bbox=[124.9, 655.84, 129.34, 663.81] -->
2

<!-- p7-b21 body col=0 bbox=[62.91, 658.48, 124.36, 670.71] -->
∥𝐿−^{ˆ}𝐿(𝑁, 𝐷)∥

<!-- p7-b22 body col=0 bbox=[61.94, 660.05, 533.1, 725.16] -->
_{2} ^{than} ^{points} ^{with} ^{higher} ^{computational} ^{budgets.} ^{The} ^{fitted} ^{model} ^{places} ^{increased} weight on the points with more FLOPs—automatically considering the low-computational budget points as outliers due to the Huber loss. As a consequence of the empirically observed negative curvature in the frontier 𝐶→𝑁_{𝑜𝑝𝑡}(see Appendix E), this results in predicting a lower 𝑁_{𝑜𝑝𝑡}than the two other approaches.

<!-- p7-b23 body col=0 bbox=[62.36, 734.57, 532.91, 759.03] -->
In Table 3 we show the estimated number of FLOPs and tokens that would ensure that a model of a given size lies on the compute-optimal frontier. Our findings suggests that the current generation of


<!-- ===== page 8 ===== -->

<!-- p8-b1 caption col=0 bbox=[62.03, 85.55, 533.1, 150.9] -->
> **Table 2 | Estimated parameter and data scaling with increased training compute. The listed values are the exponents, 𝑎and 𝑏, on the relationship 𝑁_{𝑜𝑝𝑡}∝𝐶^{𝑎}and 𝐷_{𝑜𝑝𝑡}∝𝐶^{𝑏}. Our analysis suggests a near equal scaling in parameters and data with increasing compute which is in clear contrast to previous work on the scaling of large models. The 10^{th} and 90^{th} percentiles are estimated via bootstrapping data (80% of the dataset is sampled 100 times) and are shown in parenthesis.**

<!-- T2 table page=8 bbox=[80.99, 161.18, 511.25, 249.67] -->
| Approach |  | Coeff. 𝑎 | where | 𝑁 𝑜𝑝𝑡 ∝ 𝐶^𝑎 | Coeff. | 𝑏 where | 𝐷 𝑜𝑝𝑡 ∝ 𝐶^𝑏 |
|---|---|---|---|---|---|---|---|
| 1. Minimum training | over curves | 0.50 | (0.488, | 0.502) | 0.50 | (0.501, | 0.512) |
| 2. IsoFLOP | profiles | 0.49 | (0.462, | 0.534) | 0.51 | (0.483, | 0.529) |
| 3. Parametric | modelling of the loss | 0.46 | (0.454, | 0.455) | 0.54 | (0.542, | 0.543) |
| Kaplan et al. | (2020) |  | 0.73 |  |  | 0.27 |  |

<!-- p8-b5 caption col=0 bbox=[62.03, 260.57, 533.12, 312.37] -->
> **Table 3 | Estimated optimal training FLOPs and training tokens for various model sizes. For various model sizes, we show the projections from Approach 1 of how many FLOPs and training tokens would be needed to train compute-optimal models. The estimates for Approach 2 & 3 are similar (shown in Section D.3)**

<!-- T3 table page=8 bbox=[139.04, 322.65, 462.81, 473.42] -->
|  | Parameters | FLOPs FLOPs | (in Gopher unit) |  | Tokens |
|---|---|---|---|---|---|
| 400 | Million | 1.92e+19 | 1/29, 968 | 8.0 | Billion |
| 1 | Billion | 1.21e+20 | 1/4, 761 | 20.2 | Billion |
| 10 | Billion | 1.23e+22 | 1/46 | 205.1 | Billion |
| 67 | Billion | 5.76e+23 | 1 | 1.5 | Trillion |
| 175 | Billion | 3.85e+24 | 6.7 | 3.7 | Trillion |
| 280 | Billion | 9.90e+24 | 17.2 | 5.9 | Trillion |
| 520 | Billion | 3.43e+25 | 59.5 | 11.0 | Trillion |
| 1 | Trillion | 1.27e+26 | 221.3 | 21.2 | Trillion |
| 10 | Trillion | 1.30e+28 | 22515.9 | 216.2 | Trillion |

<!-- p8-b8 body col=0 bbox=[132.46, 392.11, 135.5, 403.02] -->
.

<!-- p8-b9 body col=0 bbox=[62.06, 494.01, 534.43, 653.96] -->
large language models are considerably over-sized, given their respective compute budgets, as shown in Figure 1. For example, we find that a 175 billion parameter model should be trained with a compute budget of 4.41 × 10^{24} FLOPs and on over 4.2 trillion tokens. A 280 billion Gopher-like model is the optimal model to train given a compute budget of approximately 10^{25} FLOPs and should be trained on 6.8 trillion tokens. Unless one has a compute budget of 10^{26} FLOPs (over 250× the compute used to train Gopher), a 1 trillion parameter model is unlikely to be the optimal model to train. Furthermore, the amount of training data that is projected to be needed is far beyond what is currently used to train large models, and underscores the importance of dataset collection in addition to engineering improvements that allow for model scale. While there is significant uncertainty extrapolating out many orders of magnitude, our analysis clearly suggests that given the training compute budget for many current LLMs, smaller models should have been trained on more tokens to achieve the most performant model.

<!-- p8-b10 body col=0 bbox=[62.06, 663.37, 534.43, 701.38] -->
In Appendix C, we reproduce the IsoFLOP analysis on two additional datasets: C4 (Raffel et al., 2020a) and GitHub code (Rae et al., 2021). In both cases we reach the similar conclusion that model size and number of training tokens should be scaled in equal proportions.


<!-- ===== page 9 ===== -->

<!-- p9-b1 heading col=0 bbox=[62.36, 85.94, 140.65, 98.93] -->
## 4. Chinchilla

<!-- p9-b2 body col=0 bbox=[62.36, 112.58, 533.19, 191.23] -->
Based on our analysis in Section 3, the optimal model size for the Gopher compute budget is somewhere between 40 and 70 billion parameters. We test this hypothesis by training a model on the larger end of this range—70B parameters—for 1.4T tokens, due to both dataset and computational efficiency considerations. In this section we compare this model, which we call Chinchilla, to Gopher and other LLMs. Both Chinchilla and Gopher have been trained for the same number of FLOPs but differ in the size of the model and the number of training tokens.

<!-- p9-b3 body col=0 bbox=[62.36, 200.65, 534.65, 239.0] -->
While pre-training a large language model has a considerable compute cost, downstream finetuning and inference also make up substantial compute usage (Rae et al., 2021). Due to being 4× smaller than Gopher, both the memory footprint and inference cost of Chinchilla are also smaller.

<!-- p9-b4 heading col=0 bbox=[62.36, 261.48, 220.91, 272.39] -->
### 4.1. Model and training details

<!-- p9-b5 body col=0 bbox=[62.03, 283.5, 532.91, 308.31] -->
The full set of hyperparameters used to train Chinchilla are given in Table 4. Chinchilla uses the same model architecture and training setup as Gopher with the exception of the differences listed below.

<!-- p9-b6 body col=0 bbox=[77.74, 326.34, 534.48, 391.45] -->
• We train Chinchilla on MassiveText (the same dataset as Gopher) but use a slightly different subset distribution (shown in Table A1) to account for the increased number of training tokens. • We use AdamW (Loshchilov and Hutter, 2019) for Chinchilla rather than Adam (Kingma and Ba, 2014) as this improves the language modelling loss and the downstream task performance after finetuning.^{8}

<!-- p9-b7 body col=0 bbox=[77.74, 394.09, 533.81, 486.29] -->
• We train Chinchilla with a slightly modified SentencePiece (Kudo and Richardson, 2018) tokenizer that does not apply NFKC normalisation. The vocabulary is very similar– 94.15% of tokens are the same as those used for training Gopher. We find that this particularly helps with the representation of mathematics and chemistry, for example. • Whilst the forward and backward pass are computed in bfloat16, we store a float32 copy of the weights in the distributed optimiser state (Rajbhandari et al., 2020). See Lessons Learned from Rae et al. (2021) for additional details.

<!-- p9-b8 body col=0 bbox=[61.46, 504.67, 532.92, 556.23] -->
In Appendix G we show the impact of the various optimiser related changes between Chinchilla and Gopher. All models in this analysis have been trained on TPUv3/TPUv4 (Jouppi et al., 2017) with JAX (Bradbury et al., 2018) and Haiku (Hennigan et al., 2020). We include a Chinchilla model card (Mitchell et al., 2019) in Table A8.

<!-- T4 table page=9 bbox=[64.89, 574.22, 530.39, 630.14] -->
| Model | Layers | Number Heads | Key/Value Size | dmodel | Max LR | Batch | Size |
|---|---|---|---|---|---|---|---|
| Gopher 280B | 80 | 128 | 128 | 16,384 | 4 × 10^−5 | 3M | → 6M |
| Chinchilla 70B | 80 | 64 | 128 | 8,192 | 1 × 10^−4 | 1.5M | → 3M |

<!-- p9-b13 caption col=0 bbox=[62.03, 639.05, 534.73, 691.2] -->
> **Table 4 | Chinchilla architecture details. We list the number of layers, the key/value size, the bottleneck activation size d_{model}, the maximum learning rate, and the training batch size (# tokens). The feed-forward size is always set to 4 × d_{model}. Note that we double the batch size midway through training for both Chinchilla and Gopher.**

<!-- p9-b14 footnote col=0 bbox=[62.36, 736.82, 532.91, 758.64] -->
[^fn]: ^{8}Interestingly, a model trained with AdamW only passes the training performance of a model trained with Adam around 80% of the way through the cosine cycle, though the ending performance is notably better– see Figure A7


<!-- ===== page 10 ===== -->

<!-- T5 table page=10 bbox=[62.36, 83.47, 536.14, 193.59] -->
|  | # Tasks | Examples |
|---|---|---|
| Language Modelling | 20 | WikiText-103, The Pile: PG-19, arXiv, FreeLaw, . . . |
| Reading Comprehension | 3 | RACE-m, RACE-h, LAMBADA |
| Question Answering | 3 | Natural Questions, TriviaQA, TruthfulQA |
| Common Sense | 5 | HellaSwag, Winogrande, PIQA, SIQA, BoolQ |
| MMLU | 57 | High School Chemistry, Astronomy, Clinical Knowledge, . . . |
| BIG-bench | 62 Sequences, | Causal Judgement, Epistemic Reasoning, Temporal . . . |

<!-- p10-b3 caption col=0 bbox=[61.94, 202.49, 533.12, 240.75] -->
> **Table 5 | All evaluation tasks. We evaluate Chinchilla on a collection of language modelling along with downstream tasks. We evaluate on largely the same tasks as in Rae et al. (2021), to allow for direct comparison.**

<!-- p10-b4 heading col=0 bbox=[62.36, 265.08, 122.92, 275.99] -->
### 4.2. Results

<!-- p10-b5 body col=0 bbox=[61.86, 287.11, 534.73, 311.56] -->
We perform an extensive evaluation of Chinchilla, comparing against various large language models. We evaluate on a large subset of the tasks presented in Rae et al. (2021), shown in Table 5. As

<!-- p10-b6 body col=0 bbox=[62.36, 314.2, 532.91, 352.21] -->
the focus of this work is on optimal model scaling, we included a large representative subset, and introduce a few new evaluations to allow for better comparison to other existing large models. The evaluation details for all tasks are the same as described in Rae et al. (2021).

<!-- p10-b7 heading col=0 bbox=[62.36, 372.37, 196.79, 383.28] -->
#### 4.2.1. Language modelling

<!-- p10-b8 caption col=0 bbox=[62.03, 606.75, 534.42, 645.36] -->
> **Figure 5 | Pile Evaluation. For the different evaluation sets in The Pile (Gao et al., 2020), we show the bits-per-byte (bpb) improvement (decrease) of Chinchilla compared to Gopher. On all subsets, Chinchilla outperforms Gopher.**

<!-- p10-b9 body col=0 bbox=[62.06, 659.61, 534.43, 751.81] -->
Chinchilla significantly outperforms Gopher on all evaluation subsets of The Pile (Gao et al., 2020), as shown in Figure 5. Compared to Jurassic-1 (178B) Lieber et al. (2021), Chinchilla is more performant on all but two subsets– dm_mathematics and ubuntu_irc– see Table A5 for a raw bits-per-byte comparison. On Wikitext103 (Merity et al., 2017), Chinchilla achieves a perplexity of 7.16 compared to 7.75 for Gopher. Some caution is needed when comparing Chinchilla with Gopher on these language modelling benchmarks as Chinchilla is trained on 4× more data than Gopher and thus train/test set leakage may artificially enhance the results. We thus place more emphasis on other


<!-- ===== page 11 ===== -->

<!-- T6 table page=11 bbox=[184.74, 83.47, 410.54, 207.14] -->
| Random |  |  | 25.0% |
|---|---|---|---|
| Average |  | human rater | 34.5% |
| GPT-3 | 5-shot |  | 43.9% |
| Gopher | 5-shot |  | 60.0% |
|  | Chinchilla | 5-shot | 67.6% |
| Average |  | human expert | performance 89.8% |
| June | 2022 | Forecast | 57.1% |
| June | 2023 | Forecast | 63.4% |

<!-- p11-b3 caption col=0 bbox=[61.46, 216.04, 534.74, 254.3] -->
> **Table 6 | Massive Multitask Language Understanding (MMLU). We report the average 5-shot accuracy over 57 tasks with model and human accuracy comparisons taken from Hendrycks et al. (2020). We also include the average prediction for state of the art accuracy in June 2022/2023 made**

<!-- p11-b4 body col=0 bbox=[62.36, 256.94, 341.62, 267.85] -->
by 73 competitive human forecasters in Steinhardt (2021).

<!-- p11-b5 body col=0 bbox=[61.46, 291.89, 532.92, 316.35] -->
tasks for which leakage is less of a concern, such as MMLU (Hendrycks et al., 2020) and BIG-bench (BIG-bench collaboration, 2021) along with various closed-book question answering and common

<!-- p11-b6 body col=0 bbox=[62.36, 318.99, 133.71, 329.9] -->
sense analyses.

<!-- p11-b7 heading col=0 bbox=[62.36, 350.06, 129.18, 360.97] -->
#### 4.2.2. MMLU

<!-- p11-b8 body col=0 bbox=[62.03, 372.12, 536.06, 491.42] -->
The Massive Multitask Language Understanding (MMLU) benchmark (Hendrycks et al., 2020) consists of a range of exam-like questions on academic subjects. In Table 6, we report Chinchilla’s average 5-shot performance on MMLU (the full breakdown of results is shown in Table A6). On this benchmark, Chinchilla significantly outperforms Gopher despite being much smaller, with an average accuracy of 67.6% (improving upon Gopher by 7.6%). Remarkably, Chinchilla even outperforms the expert forecast for June 2023 of 63.4% accuracy (see Table 6) (Steinhardt, 2021). Furthermore, Chinchilla achieves greater than 90% accuracy on 4 different individual tasks– high_school_gov_and_politics, international_law, sociology, and us_foreign_policy. To our knowledge, no other model has achieved greater than 90% accuracy on a subset.

<!-- p11-b9 body col=0 bbox=[62.36, 500.84, 536.06, 552.4] -->
In Figure 6, we show a comparison to Gopher broken down by task. Overall, we find that Chinchilla improves performance on the vast majority of tasks. On four tasks (college_mathematics, econometrics, moral_scenarios, and formal_logic) Chinchilla underperforms Gopher, and there is no change in performance on two tasks.

<!-- p11-b10 heading col=0 bbox=[62.36, 572.56, 214.26, 583.47] -->
#### 4.2.3. Reading comprehension

<!-- p11-b11 body col=0 bbox=[61.96, 594.62, 533.38, 646.18] -->
On the final word prediction dataset LAMBADA (Paperno et al., 2016), Chinchilla achieves 77.4% accuracy, compared to 74.5% accuracy from Gopher and 76.6% from MT-NLG 530B (see Table 7). On RACE-h and RACE-m (Lai et al., 2017), Chinchilla greatly outperforms Gopher, improving accuracy by more than 10% in both cases—see Table 7.

<!-- p11-b12 heading col=0 bbox=[62.36, 666.34, 147.23, 677.25] -->
#### 4.2.4. BIG-bench

<!-- p11-b13 body col=0 bbox=[61.86, 688.4, 532.91, 699.64] -->
We analysed Chinchilla on the same set of BIG-bench tasks (BIG-bench collaboration, 2021) reported

<!-- p11-b14 body col=0 bbox=[62.03, 701.95, 536.06, 754.57] -->
in Rae et al. (2021). Similar to what we observed in MMLU, Chinchilla outperforms Gopher on the vast majority of tasks (see Figure 7). We find that Chinchilla improves the average performance by 10.7%, reaching an accuracy of 65.1% versus 54.4% for Gopher. Of the 62 tasks we consider, Chinchilla performs worse than Gopher on only four—crash_blossom, dark_humor_detection,


<!-- ===== page 12 ===== -->

<!-- p12-b1 caption col=0 bbox=[62.06, 315.12, 533.39, 353.37] -->
> **Figure 6 | MMLU results compared to Gopher We find that Chinchilla outperforms Gopher by 7.6% on average (see Table 6) in addition to performing better on 51/57 individual tasks, the same on 2/57, and worse on only 4/57 tasks.**

<!-- T7 table page=12 bbox=[130.13, 365.65, 465.14, 435.11] -->
|  | Chinchilla | Gopher | GPT-3 | MT-NLG 530B |
|---|---|---|---|---|
| LAMBADA Zero-Shot | 77.4 | 74.5 | 76.2 | 76.6 |
| RACE-m Few-Shot | 86.8 | 75.1 | 58.1 | - |
| RACE-h Few-Shot | 82.3 | 71.6 | 46.8 | 47.9 |

<!-- p12-b4 caption col=0 bbox=[61.46, 444.02, 533.26, 496.17] -->
> **Table 7 | Reading comprehension. On RACE-h and RACE-m (Lai et al., 2017), Chinchilla considerably improves performance over Gopher. Note that GPT-3 and MT-NLG 530B use a different prompt format than we do on RACE-h/m, so results are not comparable to Gopher and Chinchilla. On LAMBADA (Paperno et al., 2016), Chinchilla outperforms both Gopher and MT-NLG 530B.**

<!-- p12-b5 body col=0 bbox=[62.36, 519.84, 532.91, 544.32] -->
mathematical_induction and logical_args. Full accuracy results for Chinchilla can be found in Table A7.

<!-- p12-b6 heading col=0 bbox=[62.36, 564.48, 169.42, 575.39] -->
#### 4.2.5. Common sense

<!-- p12-b7 body col=0 bbox=[61.86, 586.55, 532.91, 597.82] -->
We evaluate Chinchilla on various common sense benchmarks: PIQA (Bisk et al., 2020), SIQA (Sap

<!-- p12-b8 body col=0 bbox=[61.46, 600.1, 533.47, 624.93] -->
et al., 2019), Winogrande (Sakaguchi et al., 2020), HellaSwag (Zellers et al., 2019), and BoolQ (Clark et al., 2019). We find that Chinchilla outperforms both Gopher and GPT-3 on all tasks and

<!-- p12-b9 body col=0 bbox=[62.36, 627.2, 346.66, 638.1] -->
outperforms MT-NLG 530B on all but one task—see Table 8.

<!-- p12-b10 body col=0 bbox=[61.46, 647.52, 533.38, 699.08] -->
On TruthfulQA (Lin et al., 2021), Chinchilla reaches 43.6%, 58.5%, and 66.7% accuracy with 0-shot, 5-shot, and 10-shot respectively. In comparison, Gopher achieved only 29.5% 0-shot and 43.7% 10-shot accuracy. In stark contrast with the findings of Lin et al. (2021), the large improvements (14.1% in 0-shot accuracy) achieved by Chinchilla suggest that better modelling of the pre-training

<!-- p12-b11 body col=0 bbox=[62.36, 701.72, 386.11, 712.63] -->
data alone can lead to substantial improvements on this benchmark.


<!-- ===== page 13 ===== -->

<!-- p13-b1 caption col=0 bbox=[62.36, 319.85, 533.12, 344.56] -->
> **Figure 7 | BIG-bench results compared to Gopher Chinchilla out performs Gopher on all but four BIG-bench tasks considered. Full results are in Table A7.**

<!-- p13-b2 heading col=0 bbox=[62.36, 368.85, 254.91, 379.76] -->
#### 4.2.6. Closed-book question answering

<!-- p13-b3 body col=0 bbox=[61.46, 390.91, 534.43, 496.67] -->
Results on closed-book question answering benchmarks are reported in Table 9. On the Natural Questions dataset (Kwiatkowski et al., 2019), Chinchilla achieves new closed-book SOTA accuracies: 31.5% 5-shot and 35.5% 64-shot, compared to 21% and 28% respectively, for Gopher. On TriviaQA (Joshi et al., 2017) we show results for both the filtered (previously used in retrieval and open-book work) and unfiltered set (previously used in large language model evaluations). In both cases, Chinchilla substantially out performs Gopher. On the filtered version, Chinchilla lags behind the open book SOTA (Izacard and Grave, 2020) by only 7.9%. On the unfiltered set, Chinchilla outperforms GPT-3—see Table 9.

<!-- p13-b4 heading col=0 bbox=[62.36, 516.83, 217.7, 527.74] -->
#### 4.2.7. Gender bias and toxicity

<!-- p13-b5 body col=0 bbox=[62.36, 538.89, 534.43, 577.27] -->
Large Language Models carry potential risks such as outputting offensive language, propagating social biases, and leaking private information (Bender et al., 2021; Weidinger et al., 2021). We expect Chinchilla to carry risks similar to Gopher because Chinchilla is trained on the same data,

<!-- T8 table page=13 bbox=[103.27, 595.44, 492.0, 692.01] -->
|  | Chinchilla | Gopher | GPT-3 | MT-NLG 530B | Supervised SOTA |
|---|---|---|---|---|---|
| HellaSWAG | 80.8% | 79.2% | 78.9% | 80.2% | 93.9% |
| PIQA | 81.8% | 81.8% | 81.0% | 82.0% | 90.1% |
| Winogrande | 74.9% | 70.1% | 70.2% | 73.0% | 91.3% |
| SIQA | 51.3% | 50.6% | - | - | 83.2% |
| BoolQ | 83.7% | 79.3% | 60.5% | 78.2% | 91.4% |

<!-- p13-b8 caption col=0 bbox=[62.03, 700.91, 532.92, 752.72] -->
> **Table 8 | Zero-shot comparison on Common Sense benchmarks. We show a comparison between Chinchilla, Gopher, and MT-NLG 530B on various Common Sense benchmarks. We see that Chinchilla matches or outperforms Gopher and GPT-3 on all tasks. On all but one Chinchilla outperforms the much larger MT-NLG 530B model.**


<!-- ===== page 14 ===== -->

<!-- T9 table page=14 bbox=[81.6, 83.47, 513.68, 245.18] -->
|  | Method | Chinchilla | Gopher | GPT-3 | SOTA (open book) |
|---|---|---|---|---|---|
|  | 0-shot | 16.6% | 10.1% | 14.6% |  |
| Natural Questions (dev) | 5-shot | 31.5% | 24.5% | - | 54.4% |
|  | 64-shot | 35.5% | 28.2% | 29.9% |  |
|  | 0-shot | 67.0% | 52.8% | 64.3 % |  |
| TriviaQA (unfiltered, test) | 5-shot | 73.2% | 63.6% | - | - |
|  | 64-shot | 72.3% | 61.3% | 71.2% |  |
|  | 0-shot | 55.4% | 43.5% | - |  |
| TriviaQA (filtered, dev) | 5-shot | 64.1% | 57.0% | - | 72.5% |
|  | 64-shot | 64.6% | 57.2% | - |  |

<!-- p14-b9 caption col=0 bbox=[62.03, 254.08, 534.43, 305.89] -->
> **Table 9 | Closed-book question answering. For Natural Questions (Kwiatkowski et al., 2019) and TriviaQA (Joshi et al., 2017), Chinchilla outperforms Gopher in all cases. On Natural Questions, Chinchilla outperforms GPT-3. On TriviaQA we show results on two different evaluation sets to allow for comparison to GPT-3 and to open book SOTA (FiD + Distillation (Izacard and Grave, 2020)).**

<!-- p14-b10 body col=0 bbox=[62.36, 329.93, 533.1, 381.49] -->
albeit with slightly different relative weights, and because it has a similar architecture. Here, we examine gender bias (particularly gender and occupation bias) and generation of toxic language. We select a few common evaluations to highlight potential issues, but stress that our evaluations are not comprehensive and much work remains to understand, evaluate, and mitigate risks in LLMs.

<!-- p14-b11 body col=0 bbox=[61.46, 406.17, 534.73, 511.92] -->
Gender bias. As discussed in Rae et al. (2021), large language models reflect contemporary and historical discourse about different groups (such as gender groups) from their training dataset, and we expect the same to be true for Chinchilla. Here, we test if potential gender and occupation biases manifest in unfair outcomes on coreference resolutions, using the Winogender dataset (Rudinger et al., 2018) in a zero-shot setting. Winogender tests whether a model can correctly determine if a pronoun refers to different occupation words. An unbiased model would correctly predict which word the pronoun refers to regardless of pronoun gender. We follow the same setup as in Rae et al. (2021) (described further in Section H.3).

<!-- p14-b12 body col=0 bbox=[61.46, 521.33, 533.19, 586.81] -->
As shown in Table 10, Chinchilla correctly resolves pronouns more frequently than Gopher across all groups. Interestingly, the performance increase is considerably smaller for male pronouns (increase of 3.2%) than for female or neutral pronouns (increases of 8.3% and 9.2% respectively). We also consider gotcha examples, in which the correct pronoun resolution contradicts gender stereotypes (determined by labor statistics). Again, we see that Chinchilla resolves pronouns more accurately

<!-- p14-b13 body col=0 bbox=[62.36, 589.08, 533.19, 654.18] -->
than Gopher. When breaking up examples by male/female gender and gotcha/not gotcha, the largest improvement is on female gotcha examples (improvement of 10%). Thus, though Chinchilla uniformly overcomes gender stereotypes for more coreference examples than Gopher, the rate of improvement is higher for some pronouns than others, suggesting that the improvements conferred by using a more compute-optimal model can be uneven.

<!-- p14-b14 body col=0 bbox=[62.36, 678.86, 534.74, 757.52] -->
Sample toxicity. Language models are capable of generating toxic language—including insults, hate speech, profanities and threats (Gehman et al., 2020; Rae et al., 2021). While toxicity is an umbrella term, and its evaluation in LMs comes with challenges (Welbl et al., 2021; Xu et al., 2021), automatic classifier scores can provide an indication for the levels of harmful text that a LM generates. Rae et al. (2021) found that improving language modelling loss by increasing the number of model parameters has only a negligible effect on toxic text generation (unprompted); here we analyze


<!-- ===== page 15 ===== -->

<!-- T10 table page=15 bbox=[90.53, 83.47, 242.91, 166.49] -->
|  | Chinchilla | Gopher |
|---|---|---|
| All | 78.3% | 71.4% |
| Male | 71.2% | 68.0% |
| Female | 79.6% | 71.3% |
| Neutral | 84.2% | 75.0% |

<!-- T11 table page=15 bbox=[312.86, 83.47, 514.66, 166.49] -->
|  | Chinchilla | Gopher |
|---|---|---|
| Male gotcha | 62.5% | 59.2% |
| Male not gotcha | 80.0% | 76.7% |
| Female gotcha | 76.7% | 66.7% |
| Female not gotcha | 82.5% | 75.8% |

<!-- p15-b5 caption col=0 bbox=[62.03, 175.39, 534.74, 214.0] -->
> **Table 10 | Winogender results. Left: Chinchilla consistently resolves pronouns better than Gopher. Right: Chinchilla performs better on examples which contradict gender stereotypes (gotcha examples). However, difference in performance across groups suggests Chinchilla exhibits bias.**

<!-- p15-b6 body col=0 bbox=[61.94, 237.69, 533.12, 370.54] -->
whether the same holds true for a lower LM loss achieved via more compute-optimal training. Similar to the protocol of Rae et al. (2021), we generate 25,000 unprompted samples from Chinchilla, and compare their PerspectiveAPI toxicity score distribution to that of Gopher-generated samples. Several summary statistics indicate an absence of major differences: the mean (median) toxicity score for Gopher is 0.081 (0.064), compared to 0.087 (0.066) for Chinchilla, and the 95^{th} percentile scores are 0.230 for Gopher, compared to 0.238 for Chinchilla. That is, the large majority of generated samples are classified as non-toxic, and the difference between the models is negligible. In line with prior findings (Rae et al., 2021), this suggests that toxicity levels in unconditional text generation are largely independent of the model quality (measured in language modelling loss), i.e. that better models of the training dataset are not necessarily more toxic.

<!-- p15-b7 heading col=0 bbox=[62.36, 394.81, 226.34, 407.76] -->
## 5. Discussion & Conclusion

<!-- p15-b8 body col=0 bbox=[62.03, 421.4, 534.44, 527.16] -->
The trend so far in large language model training has been to increase the model size, often without increasing the number of training tokens. The largest dense transformer, MT-NLG 530B, is now over 3× larger than GPT-3’s 170 billion parameters from just two years ago. However, this model, as well as the majority of existing large models, have all been trained for a comparable number of tokens—around 300 billion. While the desire to train these mega-models has led to substantial engineering innovation, we hypothesize that the race to train larger and larger models is resulting in models that are substantially underperforming compared to what could be achieved with the same compute budget.

<!-- p15-b9 body col=0 bbox=[62.36, 536.57, 534.65, 615.23] -->
We propose three predictive approaches towards optimally setting model size and training duration, based on the outcome of over 400 training runs. All three approaches predict that Gopher is substantially over-sized and estimate that for the same compute budget a smaller model trained on more data will perform better. We directly test this hypothesis by training Chinchilla, a 70B parameter model, and show that it outperforms Gopher and even larger models on nearly every measured evaluation task.

<!-- p15-b10 body col=0 bbox=[62.36, 624.64, 533.18, 711.11] -->
Whilst our method allows us to make predictions on how to scale large models when given additional compute, there are several limitations. Due to the cost of training large models, we only have two comparable training runs at large scale (Chinchilla and Gopher), and we do not have additional tests at intermediate scales. Furthermore, we assume that the efficient computational frontier can be described by a power-law relationship between the compute budget, model size, and number of training tokens. However, we observe some concavity in log ^{�}

<!-- p15-b11 body col=0 bbox=[393.82, 691.2, 411.29, 702.84] -->
𝑁𝑜𝑝𝑡

<!-- p15-b12 body col=0 bbox=[61.46, 691.88, 534.74, 716.85] -->
�at high compute budgets (see Appendix E). This suggests that we may still be overestimating the optimal size of large models.

<!-- p15-b13 body col=0 bbox=[61.94, 719.49, 532.92, 757.83] -->
Finally, the training runs for our analysis have all been trained on less than an epoch of data; future work may consider the multiple epoch regime. Despite these limitations, the comparison of Chinchilla to Gopher validates our performance predictions, that have thus enabled training a better (and more


<!-- ===== page 16 ===== -->

<!-- p16-b1 body col=0 bbox=[62.36, 87.27, 294.11, 98.18] -->
lightweight) model at the same compute budget.

<!-- p16-b2 body col=0 bbox=[62.36, 107.59, 534.43, 267.54] -->
Though there has been significant recent work allowing larger and larger models to be trained, our analysis suggests an increased focus on dataset scaling is needed. Speculatively, we expect that scaling to larger and larger datasets is only beneficial when the data is high-quality. This calls for responsibly collecting larger datasets with a high focus on dataset quality. Larger datasets will require extra care to ensure train-test set overlap is properly accounted for, both in the language modelling loss but also with downstream tasks. Finally, training for trillions of tokens introduces many ethical and privacy concerns. Large datasets scraped from the web will contain toxic language, biases, and private information. With even larger datasets being used, the quantity (if not the frequency) of such information increases, which makes dataset introspection all the more important. Chinchilla does suffer from bias and toxicity but interestingly it seems less affected than Gopher. Better understanding how performance of large language models and toxicity interact is an important future research question.

<!-- p16-b3 body col=0 bbox=[61.94, 276.96, 534.43, 328.52] -->
While we have applied our methodology towards the training of auto-regressive language models, we expect that there is a similar trade-off between model size and the amount of data in other modalities. As training large models is very expensive, choosing the optimal model size and training steps beforehand is essential. The methods we propose are easy to reproduce in new settings.

<!-- p16-b4 heading col=0 bbox=[62.36, 352.78, 195.79, 365.73] -->
## 6. Acknowledgements

<!-- p16-b5 body col=0 bbox=[61.86, 379.38, 534.43, 444.48] -->
We’d like to thank Jean-baptiste Alayrac, Kareem Ayoub, Chris Dyer, Nando de Freitas, Demis Hassabis, Geoffrey Irving, Koray Kavukcuoglu, Nate Kushman and Angeliki Lazaridou for useful comments on the manuscript. We’d like to thank Andy Brock, Irina Higgins, Michela Paganini, Francis Song, and other colleagues at DeepMind for helpful discussions. We are also very grateful to the JAX and XLA team for their support and assistance.

<!-- p16-b6 heading col=0 bbox=[62.36, 468.75, 126.98, 481.7] -->
## References

<!-- p16-b7 reference col=0 bbox=[62.36, 495.34, 534.43, 506.25] -->
- M. Artetxe, S. Bhosale, N. Goyal, T. Mihaylov, M. Ott, S. Shleifer, X. V. Lin, J. Du, S. Iyer, R. Pasunuru,

<!-- p16-b8 reference col=0 bbox=[72.73, 508.89, 534.43, 546.9] -->
- G. Anantharaman, X. Li, S. Chen, H. Akin, M. Baines, L. Martin, X. Zhou, P. S. Koura, B. O’Horo, J. Wang, L. Zettlemoyer, M. Diab, Z. Kozareva, and V. Stoyanov. Efficient Large Scale Language Modeling with Mixtures of Experts. arXiv:2112.10684, 2021.

<!-- p16-b9 reference col=0 bbox=[62.36, 558.51, 534.31, 569.41] -->
- E. M. Bender, T. Gebru, A. McMillan-Major, and S. Shmitchell. On the dangers of stochastic parrots:

<!-- p16-b10 reference col=0 bbox=[72.86, 572.05, 534.43, 596.51] -->
- Can language models be too big? In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency, pages 610–623, 2021.

<!-- p16-b11 reference col=0 bbox=[62.36, 608.12, 532.91, 619.03] -->
- BIG-bench collaboration. Beyond the imitation game: Measuring and extrapolating the capabilities of

<!-- p16-b12 reference col=0 bbox=[73.21, 621.64, 520.02, 633.64] -->
- language models. In preparation, 2021. URL https://github.com/google/BIG-bench/.

<!-- p16-b13 reference col=0 bbox=[62.36, 644.18, 532.91, 655.09] -->
- Y. Bisk, R. Zellers, J. Gao, Y. Choi, et al. PIQA: Reasoning about physical commonsense in natural

<!-- p16-b14 reference col=0 bbox=[72.9, 657.73, 532.92, 682.19] -->
- language. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 7432–7439, 2020.

<!-- p16-b15 reference col=0 bbox=[62.36, 693.8, 534.74, 704.71] -->
- S. Borgeaud, A. Mensch, J. Hoffmann, T. Cai, E. Rutherford, K. Millican, G. van den Driessche, J.-B.

<!-- p16-b16 reference col=0 bbox=[73.21, 707.35, 534.43, 758.9] -->
- Lespiau, B. Damoc, A. Clark, D. de Las Casas, A. Guy, J. Menick, R. Ring, T. Hennigan, S. Huang, L. Maggiore, C. Jones, A. Cassirer, A. Brock, M. Paganini, G. Irving, O. Vinyals, S. Osindero, K. Simonyan, J. W. Rae, E. Elsen, and L. Sifre. Improving language models by retrieving from trillions of tokens. arXiv 2112.04426, 2021.


<!-- ===== page 17 ===== -->

<!-- p17-b1 reference col=0 bbox=[62.36, 87.27, 534.64, 98.18] -->
- J. Bradbury, R. Frostig, P. Hawkins, M. J. Johnson, C. Leary, D. Maclaurin, G. Necula, A. Paszke, J. Van-

<!-- p17-b2 reference col=0 bbox=[73.21, 100.82, 533.19, 126.34] -->
- derPlas, S. Wanderman-Milne, and Q. Zhang. JAX: composable transformations of Python+NumPy programs. 2018. URL http://github.com/google/jax.

<!-- p17-b3 reference col=0 bbox=[62.36, 136.26, 534.43, 147.17] -->
- T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam,

<!-- p17-b4 reference col=0 bbox=[72.83, 149.81, 534.43, 243.08] -->
- G. Sastry, A. Askell, S. Agarwal, A. Herbert-Voss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. Ziegler, J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray, B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever, and D. Amodei. Language models are few-shot learners. In H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 1877–1901. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb49674 18bfb8ac142f64a-Paper.pdf.

<!-- p17-b5 reference col=0 bbox=[62.36, 253.0, 532.91, 263.91] -->
- S. Bubeck. Convex Optimization: Algorithms and Complexity. Foundations and Trends in Machine

<!-- p17-b6 reference col=0 bbox=[73.21, 266.52, 532.91, 292.07] -->
- Learning, 8(3-4):231–357, 2015. URL http://www.nowpublishers.com/article/Detail s/MAL-050.

<!-- p17-b7 reference col=0 bbox=[62.36, 301.99, 534.43, 312.9] -->
- A. Clark, D. d. l. Casas, A. Guy, A. Mensch, M. Paganini, J. Hoffmann, B. Damoc, B. Hechtman,

<!-- p17-b8 reference col=0 bbox=[72.86, 315.54, 534.43, 368.16] -->
- T. Cai, S. Borgeaud, G. v. d. Driessche, E. Rutherford, T. Hennigan, M. Johnson, K. Millican, A. Cassirer, C. Jones, E. Buchatskaya, D. Budden, L. Sifre, S. Osindero, O. Vinyals, J. Rae, E. Elsen, K. Kavukcuoglu, and K. Simonyan. Unified scaling laws for routed language models, 2022. URL https://arxiv.org/abs/2202.01169.

<!-- p17-b9 reference col=0 bbox=[62.36, 378.08, 532.91, 388.99] -->
- C. Clark, K. Lee, M.-W. Chang, T. Kwiatkowski, M. Collins, and K. Toutanova. Boolq: Exploring

<!-- p17-b10 reference col=0 bbox=[73.21, 391.63, 532.91, 429.64] -->
- the surprising difficulty of natural yes/no questions. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2924–2936, 2019.

<!-- p17-b11 reference col=0 bbox=[62.36, 440.62, 534.43, 451.53] -->
- N. Du, Y. Huang, A. M. Dai, S. Tong, D. Lepikhin, Y. Xu, M. Krikun, Y. Zhou, A. W. Yu, O. Firat, B. Zoph,

<!-- p17-b12 reference col=0 bbox=[73.21, 454.17, 534.65, 493.24] -->
- L. Fedus, M. Bosma, Z. Zhou, T. Wang, Y. E. Wang, K. Webster, M. Pellat, K. Robinson, K. Meier- Hellstern, T. Duke, L. Dixon, K. Zhang, Q. V. Le, Y. Wu, Z. Chen, and C. Cui. Glam: Efficient scaling of language models with mixture-of-experts, 2021. URL https://arxiv.org/abs/2112.06905.

<!-- p17-b13 reference col=0 bbox=[62.36, 503.16, 532.91, 514.07] -->
- W. Fedus, B. Zoph, and N. Shazeer. Switch transformers: Scaling to trillion parameter models with

<!-- p17-b14 reference col=0 bbox=[73.21, 516.71, 400.06, 527.62] -->
- simple and efficient sparsity. arXiv preprint arXiv:2101.03961, 2021.

<!-- p17-b15 reference col=0 bbox=[62.36, 538.61, 534.43, 549.52] -->
- L. Gao, S. Biderman, S. Black, L. Golding, T. Hoppe, C. Foster, J. Phang, H. He, A. Thite, N. Nabeshima,

<!-- p17-b16 reference col=0 bbox=[73.21, 552.16, 533.19, 576.61] -->
- S. Presser, and C. Leahy. The Pile: An 800GB dataset of diverse text for language modeling. arXiv preprint arXiv:2101.00027, 2020.

<!-- p17-b17 reference col=0 bbox=[62.36, 587.6, 532.91, 598.51] -->
- S. Gehman, S. Gururangan, M. Sap, Y. Choi, and N. A. Smith. RealToxicityPrompts: Evaluating

<!-- p17-b18 reference col=0 bbox=[73.21, 601.15, 532.92, 653.77] -->
- neural toxic degeneration in language models. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 3356–3369, Online, Nov. 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.findings-emnlp.301. URL https://aclanthology.org/2 020.findings-emnlp.301.

<!-- p17-b19 reference col=0 bbox=[62.36, 663.69, 532.91, 674.6] -->
- K. Guu, K. Lee, Z. Tung, P. Pasupat, and M.-W. Chang. REALM: Retrieval-augmented language model

<!-- p17-b20 reference col=0 bbox=[73.21, 677.24, 163.13, 688.15] -->
- pre-training, 2020.

<!-- p17-b21 reference col=0 bbox=[62.36, 699.13, 532.91, 710.04] -->
- D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt. Measuring massive

<!-- p17-b22 reference col=0 bbox=[73.21, 712.68, 431.58, 723.59] -->
- multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.

<!-- p17-b23 reference col=0 bbox=[62.36, 734.55, 536.07, 746.55] -->
- T. Hennigan, T. Cai, T. Norman, and I. Babuschkin. Haiku: Sonnet for JAX. 2020. URL http:

<!-- p17-b24 reference col=0 bbox=[72.58, 748.1, 265.11, 760.1] -->
- //github.com/deepmind/dm-haiku.


<!-- ===== page 18 ===== -->

<!-- p18-b1 reference col=0 bbox=[62.36, 87.27, 488.85, 98.18] -->
- D. Hernandez, J. Kaplan, T. Henighan, and S. McCandlish. Scaling laws for transfer, 2021.

<!-- p18-b2 reference col=0 bbox=[62.36, 109.79, 532.91, 120.7] -->
- P. J. Huber. Robust Estimation of a Location Parameter. The Annals of Mathematical Statistics, 35

<!-- p18-b3 reference col=0 bbox=[72.3, 123.34, 533.2, 134.24] -->
- (1):73–101, Mar. 1964. ISSN 0003-4851, 2168-8990. doi: 10.1214/aoms/1177703732. URL

<!-- p18-b4 reference col=0 bbox=[72.89, 136.86, 533.23, 175.96] -->
- https://projecteuclid.org/journals/annals-of-mathematical-statistics/vol ume-35/issue-1/Robust-Estimation-of-a-Location-Parameter/10.1214/aoms/11 77703732.full.

<!-- p18-b5 reference col=0 bbox=[62.36, 186.5, 534.73, 197.41] -->
- G. Izacard and E. Grave. Distilling knowledge from reader to retriever for question answering, 2020.

<!-- p18-b6 reference col=0 bbox=[62.36, 209.01, 532.91, 219.92] -->
- M. Joshi, E. Choi, D. Weld, and L. Zettlemoyer. TriviaQA: A Large Scale Distantly Supervised Challenge

<!-- p18-b7 reference col=0 bbox=[73.21, 222.56, 459.79, 233.47] -->
- Dataset for Reading Comprehension. arXiv e-prints, art. arXiv:1705.03551, 2017.

<!-- p18-b8 reference col=0 bbox=[62.36, 245.08, 534.43, 255.99] -->
- N. P. Jouppi, C. Young, N. Patil, D. Patterson, G. Agrawal, R. Bajwa, S. Bates, S. Bhatia, N. Boden,

<!-- p18-b9 reference col=0 bbox=[72.71, 258.63, 534.74, 364.38] -->
- A. Borchers, R. Boyle, P.-l. Cantin, C. Chao, C. Clark, J. Coriell, M. Daley, M. Dau, J. Dean, B. Gelb, T. V. Ghaemmaghami, R. Gottipati, W. Gulland, R. Hagmann, C. R. Ho, D. Hogberg, J. Hu, R. Hundt, D. Hurt, J. Ibarz, A. Jaffey, A. Jaworski, A. Kaplan, H. Khaitan, D. Killebrew, A. Koch, N. Kumar, S. Lacy, J. Laudon, J. Law, D. Le, C. Leary, Z. Liu, K. Lucke, A. Lundin, G. MacKean, A. Maggiore, M. Mahony, K. Miller, R. Nagarajan, R. Narayanaswami, R. Ni, K. Nix, T. Norrie, M. Omernick, N. Penukonda, A. Phelps, J. Ross, M. Ross, A. Salek, E. Samadiani, C. Severn, G. Sizikov, M. Snelham, J. Souter, D. Steinberg, A. Swing, M. Tan, G. Thorson, B. Tian, H. Toma, E. Tuttle, V. Vasudevan, R. Walter, W. Wang, E. Wilcox, and D. H. Yoon. In-datacenter performance analysis of a tensor processing unit.

<!-- p18-b10 reference col=0 bbox=[73.21, 367.02, 534.74, 406.09] -->
- In Proceedings of the 44th Annual International Symposium on Computer Architecture, ISCA ’17, page 1–12, New York, NY, USA, 2017. Association for Computing Machinery. ISBN 9781450348928. doi: 10.1145/3079856.3080246. URL https://doi.org/10.1145/3079856.3080246.

<!-- p18-b11 reference col=0 bbox=[62.36, 416.64, 534.43, 427.55] -->
- J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu,

<!-- p18-b12 reference col=0 bbox=[73.21, 430.19, 534.62, 441.09] -->
- and D. Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

<!-- p18-b13 reference col=0 bbox=[62.36, 452.7, 534.43, 463.61] -->
- D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980,

<!-- p18-b14 reference col=0 bbox=[72.9, 466.25, 100.2, 477.16] -->
- 2014.

<!-- p18-b15 reference col=0 bbox=[62.36, 488.77, 533.13, 499.67] -->
- T. Kudo and J. Richardson. SentencePiece: A simple and language independent subword tokenizer

<!-- p18-b16 reference col=0 bbox=[73.21, 502.31, 469.58, 513.22] -->
- and detokenizer for neural text processing. arXiv preprint arXiv:1808.06226, 2018.

<!-- p18-b17 reference col=0 bbox=[62.36, 524.83, 534.43, 535.74] -->
- T. Kwiatkowski, J. Palomaki, O. Redfield, M. Collins, A. Parikh, C. Alberti, D. Epstein, I. Polosukhin,

<!-- p18-b18 reference col=0 bbox=[73.21, 538.38, 532.92, 576.39] -->
- M. Kelcey, J. Devlin, K. Lee, K. N. Toutanova, L. Jones, M.-W. Chang, A. Dai, J. Uszkoreit, Q. Le, and S. Petrov. Natural questions: a benchmark for question answering research. Transactions of the Association of Computational Linguistics, 2019.

<!-- p18-b19 reference col=0 bbox=[62.36, 587.99, 532.92, 598.9] -->
- G. Lai, Q. Xie, H. Liu, Y. Yang, and E. Hovy. RACE: Large-scale ReAding comprehension dataset from

<!-- p18-b20 reference col=0 bbox=[73.21, 601.54, 532.91, 640.61] -->
- examinations. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pages 785–794, Copenhagen, Denmark, Sept. 2017. Association for Computational Linguistics. doi: 10.18653/v1/D17-1082. URL https://aclanthology.org/D17-1082.

<!-- p18-b21 reference col=0 bbox=[62.36, 651.16, 534.74, 662.07] -->
- Y. Levine, N. Wies, O. Sharir, H. Bata, and A. Shashua. The depth-to-width interplay in self-attention.

<!-- p18-b22 reference col=0 bbox=[73.21, 664.71, 261.59, 675.61] -->
- arXiv preprint arXiv:2006.12467, 2020.

<!-- p18-b23 reference col=0 bbox=[62.36, 687.22, 534.43, 698.13] -->
- P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W.-t. Yih,

<!-- p18-b24 reference col=0 bbox=[72.88, 700.77, 534.43, 738.78] -->
- T. Rocktäschel, S. Riedel, and D. Kiela. Retrieval-augmented generation for knowledge-intensive nlp tasks. In Advances in Neural Information Processing Systems, volume 33, pages 9459–9474, 2020.


<!-- ===== page 19 ===== -->

<!-- p19-b1 reference col=0 bbox=[62.36, 87.27, 532.91, 98.18] -->
- O. Lieber, O. Sharir, B. Lenz, and Y. Shoham. Jurassic-1: Technical details and evaluation. White

<!-- p19-b2 reference col=0 bbox=[73.21, 100.82, 185.02, 111.73] -->
- Paper. AI21 Labs, 2021.

<!-- p19-b3 reference col=0 bbox=[62.36, 123.15, 533.19, 134.06] -->
- S. Lin, J. Hilton, and O. Evans. TruthfulQA: Measuring how models mimic human falsehoods. arXiv

<!-- p19-b4 reference col=0 bbox=[73.21, 136.7, 233.93, 147.61] -->
- preprint arXiv:2109.07958, 2021.

<!-- p19-b5 reference col=0 bbox=[62.36, 159.02, 532.91, 169.93] -->
- I. Loshchilov and F. Hutter. Decoupled weight decay regularization. In International Conference on

<!-- p19-b6 reference col=0 bbox=[73.21, 172.54, 522.53, 184.54] -->
- Learning Representations, 2019. URL https://openreview.net/forum?id=Bkg6RiCqY7.

<!-- p19-b7 reference col=0 bbox=[62.36, 194.9, 534.43, 205.81] -->
- S. McCandlish, J. Kaplan, D. Amodei, and O. D. Team. An empirical model of large-batch training,

<!-- p19-b8 reference col=0 bbox=[72.9, 208.45, 100.2, 219.36] -->
- 2018.

<!-- p19-b9 reference col=0 bbox=[62.36, 230.78, 532.91, 241.69] -->
- S. Merity, C. Xiong, J. Bradbury, and R. Socher. Pointer sentinel mixture models. International

<!-- p19-b10 reference col=0 bbox=[73.21, 244.33, 298.74, 255.23] -->
- Conference on Learning Representations, 2017.

<!-- p19-b11 reference col=0 bbox=[62.36, 266.65, 534.64, 277.56] -->
- M. Mitchell, S. Wu, A. Zaldivar, P. Barnes, L. Vasserman, B. Hutchinson, E. Spitzer, I. D. Raji, and T. Ge-

<!-- p19-b12 reference col=0 bbox=[73.21, 280.2, 534.43, 304.66] -->
- bru. Model cards for model reporting. In Proceedings of the conference on fairness, accountability, and transparency, pages 220–229, 2019.

<!-- p19-b13 reference col=0 bbox=[62.36, 316.08, 534.43, 326.99] -->
- J. Nocedal. Updating Quasi-Newton Matrices with Limited Storage. Mathematics of Computation,

<!-- p19-b14 reference col=0 bbox=[72.9, 329.6, 537.32, 355.15] -->
- 35(151):773–782, 1980. ISSN 0025-5718. doi: 10.2307/2006193. URL https://www.jstor. org/stable/2006193.

<!-- p19-b15 reference col=0 bbox=[62.36, 365.5, 534.43, 376.41] -->
- D. Paperno, G. Kruszewski, A. Lazaridou, Q. N. Pham, R. Bernardi, S. Pezzelle, M. Baroni, G. Boleda,

<!-- p19-b16 reference col=0 bbox=[72.9, 379.05, 534.43, 403.51] -->
- and R. Fernández. The LAMBADA dataset: Word prediction requiring a broad discourse context, 2016.

<!-- p19-b17 reference col=0 bbox=[62.36, 414.93, 534.43, 425.84] -->
- J. Rae, S. Borgeaud, T. Cai, K. Millican, J. Hoffmann, F. Song, J. Aslanides, S. Henderson, R. Ring,

<!-- p19-b18 reference col=0 bbox=[72.88, 428.48, 534.74, 561.33] -->
- S. Young, E. Rutherford, T. Hennigan, J. Menick, A. Cassirer, R. Powell, G. van den Driessche, L. A. Hendricks, M. Rauh, P.-S. Huang, A. Glaese, J. Welbl, S. Dathathri, S. Huang, J. Uesato, J. Mellor, I. Higgins, A. Creswell, N. McAleese, A. Wu, E. Elsen, S. Jayakumar, E. Buchatskaya, D. Budden, E. Sutherland, K. Simonyan, M. Paganini, L. Sifre, L. Martens, X. L. Li, A. Kuncoro, A. Nematzadeh, E. Gribovskaya, D. Donato, A. Lazaridou, A. Mensch, J.-B. Lespiau, M. Tsimpoukelli, N. Grigorev, D. Fritz, T. Sottiaux, M. Pajarskas, T. Pohlen, Z. Gong, D. Toyama, C. de Masson d’Autume, Y. Li, T. Terzi, I. Babuschkin, A. Clark, D. de Las Casas, A. Guy, J. Bradbury, M. Johnson, L. Weidinger, I. Gabriel, W. Isaac, E. Lockhart, S. Osindero, L. Rimell, C. Dyer, O. Vinyals, K. Ayoub, J. Stanway, L. Bennett, D. Hassabis, K. Kavukcuoglu, and G. Irving. Scaling language models: Methods, analysis & insights from training Gopher. arXiv 2112.11446, 2021.

<!-- p19-b19 reference col=0 bbox=[62.36, 572.75, 534.43, 583.66] -->
- J. W. Rae, A. Potapenko, S. M. Jayakumar, T. P. Lillicrap, K. Choromanski, V. Likhosherstov, D. Dohan,

<!-- p19-b20 reference col=0 bbox=[72.88, 586.3, 534.73, 610.76] -->
- X. Song, A. Gane, T. Sarlos, et al. Compressive transformers for long-range sequence modelling. Advances in Neural Information Processing Systems, 33:6154–6158, 2020.

<!-- p19-b21 reference col=0 bbox=[62.36, 622.17, 532.91, 633.08] -->
- C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu. Exploring

<!-- p19-b22 reference col=0 bbox=[73.21, 635.72, 532.91, 661.24] -->
- the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21(140):1–67, 2020a. URL http://jmlr.org/papers/v21/20-074.html.

<!-- p19-b23 reference col=0 bbox=[62.36, 671.6, 532.91, 682.51] -->
- C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu. Exploring

<!-- p19-b24 reference col=0 bbox=[73.21, 685.15, 532.91, 709.61] -->
- the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21(140):1–67, 2020b.

<!-- p19-b25 reference col=0 bbox=[62.36, 721.02, 532.91, 731.93] -->
- S. Rajbhandari, J. Rasley, O. Ruwase, and Y. He. Zero: Memory optimizations toward training

<!-- p19-b26 reference col=0 bbox=[73.21, 734.57, 534.43, 759.03] -->
- trillion parameter models. In SC20: International Conference for High Performance Computing, Networking, Storage and Analysis, pages 1–16. IEEE, 2020.


<!-- ===== page 20 ===== -->

<!-- p20-b1 reference col=0 bbox=[62.36, 87.27, 534.43, 98.18] -->
- H. Robbins and S. Monro. A Stochastic Approximation Method. The Annals of Mathematical Statistics,

<!-- p20-b2 reference col=0 bbox=[72.9, 100.82, 206.13, 111.73] -->
- 22(3):400–407, Sept. 1951.

<!-- p20-b3 reference col=0 bbox=[62.36, 123.34, 534.65, 134.24] -->
- R. Rudinger, J. Naradowsky, B. Leonard, and B. Van Durme. Gender bias in coreference resolu-

<!-- p20-b4 reference col=0 bbox=[72.86, 136.88, 534.74, 174.89] -->
- tion. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, New Orleans, Louisiana, June 2018. Association for Computational Linguistics.

<!-- p20-b5 reference col=0 bbox=[62.36, 186.5, 532.91, 197.41] -->
- K. Sakaguchi, R. Le Bras, C. Bhagavatula, and Y. Choi. Winogrande: An adversarial winograd schema

<!-- p20-b6 reference col=0 bbox=[73.21, 200.05, 534.42, 224.51] -->
- challenge at scale. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 8732–8740, 2020.

<!-- p20-b7 reference col=0 bbox=[62.36, 236.11, 533.1, 247.02] -->
- M. Sap, H. Rashkin, D. Chen, R. LeBras, and Y. Choi. SocialIQA: Commonsense reasoning about

<!-- p20-b8 reference col=0 bbox=[73.21, 249.66, 532.91, 274.12] -->
- social interactions. Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing, 2019.

<!-- p20-b9 reference col=0 bbox=[62.36, 285.73, 532.92, 296.64] -->
- C. J. Shallue, J. Lee, J. Antognini, J. Sohl-Dickstein, R. Frostig, and G. E. Dahl. Measuring the effects

<!-- p20-b10 reference col=0 bbox=[73.21, 299.28, 488.33, 310.19] -->
- of data parallelism on neural network training. arXiv preprint arXiv:1811.03600, 2018.

<!-- p20-b11 reference col=0 bbox=[62.36, 321.79, 534.74, 332.7] -->
- J. W. Siegel and J. Xu. Approximation rates for neural networks with general activation functions.

<!-- p20-b12 reference col=0 bbox=[73.21, 335.31, 532.91, 360.86] -->
- Neural Networks, 128:313–321, Aug. 2020. URL https://www.sciencedirect.com/scienc e/article/pii/S0893608020301891.

<!-- p20-b13 reference col=0 bbox=[62.36, 371.41, 534.43, 382.31] -->
- S. Smith, M. Patwary, B. Norick, P. LeGresley, S. Rajbhandari, J. Casper, Z. Liu, S. Prabhumoye,

<!-- p20-b14 reference col=0 bbox=[72.83, 384.95, 534.64, 436.51] -->
- G. Zerveas, V. Korthikanti, E. Zhang, R. Child, R. Y. Aminabadi, J. Bernauer, X. Song, M. Shoeybi, Y. He, M. Houston, S. Tiwary, and B. Catanzaro. Using Deepspeed and Megatron to Train Megatronturing NLG 530b, A Large-Scale Generative Language Model. arXiv preprint arXiv:2201.11990, 2022.

<!-- p20-b15 reference col=0 bbox=[62.36, 448.09, 532.91, 460.09] -->
- J. Steinhardt. Updates and lessons from AI forecasting, 2021. URL https://bounded-regret.g

<!-- p20-b16 reference col=0 bbox=[73.21, 461.64, 221.64, 473.64] -->
- host.io/ai-forecasting/.

<!-- p20-b17 reference col=0 bbox=[62.36, 484.18, 534.43, 495.09] -->
- Y. Tay, M. Dehghani, J. Rao, W. Fedus, S. Abnar, H. W. Chung, S. Narang, D. Yogatama, A. Vaswani,

<!-- p20-b18 reference col=0 bbox=[73.21, 497.73, 525.63, 508.64] -->
- and D. Metzler. Scale efficiently: Insights from pre-training and fine-tuning transformers, 2021.

<!-- p20-b19 reference col=0 bbox=[62.36, 520.25, 534.43, 531.16] -->
- R. Thoppilan, D. D. Freitas, J. Hall, N. Shazeer, A. Kulshreshtha, H.-T. Cheng, A. Jin, T. Bos, L. Baker,

<!-- p20-b20 reference col=0 bbox=[72.73, 533.8, 534.65, 626.0] -->
- Y. Du, Y. Li, H. Lee, H. S. Zheng, A. Ghafouri, M. Menegali, Y. Huang, M. Krikun, D. Lepikhin, J. Qin, D. Chen, Y. Xu, Z. Chen, A. Roberts, M. Bosma, Y. Zhou, C.-C. Chang, I. Krivokon, W. Rusch, M. Pickett, K. Meier-Hellstern, M. R. Morris, T. Doshi, R. D. Santos, T. Duke, J. Soraker, B. Zevenbergen, V. Prabhakaran, M. Diaz, B. Hutchinson, K. Olson, A. Molina, E. Hoffman-John, J. Lee, L. Aroyo, R. Rajakumar, A. Butryna, M. Lamm, V. Kuzmina, J. Fenton, A. Cohen, R. Bernstein, R. Kurzweil, B. Aguera-Arcas, C. Cui, M. Croak, E. Chi, and Q. Le. LaMDA: Language models for dialog applications, 2022.

<!-- p20-b21 reference col=0 bbox=[62.36, 637.61, 534.73, 648.52] -->
- A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin.

<!-- p20-b22 reference col=0 bbox=[72.86, 651.16, 534.43, 675.61] -->
- Attention is all you need. In Advances in neural information processing systems, pages 5998–6008, 2017.

<!-- p20-b23 reference col=0 bbox=[62.36, 687.22, 534.43, 698.13] -->
- L. Weidinger, J. Mellor, M. Rauh, C. Griffin, J. Uesato, P.-S. Huang, M. Cheng, M. Glaese, B. Balle,

<!-- p20-b24 reference col=0 bbox=[72.86, 700.77, 534.43, 738.78] -->
- A. Kasirzadeh, Z. Kenton, S. Brown, W. Hawkins, T. Stepleton, C. Biles, A. Birhane, J. Haas, L. Rimell, L. A. Hendricks, W. Isaac, S. Legassick, G. Irving, and I. Gabriel. Ethical and social risks of harm from language models. arXiv submission, 2021.


<!-- ===== page 21 ===== -->

<!-- p21-b1 reference col=0 bbox=[62.36, 87.27, 534.43, 98.18] -->
- J. Welbl, A. Glaese, J. Uesato, S. Dathathri, J. Mellor, L. A. Hendricks, K. Anderson, P. Kohli, B. Coppin,

<!-- p21-b2 reference col=0 bbox=[73.21, 100.82, 537.32, 153.44] -->
- and P.-S. Huang. Challenges in detoxifying language models. In Findings of the Association for Computational Linguistics: EMNLP 2021, pages 2447–2469, Punta Cana, Dominican Republic, Nov. 2021. Association for Computational Linguistics. URL https://aclanthology.org/2021. findings-emnlp.210.

<!-- p21-b3 reference col=0 bbox=[62.36, 163.98, 532.91, 174.89] -->
- A. Xu, E. Pathak, E. Wallace, S. Gururangan, M. Sap, and D. Klein. Detoxifying language models

<!-- p21-b4 reference col=0 bbox=[72.9, 177.53, 533.82, 230.15] -->
- risks marginalizing minority voices. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2390–2397, Online, June 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021 .naacl-main.190. URL https://aclanthology.org/2021.naacl-main.190.

<!-- p21-b5 reference col=0 bbox=[62.36, 240.7, 534.73, 251.6] -->
- G. Yang, E. J. Hu, I. Babuschkin, S. Sidor, X. Liu, D. Farhi, N. Ryder, J. Pachocki, W. Chen, and J. Gao.

<!-- p21-b6 reference col=0 bbox=[72.88, 254.24, 534.73, 293.32] -->
- Tuning large neural networks via zero-shot hyperparameter transfer. In A. Beygelzimer, Y. Dauphin, P. Liang, and J. W. Vaughan, editors, Advances in Neural Information Processing Systems, 2021. URL https://openreview.net/forum?id=Bx6qKuBM2AD.

<!-- p21-b7 reference col=0 bbox=[62.36, 303.86, 532.91, 314.77] -->
- R. Zellers, A. Holtzman, Y. Bisk, A. Farhadi, and Y. Choi. HellaSwag: Can a machine really finish

<!-- p21-b8 reference col=0 bbox=[73.21, 317.41, 532.91, 341.87] -->
- your sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 2019.

<!-- p21-b9 reference col=0 bbox=[62.36, 353.47, 532.91, 364.38] -->
- G. Zhang, L. Li, Z. Nado, J. Martens, S. Sachdeva, G. Dahl, C. Shallue, and R. B. Grosse. Which

<!-- p21-b10 reference col=0 bbox=[72.58, 367.02, 536.06, 433.19] -->
- algorithmic choices matter at which batch sizes? insights from a noisy quadratic model. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019. URL https: //proceedings.neurips.cc/paper/2019/file/e0eacd983971634327ae1819ea8b621 4-Paper.pdf.

<!-- p21-b11 reference col=0 bbox=[62.36, 443.73, 532.92, 454.64] -->
- B. Zoph, I. Bello, S. Kumar, N. Du, Y. Huang, J. Dean, N. Shazeer, and W. Fedus. Designing effective

<!-- p21-b12 reference col=0 bbox=[73.21, 457.28, 206.08, 468.19] -->
- sparse expert models, 2022.


<!-- ===== page 22 ===== -->

<!-- p22-b1 heading col=0 bbox=[243.18, 86.79, 352.1, 111.58] -->
## Appendix

<!-- p22-b2 heading col=0 bbox=[62.36, 127.43, 179.46, 140.38] -->
### A. Training dataset

<!-- p22-b3 body col=0 bbox=[62.36, 154.02, 533.09, 178.82] -->
In Table A1 we show the training dataset makeup used for Chinchilla and all scaling runs. Note that both the MassiveWeb and Wikipedia subsets are both used for more than one epoch.

<!-- T12 table page=22 bbox=[90.73, 189.05, 504.55, 299.17] -->
|  | Disk | Size | Documents | Sampling | proportion | Epochs in 1.4T tokens |
|---|---|---|---|---|---|---|
| MassiveWeb | 1.9 | TB | 604M | 45% | (48%) | 1.24 |
| Books | 2.1 | TB | 4M | 30% | (27%) | 0.75 |
| C4 | 0.75 | TB | 361M | 10% | (10%) | 0.77 |
| News | 2.7 | TB | 1.1B | 10% | (10%) | 0.21 |
| GitHub | 3.1 | TB | 142M | 4% | (3%) | 0.13 |
| Wikipedia | 0.001 | TB | 6M | 1% | (2%) | 3.40 |

<!-- p22-b6 caption col=0 bbox=[62.03, 308.07, 533.1, 359.88] -->
> **Table A1 | MassiveText data makeup. For each subset of MassiveText, we list its total disk size, the number of documents and the sampling proportion used during training—we use a slightly different distribution than in Rae et al. (2021) (shown in parenthesis). In the rightmost column show the number of epochs that are used in 1.4 trillion tokens.**

<!-- p22-b7 heading col=0 bbox=[62.36, 391.02, 245.51, 403.97] -->
### B. Optimal cosine cycle length

<!-- p22-b8 body col=0 bbox=[61.46, 417.62, 532.92, 442.08] -->
One key assumption is made on the cosine cycle length and the corresponding learning rate drop (we use a 10× learning rate decay in line with Rae et al. (2021)).^{9} We find that setting the cosine

<!-- p22-b9 body col=0 bbox=[62.36, 444.72, 532.92, 496.27] -->
cycle length too much longer than the target number of training steps results in sub-optimally trained models, as shown in Figure A1. As a result, we assume that an optimally trained model will have the cosine cycle length correctly calibrated to the maximum number of steps, given the FLOP budget; we follow this rule in our main analysis.

<!-- p22-b10 heading col=0 bbox=[62.36, 520.54, 350.97, 533.49] -->
### C. Consistency of scaling results across datasets

<!-- p22-b11 body col=0 bbox=[61.86, 547.13, 534.31, 599.03] -->
We show scaling results from an IsoFLOP (Approach 2) analysis after training on two different datasets: C4 (Raffel et al., 2020b) and GitHub code (we show results with data from Rae et al. (2021)), results are shown in Table A2. For both set of experiments using subsets of MassiveText, we use the same tokenizer as the MassiveText experiments.

<!-- p22-b12 body col=0 bbox=[62.36, 608.1, 534.43, 646.11] -->
We find that the scaling behaviour on these datasets is very similar to what we found on MassiveText, as shown in Figure A2 and Table A2. This suggests that our results are independent of the dataset as long as one does not train for more than one epoch.

<!-- p22-b13 footnote col=0 bbox=[62.36, 736.82, 534.16, 758.64] -->
[^fn]: ^{9}We find the difference between decaying by 10× and decaying to 0.0 (over the same number of steps) to be small, though decaying by a factor of 10× to be slightly more performant. Decaying by less (5×) is clearly worse.


<!-- ===== page 23 ===== -->

<!-- p23-b1 body col=0 bbox=[455.73, 99.19, 499.7, 116.03] -->
1.0× num. steps 1.1× num. steps

<!-- p23-b2 body col=0 bbox=[455.73, 115.07, 502.77, 131.91] -->
1.25× num. steps 1.5× num. steps

<!-- p23-b3 body col=0 bbox=[108.58, 195.57, 224.13, 211.23] -->
0 2 4 6 8 Million Sequences

<!-- p23-b4 body col=0 bbox=[250.74, 195.57, 366.28, 211.23] -->
0 2 4 6 8 Million Sequences

<!-- p23-b5 body col=0 bbox=[392.89, 195.57, 482.54, 211.23] -->
0 2 4 6 Million Sequences

<!-- p23-b6 body col=0 bbox=[106.06, 323.99, 221.32, 339.64] -->
0.0 2.5 5.0 7.5 10.0 12.5 Million Sequences

<!-- p23-b7 body col=0 bbox=[248.21, 323.99, 363.48, 339.64] -->
0.0 2.5 5.0 7.5 10.0 12.5 Million Sequences

<!-- p23-b8 body col=0 bbox=[390.36, 323.99, 506.15, 339.64] -->
0.0 2.5 5.0 7.5 10.0 12.5 Million Sequences

<!-- p23-b9 caption col=0 bbox=[61.46, 350.37, 534.73, 415.73] -->
> **Figure A1 | Grid over cosine cycle length. We show 6 curves with the cosine cycle length set to 1, 1.1, 1.25, 1.5, 2, and 5× longer than the target number of training steps. When the cosine cycle length is too long, and the learning rate does not drop appropriately, then performance is impaired. We find that overestimating the number of training steps beyond 25% leads to clear drops in performance. We show results where we have set the number of training steps to two different values (top and**

<!-- p23-b10 body col=0 bbox=[62.36, 418.37, 103.9, 429.27] -->
bottom).

<!-- p23-b11 body col=0 bbox=[239.04, 540.52, 366.96, 555.41] -->
10^{17} 10^{19} 10^{21} 10^{23} 10^{25} FLOPs

<!-- p23-b12 body col=0 bbox=[379.58, 540.52, 507.5, 555.41] -->
10^{17} 10^{19} 10^{21} 10^{23} 10^{25} FLOPs

<!-- p23-b13 body col=0 bbox=[109.77, 540.67, 225.95, 555.41] -->
100M 300M 1B 3B 6B 30B Parameters

<!-- p23-b14 body col=0 bbox=[239.04, 653.38, 366.96, 668.27] -->
10^{17} 10^{19} 10^{21} 10^{23} 10^{25} FLOPs

<!-- p23-b15 body col=0 bbox=[379.58, 653.38, 507.5, 668.27] -->
10^{17} 10^{19} 10^{21} 10^{23} 10^{25} FLOPs

<!-- p23-b16 body col=0 bbox=[109.77, 653.52, 225.95, 668.27] -->
100M 300M 1B 3B 6B 30B Parameters

<!-- p23-b17 caption col=0 bbox=[62.36, 679.06, 533.1, 717.32] -->
> **Figure A2 | C4 and GitHub IsoFLOP curves. Using the C4 dataset (Raffel et al., 2020b) and a GitHub dataset (Rae et al., 2021), we generate 4 IsoFLOP profiles and show the parameter and token count scaling, as in Figure 3. Scaling coefficients are shown in Table A2.**


<!-- ===== page 24 ===== -->

<!-- T13 table page=24 bbox=[120.51, 83.47, 474.77, 158.41] -->
| Approach | Coef. 𝑎 where 𝑁 𝑜𝑝𝑡 ∝ 𝐶^𝑎 | Coef. 𝑏 where 𝐷 𝑜𝑝𝑡 ∝ 𝐶^𝑏 |
|---|---|---|
| C4 | 0.50 | 0.50 |
| GitHub | 0.53 | 0.47 |
| Kaplan et al. (2020) | 0.73 | 0.27 |

<!-- p24-b4 caption col=0 bbox=[62.03, 167.32, 534.69, 192.32] -->
> **Table A2 | Estimated parameter and data scaling with increased training compute on two alternate datasets. The listed values are the exponents, 𝑎and 𝑏, on the relationship 𝑁_{𝑜𝑝𝑡}∝𝐶^{𝑎}and**

<!-- p24-b5 body col=0 bbox=[62.91, 192.19, 458.73, 205.58] -->
𝐷_{𝑜𝑝𝑡}∝𝐶^{𝑏}. Using IsoFLOP profiles, we estimate the scaling on two different datasets.

<!-- p24-b6 heading col=0 bbox=[62.36, 226.56, 264.97, 239.51] -->
### D. Details on the scaling analyses

<!-- p24-b7 body col=0 bbox=[62.36, 254.56, 407.43, 265.47] -->
D.1. Approach 1: Fixing model sizes and varying training sequences

<!-- p24-b8 body col=0 bbox=[61.86, 272.62, 533.1, 287.49] -->
We use a maximum learning rate of 2 × 10^{−}^{4} for the smallest models and 1.25 × 10^{−}^{4} for the largest

<!-- p24-b9 body col=0 bbox=[61.86, 289.88, 534.73, 314.59] -->
models. In all cases, the learning rate drops by a factor of 10× during training, using a cosine schedule. We make the assumption that the cosine cycle length should be approximately matched to the number

<!-- p24-b10 body col=0 bbox=[61.94, 317.23, 532.91, 355.24] -->
of training steps. We find that when the cosine cycle overshoots the number of training steps by more than 25%, performance is noticeably degraded—see Figure A1.^{10} We use Gaussian smoothing with a window length of 10 steps to smooth the training curve.

<!-- p24-b11 body col=0 bbox=[62.36, 377.18, 296.41, 388.09] -->
D.2. Approach 3: Parametric fitting of the loss

<!-- p24-b12 body col=0 bbox=[62.36, 399.2, 533.12, 423.66] -->
In this section, we first show how Equation (2) can be derived. We repeat the equation below for clarity,

<!-- p24-b13 equation col=0 bbox=[243.23, 421.75, 319.2, 440.98] -->
$$ ˆ𝐿(𝑁, 𝐷) ≜𝐸+ ^{𝐴} $$  (eq. 5)

<!-- p24-b14 equation col=0 bbox=[309.69, 421.75, 345.26, 446.65] -->
$$ 𝑁^{𝛼}^{+} ^{𝐵} $$  (eq. 5)

<!-- p24-b15 equation col=0 bbox=[336.34, 429.13, 352.37, 447.04] -->
$$ 𝐷^{𝛽}^{,} $$  (eq. 5)

<!-- p24-b16 equation col=0 bbox=[517.75, 430.32, 532.91, 441.23] -->
$$ (5) $$  (eq. 5)

<!-- p24-b17 body col=0 bbox=[62.36, 452.45, 532.91, 490.45] -->
based on a decomposition of the expected risk between a function approximation term and an optimisation suboptimality term. We then give details on the optimisation procedure for fitting the parameters.

<!-- p24-b18 body col=0 bbox=[62.36, 513.53, 533.12, 552.72] -->
Loss decomposition. Formally, we consider the task of predicting the next token 𝑦∈Y based on the previous tokens in a sequence 𝑥∈Y^{𝑠}, with 𝑠varying from 0 to 𝑠_{max}—the maximum sequence length. We consider a distribution 𝑃∈D(X × Y) of tokens in Y and their past in X. A predictor

<!-- p24-b19 body col=0 bbox=[61.94, 554.17, 534.42, 593.37] -->
𝑓: X →D(Y) computes the probability of each token given the past sequence. The Bayes classifier, 𝑓^{★}, minimizes the cross-entropy of 𝑓(𝑥) with the observed tokens 𝑦, with expectation taken on the whole data distribution. We let 𝐿be the expected risk

<!-- p24-b20 equation col=0 bbox=[159.65, 601.29, 400.97, 615.68] -->
$$ 𝐿( 𝑓) ≜𝔼[log 𝑓(𝑥)_{𝑦}], and set 𝑓^{★}≜ argmin $$  (eq. 6)

<!-- p24-b21 equation col=0 bbox=[413.21, 603.13, 435.87, 614.97] -->
$$ 𝐿( 𝑓). $$  (eq. 6)

<!-- p24-b22 equation col=0 bbox=[517.75, 604.31, 532.91, 615.22] -->
$$ (6) $$  (eq. 6)

<!-- p24-b23 equation col=0 bbox=[358.34, 615.67, 410.32, 624.32] -->
$$ 𝑓∈F(X,D(Y)) $$  (eq. 6)

<!-- p24-b24 body col=0 bbox=[62.03, 633.1, 533.2, 672.29] -->
The set of all transformers of size 𝑁, that we denote H_{𝑁}, forms a subset of all functions that map sequences to distributions of tokens X →D(Y). Fitting a transformer of size 𝑁on the expected risk 𝐿( 𝑓) amounts to minimizing such risk on a restricted functional space

<!-- p24-b25 equation col=0 bbox=[257.48, 681.06, 314.33, 694.15] -->
$$ 𝑓_{𝑁}≜argmin $$  (eq. 7)

<!-- p24-b26 equation col=0 bbox=[316.59, 682.05, 339.24, 693.9] -->
$$ 𝐿( 𝑓). $$  (eq. 7)

<!-- p24-b27 equation col=0 bbox=[517.75, 683.24, 532.91, 694.15] -->
$$ (7) $$  (eq. 7)

<!-- p24-b28 equation col=0 bbox=[287.72, 694.59, 307.35, 703.25] -->
$$ 𝑓∈H_{𝑁} $$  (eq. 7)

<!-- p24-b29 body col=0 bbox=[61.86, 712.02, 532.92, 726.71] -->
When we observe a dataset (𝑥_{𝑖}, 𝑦_{𝑖})_{𝑖𝑖}_{∈[}_{1}_{,𝐷}_{]} of size 𝐷, we do not have access to 𝔼_{𝑃}, but instead to the

<!-- p24-b30 body col=0 bbox=[62.36, 726.44, 532.81, 739.84] -->
empirical expectation ^{ˆ}𝔼_{𝐷}over the empirical distribution ^{ˆ}𝑃_{𝐷}. What happens when we are given 𝐷

<!-- p24-b31 footnote col=0 bbox=[73.3, 747.78, 532.88, 758.64] -->
[^fn]: ^{10}This further emphasises the point of not only determining model size, but also training length before training begins.


<!-- ===== page 25 ===== -->

<!-- p25-b1 body col=0 bbox=[62.36, 87.27, 532.92, 125.28] -->
datapoints that we can only see once, and when we constrain the size of the hypothesis space to be 𝑁-dimensional ? We are making steps toward minimizing the empirical risk within a finite-dimensional functional space H_{𝑁}:

<!-- p25-b2 equation col=0 bbox=[158.07, 134.01, 406.65, 147.56] -->
$$ ˆ𝐿_{𝐷}( 𝑓) ≜ˆ𝔼_{𝐷}[log 𝑓(𝑥)_{𝑦}], setting ˆ𝑓_{𝑁,𝐷}≜argmin $$  (eq. 8)

<!-- p25-b3 equation col=0 bbox=[408.91, 134.63, 437.46, 146.85] -->
$$ ˆ𝐿_{𝐷}( 𝑓). $$  (eq. 8)

<!-- p25-b4 equation col=0 bbox=[517.75, 136.2, 532.91, 147.11] -->
$$ (8) $$  (eq. 8)

<!-- p25-b5 equation col=0 bbox=[380.04, 147.55, 399.67, 156.21] -->
$$ 𝑓∈H_{𝑁} $$  (eq. 8)

<!-- p25-b6 body col=0 bbox=[61.86, 166.11, 534.74, 179.14] -->
We are never able to obtain ^{ˆ}𝑓_{𝑁,𝐷}as we typically perform a single epoch over the dataset of size 𝐷.

<!-- p25-b7 body col=0 bbox=[62.36, 179.66, 532.92, 219.79] -->
Instead, be obtain ^{¯}𝑓_{𝑁,𝐷}, which is the result of applying a certain number of gradient steps based on the 𝐷datapoints—the number of steps to perform depends on the gradient batch size, for which we use well-tested heuristics.

<!-- p25-b8 body col=0 bbox=[62.36, 226.73, 533.2, 253.66] -->
Using the Bayes-classifier 𝑓^{★}, the expected-risk minimizer 𝑓_{𝑁}and the “single-epoch empirical-risk minimizer” ^{¯}𝑓_{𝑁,𝐷}, we can finally decompose the loss 𝐿(𝑁, 𝐷) into

<!-- p25-b9 equation col=0 bbox=[151.09, 261.56, 283.95, 283.3] -->
$$ 𝐿(𝑁, 𝐷) ≜𝐿( ^{¯}𝑓_{𝑁,𝐷}) = 𝐿( 𝑓^{★}) + ^{�} $$  (eq. 9)

<!-- p25-b10 equation col=0 bbox=[284.6, 261.56, 365.44, 283.3] -->
$$ 𝐿( 𝑓_{𝑁}) −𝐿( 𝑓^{★})^{�}+ ^{�} $$  (eq. 9)

<!-- p25-b11 equation col=0 bbox=[366.09, 262.45, 438.78, 275.49] -->
$$ 𝐿( ^{¯}𝑓_{𝑁,𝐷}) −𝐿( 𝑓_{𝑁})^{�} $$  (eq. 9)

<!-- p25-b12 equation col=0 bbox=[441.65, 263.39, 444.44, 273.43] -->
$$ . $$  (eq. 9)

<!-- p25-b13 equation col=0 bbox=[517.75, 264.58, 532.91, 275.49] -->
$$ (9) $$  (eq. 9)

<!-- p25-b14 body col=0 bbox=[62.03, 286.41, 533.1, 351.51] -->
The loss comprises three terms: the Bayes risk, i.e. the minimal loss achievable for next-token prediction on the full distribution 𝑃, a.k.a the “entropy of natural text.”; a functional approximation term that depends on the size of the hypothesis space; finally, a stochastic approximation term that captures the suboptimality of minimizing ^{ˆ}𝐿_{𝐷}instead of 𝐿, and of making a single epoch on the provided dataset.

<!-- p25-b15 body col=0 bbox=[62.09, 375.77, 533.1, 414.13] -->
Expected forms of the loss terms. In the decomposition (9), the second term depends entirely on the number of parameters 𝑁that defines the size of the functional approximation space. On the set of two-layer neural networks, it is expected to be proportional to

<!-- p25-b16 body col=0 bbox=[374.69, 400.92, 379.12, 408.89] -->
1

<!-- p25-b17 body col=0 bbox=[62.36, 402.87, 534.43, 440.87] -->
_{𝑁}^{1}^{/}^{2} ^{(}^{Siegel and Xu}^{,}^{2020}^{).} ^{Finally,} given that it corresponds to early stopping in stochastic first order methods, the third term should scale as the convergence rate of these methods, which is lower-bounded by

<!-- p25-b18 body col=0 bbox=[421.33, 428.02, 425.76, 435.99] -->
1

<!-- p25-b19 body col=0 bbox=[61.46, 429.97, 534.74, 481.52] -->
_{𝐷}^{1}^{/}^{2} ^{(}^{Robbins and Monro}^{,} 1951) (and may attain the bound). This convergence rate is expected to be dimension free (see e.g. Bubeck, 2015, for a review) and depends only on the loss smoothness; hence we assume that the second term only depends on 𝐷in (2). Empirically, we find after fitting (2) that

<!-- p25-b20 equation col=0 bbox=[232.84, 487.89, 313.19, 507.44] -->
$$ 𝐿(𝑁, 𝐷) = 𝐸+ 𝐴 $$  (eq. 10)

<!-- p25-b21 equation col=0 bbox=[344.31, 487.89, 350.21, 497.92] -->
$$ 𝐵 $$  (eq. 10)

<!-- p25-b22 equation col=0 bbox=[335.85, 495.27, 362.77, 513.26] -->
$$ 𝐷^{0}^{.}^{28}^{,} $$  (eq. 10)

<!-- p25-b23 equation col=0 bbox=[298.16, 496.21, 331.69, 513.26] -->
$$ 𝑁^{0}^{.}^{34} ^{+} $$  (eq. 10)

<!-- p25-b24 equation col=0 bbox=[511.68, 496.46, 532.91, 507.37] -->
$$ (10) $$  (eq. 10)

<!-- p25-b25 body col=0 bbox=[61.94, 520.6, 533.12, 546.25] -->
with 𝐸= 1.69, 𝐴= 406.4, 𝐵= 410.7. We note that the parameter/data coefficients are both lower than ^{1}

<!-- p25-b26 body col=0 bbox=[62.36, 535.34, 534.74, 559.8] -->
_{2}^{;} ^{this} ^{is} ^{expected} ^{for} ^{the} ^{data-efficiency} ^{coefficient} ^{(but} ^{far} ^{from} ^{the} ^{known} ^{lower-bound).} Future models and training approaches should endeavor to increase these coefficients.

<!-- p25-b27 body col=0 bbox=[62.36, 584.05, 465.61, 595.25] -->
Fitting the decomposition to data. We effectively minimize the following problem

<!-- p25-b28 equation col=0 bbox=[245.22, 604.16, 271.72, 619.31] -->
$$ � LSE^{�} $$  (eq. 11)

<!-- p25-b29 equation col=0 bbox=[430.21, 604.16, 437.76, 617.25] -->
$$ � , $$  (eq. 11)

<!-- p25-b30 equation col=0 bbox=[191.79, 605.05, 206.22, 615.96] -->
$$ ∑︁ $$  (eq. 11)

<!-- p25-b31 equation col=0 bbox=[272.15, 607.21, 429.42, 619.31] -->
$$ 𝑎−𝛼log 𝑁_{𝑖}, 𝑏−𝛽log 𝐷_{𝑖}, 𝑒^{�}−log 𝐿_{𝑖} $$  (eq. 11)

<!-- p25-b32 equation col=0 bbox=[164.37, 608.4, 182.81, 619.31] -->
$$ min $$  (eq. 11)

<!-- p25-b33 equation col=0 bbox=[210.06, 608.4, 243.85, 619.31] -->
$$ Huber_{𝛿} $$  (eq. 11)

<!-- p25-b34 equation col=0 bbox=[511.68, 608.4, 532.91, 619.31] -->
$$ (11) $$  (eq. 11)

<!-- p25-b35 equation col=0 bbox=[159.39, 617.09, 187.62, 624.42] -->
$$ 𝑎,𝑏,𝑒,𝛼,𝛽 $$  (eq. 11)

<!-- p25-b36 equation col=0 bbox=[189.77, 622.68, 208.16, 631.52] -->
$$ Run 𝑖 $$  (eq. 11)

<!-- p25-b37 body col=0 bbox=[61.94, 638.54, 454.98, 650.71] -->
where 𝐿𝑆𝐸is the log-sum-exp operator. We then set 𝐴, 𝐵, 𝐸= exp(𝑎), exp(𝑏), exp(𝑒).

<!-- p25-b38 body col=0 bbox=[62.36, 660.05, 532.92, 711.61] -->
We use the LBFGS algorithm to find local minima of the objective above, started on a grid of initialisation given by: 𝛼∈{0., 0.5, . . . , 2.}, 𝛽∈{0., 0.5, . . . , 2.}, 𝑒∈{−1., −.5, . . . , 1.}, 𝑎∈ {0, 5, . . . , 25}, and 𝑏∈{0, 5, . . . , 25}. We find that the optimal initialisation is not on the boundary of our initialisation sweep.

<!-- p25-b39 body col=0 bbox=[62.36, 717.06, 533.1, 759.03] -->
We use 𝛿= 10^{−}^{3} for the Huber loss. We find that using larger values of 𝛿pushes the model to overfit the small compute regime and poorly predict held-out data from larger runs. We find that using a 𝛿smaller than 10^{−}^{3} does not impact the resulting predictions.


<!-- ===== page 26 ===== -->

<!-- p26-b1 body col=0 bbox=[62.36, 87.56, 375.64, 98.47] -->
D.3. Predicted compute optimal frontier for all three methods

<!-- p26-b2 body col=0 bbox=[62.09, 109.59, 532.92, 147.59] -->
For Approaches 2 and 3, we show the estimated model size and number of training tokens for a variety of compute budgets in Table A3. We plot the predicted number of tokens and parameters for a variety of FLOP budgets for the three methods in Figure A3.

<!-- T14 table page=26 bbox=[124.61, 158.17, 470.67, 327.96] -->
|  | Approach 2 | Approach 3 |
|---|---|---|
| Parameters | FLOPs Tokens | FLOPs Tokens |
| 400 Million | 1.84e+19 7.7 Billion | 2.21e+19 9.2 Billion |
| 1 Billion | 1.20e+20 20.0 Billion | 1.62e+20 27.1 Billion |
| 10 Billion | 1.32e+22 219.5 Billion | 2.46e+22 410.1 Billion |
| 67 Billion | 6.88e+23 1.7 Trillion | 1.71e+24 4.1 Trillion |
| 175 Billion | 4.54e+24 4.3 Trillion | 1.26e+24 12.0 Trillion |
| 280 Billion | 1.18e+25 7.1 Trillion | 3.52e+25 20.1 Trillion |
| 520 Billion | 4.19e+25 13.4 Trillion | 1.36e+26 43.5 Trillion |
| 1 Trillion | 1.59e+26 26.5 Trillion | 5.65e+26 94.1 Trillion |
| 10 Trillion | 1.75e+28 292.0 Trillion | 8.55e+28 1425.5 Trillion |

<!-- p26-b6 caption col=0 bbox=[62.03, 336.86, 534.65, 375.12] -->
> **Table A3 | Estimated optimal training FLOPs and training tokens for various model sizes. Analogous to Table 3, we show the model size/token count projections from Approaches 2 and 3 for various compute budgets.**

<!-- p26-b7 body col=0 bbox=[294.35, 377.76, 297.39, 388.66] -->
.

<!-- p26-b8 caption col=0 bbox=[62.06, 637.47, 534.43, 675.73] -->
> **Figure A3 | Optimal number of tokens and parameters for a training FLOP budget. For a fixed FLOP budget, we show the optimal number of tokens and parameters as predicted by Approaches 1, 2, and 3. For an alternate representation, see Figure 1.**

<!-- p26-b9 body col=0 bbox=[62.36, 712.16, 325.06, 723.11] -->
D.4. Small-scale comparison to Kaplan et al. (2020)

<!-- p26-b10 body col=0 bbox=[62.36, 730.27, 532.92, 758.69] -->
For 10^{21} FLOPs, we perform a head-to-head comparison of a model predicted by Approach 1 and that predicted by Kaplan et al. (2020). For both models, we use a batch size of 0.5M tokens and a


<!-- ===== page 27 ===== -->

<!-- p27-b1 body col=0 bbox=[62.36, 83.31, 533.18, 165.93] -->
maximum learning rate of 1.5 × 10^{−}^{4} that decays by 10×. From Kaplan et al. (2020), we find that the optimal model size should be 4.68 billion parameters. From our approach 1, we estimate a 2.86 billion parameter model should be optimal. We train a 4.74 billion parameter and a 2.80 billion parameter transformer to test this hypothesis, using the same depth-to-width ratio to avoid as many confounding factors as possible. We find that our predicted model outperforms the model predicted by Kaplan et al. (2020) as shown in Figure A4.

<!-- p27-b2 body col=0 bbox=[406.45, 188.17, 497.86, 218.28] -->
Kaplan et al (2020) Approach 1

<!-- p27-b3 body col=0 bbox=[331.12, 371.84, 505.85, 399.83] -->
0.0 0.2 0.4 0.6 0.8 1.0 FLOPs ×10^{21}

<!-- p27-b4 caption col=0 bbox=[61.46, 407.5, 534.73, 449.18] -->
> **Figure A4 | Comparison to Kaplan et al. (2020) at 10^{21} FLOPs. We train 2.80 and 4.74 billion parameter transformers predicted as optimal for 10^{21} FLOPs by Approach 1 and by Kaplan et al. (2020). We find that our prediction results in a more performant model at the end of training.**

<!-- p27-b5 heading col=0 bbox=[62.36, 485.36, 288.17, 498.31] -->
### E. Curvature of the FLOP-loss frontier

<!-- p27-b6 body col=0 bbox=[61.86, 511.95, 532.91, 522.86] -->
We observe that as models increase there is a curvature in the FLOP-minimal loss frontier. This means

<!-- p27-b7 body col=0 bbox=[61.94, 525.5, 534.74, 577.06] -->
that projections from very small models lead to different predictions than those from larger models. In Figure A5 we show linear fits using the first, middle, and final third of frontier-points. In this work, we do not take this in to account and we leave this as interesting future work as it suggests that even smaller models may be optimal for large FLOP budgets.

<!-- p27-b8 heading col=0 bbox=[62.36, 600.28, 195.19, 613.23] -->
### F. FLOPs computation

<!-- p27-b9 body col=0 bbox=[61.86, 626.87, 533.12, 637.78] -->
We include all training FLOPs, including those contributed to by the embedding matrices, in our

<!-- p27-b10 body col=0 bbox=[62.36, 640.42, 532.91, 678.43] -->
analysis. Note that we also count embeddings matrices in the total parameter count. For large models the FLOP and parameter contribution of embedding matrices is small. We use a factor of 2 to describe the multiply accumulate cost. For the forward pass, we consider contributions from:

<!-- p27-b11 body col=0 bbox=[77.74, 695.07, 148.0, 705.98] -->
• Embeddings

<!-- p27-b12 body col=0 bbox=[102.73, 712.5, 282.12, 723.95] -->
– 2 × seq_len × vocab_size × d_model

<!-- p27-b13 body col=0 bbox=[77.74, 730.44, 204.46, 741.35] -->
• Attention (Single Layer)

<!-- p27-b14 body col=0 bbox=[102.73, 747.87, 532.37, 759.32] -->
– Key, query and value projections: 2 × 3 × seq_len × d_model × (key_size × num_heads)


<!-- ===== page 28 ===== -->

<!-- p28-b1 caption col=0 bbox=[62.36, 309.51, 534.43, 334.22] -->
> **Figure A5 | Training curve envelopes. We fit to the first third (orange), the middle third (green), and the last third (blue) of all points along the loss frontier. We plot only a subset of the points.**

<!-- p28-b2 body col=0 bbox=[102.73, 356.69, 489.66, 408.79] -->
– Key @ Query logits: 2 × seq_len × seq_len × (key_size × num_heads) – Softmax: 3 × num_heads × seq_len × seq_len – Softmax @ query reductions: 2 × seq_len × seq_len × (key_size × num_heads) – Final Linear: 2 × seq_len × (key_size × num_heads) × d_model

<!-- p28-b3 body col=0 bbox=[77.74, 415.29, 218.02, 426.2] -->
• Dense Block (Single Layer)

<!-- p28-b4 body col=0 bbox=[102.73, 432.74, 382.34, 444.19] -->
– 2 × seq_len × (d_model × ffw_size + d_model × ffw_size)

<!-- p28-b5 body col=0 bbox=[77.74, 450.7, 144.87, 461.6] -->
• Final Logits

<!-- p28-b6 body col=0 bbox=[102.73, 468.15, 282.12, 479.6] -->
– 2 × seq_len × d_model × vocab_size

<!-- p28-b7 body col=0 bbox=[77.74, 485.85, 532.91, 497.3] -->
• Total forward pass FLOPs: embeddings+num_layers×(total_attention+dense_block) + logits

<!-- p28-b8 body col=0 bbox=[61.46, 513.74, 534.73, 551.75] -->
As in Kaplan et al. (2020) we assume that the backward pass has twice the FLOPs of the forward pass. We show a comparison between our calculation and that using the common approximation 𝐶= 6𝐷𝑁 (Kaplan et al., 2020) where 𝐶is FLOPs, 𝐷is the number of training tokens, and 𝑁is the number of

<!-- p28-b9 body col=0 bbox=[62.36, 554.39, 533.1, 578.85] -->
parameters in Table A4. We find the differences in FLOP calculation to be very small and they do not impact our analysis. Compared to the results presented in Rae et al. (2021), we use a slightly more

<!-- T15 table page=28 bbox=[62.36, 588.76, 540.95, 698.88] -->
| Parameters | num_layers | d_model | ffw_size | num_heads | k/q size | FLOP Ratio (Ours/6𝑁𝐷) |
|---|---|---|---|---|---|---|
| 73M | 10 | 640 | 2560 | 10 | 64 | 1.03 |
| 305M | 20 | 1024 | 4096 | 16 | 64 | 1.10 |
| 552M | 24 | 1280 | 5120 | 10 | 128 | 1.08 |
| 1.1B | 26 | 1792 | 7168 | 14 | 128 | 1.04 |
| 1.6B | 28 | 2048 | 8192 | 16 | 128 | 1.03 |
| 6.8B | 40 | 3584 | 14336 | 28 | 128 | 0.99 |

<!-- p28-b12 caption col=0 bbox=[62.03, 707.78, 532.92, 732.49] -->
> **Table A4 | FLOP comparison. For a variety of different model sizes, we show the ratio of the FLOPs that we compute per sequence to that using the 6𝑁𝐷approximation.**

<!-- p28-b13 body col=0 bbox=[62.36, 744.16, 490.97, 759.03] -->
accurate calculation giving a slightly different value (6.3 × 10^{23} compared to 5.76 × 10^{23}).


<!-- ===== page 29 ===== -->

<!-- p29-b1 heading col=0 bbox=[62.36, 85.94, 377.65, 98.93] -->
### G. Other differences between Chinchilla and Gopher

<!-- p29-b2 body col=0 bbox=[61.94, 112.58, 534.43, 177.68] -->
Beyond differences in model size and number of training tokens, there are some additional minor differences between Chinchilla and Gopher. Specifically, Gopher was trained with Adam (Kingma and Ba, 2014) whereas Chinchilla was trained with AdamW (Loshchilov and Hutter, 2019). Furthermore, as discussed in Lessons Learned in Rae et al. (2021), Chinchilla stored a higher-precision copy of the weights in the sharded optimiser state.

<!-- p29-b3 body col=0 bbox=[61.86, 187.1, 534.74, 211.55] -->
We show comparisons of models trained with Adam and AdamW in Figure A6 and Figure A7. We find that, independent of the learning rate schedule, AdamW trained models outperform models

<!-- p29-b4 body col=0 bbox=[62.36, 214.2, 532.92, 225.1] -->
trained with Adam. In Figure A6 we show a comparison of an 680 million parameter model trained

<!-- p29-b5 caption col=0 bbox=[61.46, 370.8, 532.92, 422.95] -->
> **Figure A6 | Comparison of other differences. Using an 680 million parameter model, we show a comparison between the setup used to train Gopher and Chinchilla— the change in optimiser and using a higher precision copy of the weights in the optimiser state. The setup used for Chinchilla (orange) clearly outperforms the setup used to train Gopher (green).**

<!-- p29-b6 caption col=0 bbox=[62.01, 581.48, 532.91, 606.19] -->
> **Figure A7 | Adam vs AdamW. For a 417M (blue) and 1.4B model (green), we find that training with AdamW improves performance over training with Adam.**

<!-- p29-b7 body col=0 bbox=[61.94, 622.48, 533.06, 633.39] -->
with and without the higher precision copy of the weights and with Adam/AdamW for comparison.

<!-- p29-b8 heading col=0 bbox=[62.36, 657.65, 125.31, 670.61] -->
### H. Results

<!-- p29-b9 body col=0 bbox=[62.36, 685.65, 129.47, 696.56] -->
H.1. The Pile

<!-- p29-b10 body col=0 bbox=[62.36, 707.68, 534.43, 746.75] -->
In Table A5 we show the bits-per-byte (bpb) on The Pile (Gao et al., 2020) of Chinchilla, Gopher, and Jurassic-1. Chinchilla outperforms Gopher on all subsets. Jurassic-1 outperforms Chinchilla on 2 subsets— dm_mathematics and ubuntu_irc.


<!-- ===== page 30 ===== -->

<!-- T16 table page=30 bbox=[112.02, 83.47, 483.25, 369.73] -->
| Subset | Chinchilla (70B) | Gopher (280B) | Jurassic-1 (170B) |
|---|---|---|---|
| pile_cc | 0.667 | 0.691 | 0.669 |
| pubmed_abstracts | 0.559 | 0.578 | 0.587 |
| stackexchange | 0.614 | 0.641 | 0.655 |
| github | 0.337 | 0.377 | 0.358 |
| openwebtext2 | 0.647 | 0.677 | - |
| arxiv | 0.627 | 0.662 | 0.680 |
| uspto_backgrounds | 0.526 | 0.546 | 0.537 |
| freelaw | 0.476 | 0.513 | 0.514 |
| pubmed_central | 0.504 | 0.525 | 0.579 |
| dm_mathematics | 1.111 | 1.142 | 1.037 |
| hackernews | 0.859 | 0.890 | 0.869 |
| nih_exporter | 0.572 | 0.590 | 0.590 |
| opensubtitles | 0.871 | 0.900 | 0.879 |
| europarl | 0.833 | 0.938 | - |
| books3 | 0.675 | 0.712 | 0.835 |
| philpapers | 0.656 | 0.695 | 0.742 |
| gutenberg_pg_19 | 0.548 | 0.656 | 0.890 |
| bookcorpus2 | 0.714 | 0.741 | - |
| ubuntu_irc | 1.026 | 1.090 | 0.857 |

<!-- p30-b3 caption col=0 bbox=[62.03, 378.63, 532.91, 403.34] -->
> **Table A5 | Bits-per-Byte on The Pile. We show the bpb on The Pile for Chinchilla compared to Gopher and Jurassic-1.**

<!-- p30-b4 body col=0 bbox=[62.36, 426.01, 121.41, 436.92] -->
H.2. MMLU

<!-- p30-b5 body col=0 bbox=[62.36, 448.04, 483.07, 459.29] -->
In Table A6 we show the performance of Chinchilla and Gopher on each subset of MMLU.

<!-- p30-b6 body col=0 bbox=[62.36, 480.94, 182.18, 491.85] -->
H.3. Winogender Setup

<!-- p30-b7 body col=0 bbox=[61.86, 502.97, 532.91, 514.25] -->
We follow the same setup as in Rae et al. (2021). To test coreference resolution in Chinchilla, we

<!-- p30-b8 body col=0 bbox=[61.04, 516.52, 532.92, 568.07] -->
input a sentence which includes a pronoun reference (e.g., “The librarian helped the child pick out a book because {pronoun} liked to encourage reading.”), then measure the probability of the model completing the sentence “‘{Pronoun}’ refers to the” with different sentence roles (“librarian” and “child” in this example). Each example is annotated with the correct pronoun resolution (the pronoun

<!-- p30-b9 body col=0 bbox=[62.36, 570.71, 532.92, 608.72] -->
corresponds to the librarian in this example). Each sentence is tested with a female, male, and gender-neutral pronoun. An unbiased model would correctly predict which word the pronoun refers to regardless of pronoun gender.

<!-- p30-b10 body col=0 bbox=[62.36, 630.72, 141.02, 641.63] -->
H.4. BIG-bench

<!-- p30-b11 body col=0 bbox=[62.36, 652.74, 534.73, 663.97] -->
In Table A7 we show Chinchilla and Gopher performance on each subset of BIG-bench that we consider.

<!-- p30-b12 heading col=0 bbox=[62.36, 686.67, 146.77, 699.63] -->
## I. Model Card

<!-- p30-b13 body col=0 bbox=[61.86, 713.27, 532.92, 724.55] -->
We present the Chinchilla model card in Table A8, following the framework presented by Mitchell

<!-- p30-b14 body col=0 bbox=[62.36, 726.82, 125.26, 737.73] -->
et al. (2019).


<!-- ===== page 31 ===== -->

<!-- T17 table page=31 bbox=[62.36, 83.47, 547.47, 457.4] -->
| Task | Chinchilla | Gopher | Task | Chinchilla | Gopher |
|---|---|---|---|---|---|
| abstract_algebra | 31.0 | 25.0 | anatomy | 70.4 | 56.3 |
| astronomy | 73.0 | 65.8 | business_ethics | 72.0 | 70.0 |
| clinical_knowledge | 75.1 | 67.2 | college_biology | 79.9 | 70.8 |
| college_chemistry | 51.0 | 45.0 | college_computer_science | 51.0 | 49.0 |
| college_mathematics | 32.0 | 37.0 | college_medicine | 66.5 | 60.1 |
| college_physics | 46.1 | 34.3 | computer_security | 76.0 | 65.0 |
| conceptual_physics | 67.2 | 49.4 | econometrics | 38.6 | 43.0 |
| electrical_engineering | 62.1 | 60.0 | elementary_mathematics | 41.5 | 33.6 |
| formal_logic | 33.3 | 35.7 | global_facts | 39.0 | 38.0 |
| high_school_biology | 80.3 | 71.3 | high_school_chemistry | 58.1 | 47.8 |
| high_school_computer_science | 58.0 | 54.0 | high_school_european_history | 78.8 | 72.1 |
| high_school_geography | 86.4 | 76.8 | high_school_gov_and_politics | 91.2 | 83.9 |
| high_school_macroeconomics | 70.5 | 65.1 | high_school_mathematics | 31.9 | 23.7 |
| high_school_microeconomics | 77.7 | 66.4 | high_school_physics | 36.4 | 33.8 |
| high_school_psychology | 86.6 | 81.8 | high_school_statistics | 58.8 | 50.0 |
| high_school_us_history | 83.3 | 78.9 | high_school_world_history | 85.2 | 75.1 |
| human_aging | 77.6 | 66.4 | human_sexuality | 86.3 | 67.2 |
| international_law | 90.9 | 77.7 | jurisprudence | 79.6 | 71.3 |
| logical_fallacies | 80.4 | 72.4 | machine_learning | 41.1 | 41.1 |
| management | 82.5 | 77.7 | marketing | 89.7 | 83.3 |
| medical_genetics | 69.0 | 69.0 | miscellaneous | 84.5 | 75.7 |
| moral_disputes | 77.5 | 66.8 | moral_scenarios | 36.5 | 40.2 |
| nutrition | 77.1 | 69.9 | philosophy | 79.4 | 68.8 |
| prehistory | 81.2 | 67.6 | professional_accounting | 52.1 | 44.3 |
| professional_law | 56.5 | 44.5 | professional_medicine | 75.4 | 64.0 |
| professional_psychology | 75.7 | 68.1 | public_relations | 73.6 | 71.8 |
| security_studies | 75.9 | 64.9 | sociology | 91.0 | 84.1 |
| us_foreign_policy | 92.0 | 81.0 | virology | 53.6 | 47.0 |
| world_religions | 87.7 | 84.2 |  |  |  |

<!-- p31-b3 caption col=0 bbox=[62.03, 466.3, 533.34, 491.36] -->
> **Table A6 | Chinchilla MMLU results. For each subset of MMLU (Hendrycks et al., 2020), we show Chinchilla’s accuracy compared to Gopher.**

<!-- T18 table page=31 bbox=[62.36, 521.34, 533.7, 684.59] -->
|  | Model Details |
|---|---|
| Organization Developing the Model | DeepMind |
| Model Date | March 2022 |
| Model Type | Autoregressive Transformer Language Model (Section 4.1 for |
|  | details) |
| Feedback on the Model | {jordanhoffmann, sborgeaud, |
|  | amensch,sifre}@deepmind.com |
|  | Intended Uses |

<!-- p31-b10 body col=0 bbox=[68.34, 686.84, 529.12, 724.84] -->
Primary Intended Uses The primary use is research on language models, including: research on the scaling behaviour of language models along with those listed in Rae et al. (2021).


<!-- ===== page 32 ===== -->

<!-- p32-b1 body col=0 bbox=[68.34, 87.27, 527.73, 111.73] -->
Primary Intended Users DeepMind researchers. We will not make this model available publicly.

<!-- p32-b2 body col=0 bbox=[68.34, 119.84, 529.46, 171.4] -->
Out-of-Scope Uses Uses of the language model for language generation in harmful or deceitful settings. More generally, the model should not be used for downstream applications without further safety and fairness mitigations.

<!-- p32-b3 body col=0 bbox=[68.34, 209.6, 529.46, 274.71] -->
Card Prompts – Relevant Factor Relevant factors include which language is used. Our model is trained on English data. Furthermore, in the analysis of models trained on the same corpus in Rae et al. (2021), we found it has unequal performance when modelling some dialects (e.g., African American English). Our model is designed for

<!-- p32-b4 body col=0 bbox=[245.39, 277.35, 529.45, 315.35] -->
research. The model should not be used for downstream applications without further analysis on factors in the proposed downstream application.

<!-- p32-b5 body col=0 bbox=[68.34, 323.47, 527.72, 347.93] -->
Card Prompts – Evaluation Factors See the results in Rae et al. (2021) which analyzes models trained on the same text corpus.

<!-- p32-b6 body col=0 bbox=[68.34, 386.13, 208.55, 397.04] -->
Model Performance Measures

<!-- p32-b7 body col=0 bbox=[260.77, 399.68, 529.24, 532.53] -->
• Perplexity and bits per byte on language modelling datasets • Accuracy on completion tasks, reading comprehension, MMLU, BIG-bench and fact checking. • Exact match accuracy for question answering. • Generation toxicity from Real Toxicity Prompts (RTP) alongside toxicity classification accuracy. • Gender and occupation bias. Test include comparing the probability of generating different gender terms and the Winogender coreference resolution task.

<!-- p32-b8 body col=0 bbox=[244.89, 548.72, 527.73, 559.99] -->
We principally focus on Chinchilla’s performance compared

<!-- p32-b9 body col=0 bbox=[245.39, 562.27, 431.71, 573.52] -->
to Gopher on text likelihood prediction.

<!-- T19 table page=32 bbox=[62.36, 575.21, 533.7, 628.14] -->
| Decision | thresholds | N/A |
|---|---|---|
| Approaches Uncertainty | to and Vari- | Due to the costs of training large language models, we did |
| ability |  | not train Chinchilla multiple times. However, the breadth |

<!-- p32-b12 body col=0 bbox=[245.39, 600.31, 529.55, 692.52] -->
Due to the costs of training large language models, we did not train Chinchilla multiple times. However, the breadth of our evaluation on a range of different task types gives a reasonable estimate of the overall performance of the model. Furthermore, the existence of another large model trained on the same dataset (Gopher) provides a clear point of comparison.


<!-- ===== page 33 ===== -->

<!-- p33-b1 body col=0 bbox=[260.77, 109.79, 529.46, 351.03] -->
• Language modelling on LAMBADA, Wikitext103 (Merity et al., 2017), C4 (Raffel et al., 2020a), PG-19 (Rae et al., 2020) and the Pile (Gao et al., 2020). • Language understanding, real world knowledge, mathematical and logical reasoning on the Massive Multitask Language Understanding (MMLU) benchmark (Hendrycks et al., 2020) and on the “Beyond the Imitation Game Benchmark” (BIG-bench) (BIG-bench collaboration, 2021). • Question answering (closed book) on Natural Questions (Kwiatkowski et al., 2019) and TriviaQA (Joshi et al., 2017). • Reading comprehension on RACE (Lai et al., 2017) • Common sense understanding on HellaSwag (Zellers et al., 2019), PIQA (Bisk et al., 2020), Winogrande (Sakaguchi et al., 2020), SIQA (Sap et al., 2019), BoolQ (Clark et al., 2019), and TruthfulQA (Lin et al., 2021).

<!-- p33-b2 body col=0 bbox=[68.34, 372.69, 527.72, 397.5] -->
Motivation We chose evaluations from Rae et al. (2021) to allow us to most directly compare to Gopher.

<!-- p33-b3 body col=0 bbox=[68.34, 405.27, 527.93, 456.82] -->
Preprocessing Input text is tokenized using a SentencePiece tokenizer with a vocabulary of size 32,000. Unlike the tokenizer used for Gopher, the tokenizer used for Chinchilla does not perform NFKC normalization.

<!-- p33-b4 body col=0 bbox=[261.61, 484.13, 331.42, 495.04] -->
Training Data

<!-- p33-b5 body col=0 bbox=[68.44, 506.98, 524.59, 517.89] -->
The same dataset is used as in Rae et al. (2021). Differences in sampling are shown in Table A1.

<!-- p33-b6 body col=0 bbox=[243.04, 533.24, 353.02, 544.15] -->
Quantitative Analyses

<!-- p33-b7 body col=0 bbox=[68.34, 556.09, 527.73, 580.55] -->
Unitary Results Section 4.2 gives a detailed description of our analysis. Main take-aways include:

<!-- p33-b8 body col=0 bbox=[260.77, 592.16, 529.52, 738.56] -->
• Our model is capable of outputting toxic language as measured by the PerspectiveAPI. This is particularly true when the model is prompted with toxic prompts. • Gender: Our model emulates stereotypes found in our dataset, with occupations such as “dietician” and “receptionist” being more associated with women and “carpenter” and “sheriff” being more associated with men. • Race/religion/country sentiment: Prompting our model to discuss some groups leads to sentences with lower or higher sentiment, likely reflecting text in our dataset.


<!-- ===== page 34 ===== -->

<!-- p34-b1 body col=0 bbox=[68.34, 87.27, 452.42, 98.18] -->
Intersectional Results We did not investigate intersectional biases.

<!-- T20 table page=34 bbox=[62.36, 101.33, 533.7, 185.9] -->
|  | Ethical | Considerations |
|---|---|---|
| Data | The data is | the same as described in Rae et al. (2021). |
| Human Life | The model | is not intended to inform decisions about matters |
|  | central to | human life or flourishing. |

<!-- p34-b5 body col=0 bbox=[68.34, 187.98, 527.91, 266.63] -->
Mitigations We considered filtering the dataset to remove toxic content but decided against it due to the observation that this can introduce new biases as studied by Welbl et al. (2021). More work is needed on mitigation approaches to toxic content and other types of risks associated with language models, such as those discussed in Weidinger et al. (2021).

<!-- p34-b6 body col=0 bbox=[68.34, 274.74, 529.46, 339.85] -->
Risks and Harms The data is collected from the internet, and thus undoubtedly there is toxic/biased content in our training dataset. Furthermore, it is likely that personal information is also in the dataset that has been used to train our models. We defer to the more detailed discussion in Weidinger et al. (2021).

<!-- p34-b7 body col=0 bbox=[68.34, 347.96, 529.46, 426.62] -->
Use Cases Especially fraught use cases include the generation of factually incorrect information with the intent of distributing it or using the model to generate racist, sexist or otherwise toxic text with harmful intent. Many more use cases that could cause harm exist. Such applications to malicious use are discussed in detail in Weidinger et al. (2021).

<!-- p34-b8 caption col=0 bbox=[60.91, 441.72, 517.79, 453.17] -->
> **Table A8 | Chinchilla model card. We follow the framework presented in Mitchell et al. (2019).**

<!-- p34-b9 body col=0 bbox=[62.36, 479.67, 209.02, 492.62] -->
J. List of trained models

<!-- p34-b10 body col=0 bbox=[62.36, 506.26, 532.91, 530.72] -->
In Table A9 we list the model size and configuration of all models used in this study. Many models have been trained multiple times, for a different number of training steps.


<!-- ===== page 35 ===== -->

<!-- T21 table page=35 bbox=[62.36, 203.5, 547.84, 601.33] -->
| Task | Chinchilla | Gopher | Task | Chinchilla | Gopher |
|---|---|---|---|---|---|
| hyperbaton | 54.2 | 51.7 | movie_dialog_same_or_diff | 54.5 | 50.7 |
| causal_judgment | 57.4 | 50.8 | winowhy | 62.5 | 56.7 |
| formal_fallacies_syllogisms_neg | 52.1 | 50.7 | movie_recommendation | 75.6 | 50.5 |
| crash_blossom | 47.6 | 63.6 | moral_permissibility | 57.3 | 55.1 |
| discourse_marker_prediction | 13.1 | 11.7 | strategyqa | 68.3 | 61.0 |
| general_knowledge_json | 94.3 | 93.9 | nonsense_words_grammar | 78.0 | 61.4 |
| sports_understanding | 71.0 | 54.9 | metaphor_boolean | 93.1 | 59.3 |
| implicit_relations | 49.4 | 36.4 | navigate | 52.6 | 51.1 |
| penguins_in_a_table | 48.7 | 40.6 | presuppositions_as_nli | 49.9 | 34.0 |
| intent_recognition | 92.8 | 88.7 | temporal_sequences | 32.0 | 19.0 |
| reasoning_about_colored_objects | 59.7 | 49.2 | question_selection | 52.6 | 41.4 |
| logic_grid_puzzle | 44.0 | 35.1 | logical_fallacy_detection | 72.1 | 58.9 |
| timedial | 68.8 | 50.9 | physical_intuition | 79.0 | 59.7 |
| epistemic_reasoning | 60.6 | 56.4 | physics_mc | 65.5 | 50.9 |
| ruin_names | 47.1 | 38.6 | identify_odd_metaphor | 68.8 | 38.6 |
| hindu_knowledge | 91.4 | 80.0 | understanding_fables | 60.3 | 39.6 |
| misconceptions | 65.3 | 61.7 | logical_sequence | 64.1 | 36.4 |
| implicatures | 75.0 | 62.0 | mathematical_induction | 47.3 | 57.6 |
| disambiguation_q | 54.7 | 45.5 | fantasy_reasoning | 69.0 | 64.1 |
| known_unknowns | 65.2 | 63.6 | SNARKS | 58.6 | 48.3 |
| dark_humor_detection | 66.2 | 83.1 | crass_ai | 75.0 | 56.8 |
| analogical_similarity | 38.1 | 17.2 | entailed_polarity | 94.0 | 89.5 |
| sentence_ambiguity | 71.7 | 69.1 | irony_identification | 73.0 | 69.7 |
| riddle_sense | 85.7 | 68.2 | evaluating_info_essentiality | 17.6 | 16.7 |
| date_understanding | 52.3 | 44.1 | phrase_relatedness | 94.0 | 81.8 |
| analytic_entailment | 67.1 | 53.0 | novel_concepts | 65.6 | 59.1 |
| odd_one_out | 70.9 | 32.5 | empirical_judgments | 67.7 | 52.5 |
| logical_args | 56.2 | 59.1 | figure_of_speech_detection | 63.3 | 52.7 |
| alignment_questionnaire | 91.3 | 79.2 | english_proverbs | 82.4 | 57.6 |
| similarities_abstraction | 87.0 | 81.8 | Human_organs_senses_mcc | 85.7 | 84.8 |
| anachronisms | 69.1 | 56.4 | gre_reading_comprehension | 53.1 | 27.3 |

<!-- p35-b3 caption col=0 bbox=[62.03, 610.24, 534.43, 635.29] -->
> **Table A7 | Chinchilla BIG-bench results. For each subset of BIG-bench (BIG-bench collaboration, 2021), we show Chinchilla and Gopher’s accuracy.**


<!-- ===== page 36 ===== -->

<!-- T22 table page=36 bbox=[141.72, 93.57, 453.56, 667.74] -->
| Parameters (million) | d_model | ffw_size | kv_size | n_heads | n_layers |
|---|---|---|---|---|---|
| 44 | 512 | 2048 | 64 | 8 | 8 |
| 57 | 576 | 2304 | 64 | 9 | 9 |
| 74 | 640 | 2560 | 64 | 10 | 10 |
| 90 | 640 | 2560 | 64 | 10 | 13 |
| 106 | 640 | 2560 | 64 | 10 | 16 |
| 117 | 768 | 3072 | 64 | 12 | 12 |
| 140 | 768 | 3072 | 64 | 12 | 15 |
| 163 | 768 | 3072 | 64 | 12 | 18 |
| 175 | 896 | 3584 | 64 | 14 | 14 |
| 196 | 896 | 3584 | 64 | 14 | 16 |
| 217 | 896 | 3584 | 64 | 14 | 18 |
| 251 | 1024 | 4096 | 64 | 16 | 16 |
| 278 | 1024 | 4096 | 64 | 16 | 18 |
| 306 | 1024 | 4096 | 64 | 16 | 20 |
| 425 | 1280 | 5120 | 128 | 10 | 18 |
| 489 | 1280 | 5120 | 128 | 10 | 21 |
| 509 | 1408 | 5632 | 128 | 11 | 18 |
| 552 | 1280 | 5120 | 128 | 10 | 24 |
| 587 | 1408 | 5632 | 128 | 11 | 21 |
| 632 | 1536 | 6144 | 128 | 12 | 19 |
| 664 | 1408 | 5632 | 128 | 11 | 24 |
| 724 | 1536 | 6144 | 128 | 12 | 22 |
| 816 | 1536 | 6144 | 128 | 12 | 25 |
| 893 | 1792 | 7168 | 128 | 14 | 20 |
| 1,018 | 1792 | 7168 | 128 | 14 | 23 |
| 1,143 | 1792 | 7168 | 128 | 14 | 26 |
| 1,266 | 2048 | 8192 | 128 | 16 | 22 |
| 1,424 | 2176 | 8704 | 128 | 17 | 22 |
| 1,429 | 2048 | 8192 | 128 | 16 | 25 |
| 1,593 | 2048 | 8192 | 128 | 16 | 28 |
| 1,609 | 2176 | 8704 | 128 | 17 | 25 |
| 1,731 | 2304 | 9216 | 128 | 18 | 24 |
| 1,794 | 2176 | 8704 | 128 | 17 | 28 |
| 2,007 | 2304 | 9216 | 128 | 18 | 28 |
| 2,283 | 2304 | 9216 | 128 | 18 | 32 |
| 2,298 | 2560 | 10240 | 128 | 20 | 26 |
| 2,639 | 2560 | 10240 | 128 | 20 | 30 |
| 2,980 | 2560 | 10240 | 128 | 20 | 34 |
| 3,530 | 2688 | 10752 | 128 | 22 | 36 |
| 3,802 | 2816 | 11264 | 128 | 22 | 36 |
| 4,084 | 2944 | 11776 | 128 | 22 | 36 |
| 4,516 | 3072 | 12288 | 128 | 24 | 36 |
| 6,796 | 3584 | 14336 | 128 | 28 | 40 |
| 9,293 | 4096 | 16384 | 128 | 32 | 42 |
| 11,452 | 4352 | 17408 | 128 | 32 | 47 |
| 12,295 | 4608 | 18432 | 128 | 36 | 44 |
| 12,569 | 4608 | 18432 | 128 | 32 | 47 |
| 13,735 | 4864 | 19456 | 128 | 32 | 47 |
| 14,940 | 4992 | 19968 | 128 | 32 | 49 |
| 16,183 | 5120 | 20480 | 128 | 40 | 47 |

<!-- p36-b3 caption col=0 bbox=[60.79, 674.83, 533.49, 713.09] -->
> **Table A9 | All models. We list the hyperparameters and size of all models trained as part of this work. Many shown models have been trained with multiple learning rate schedules/number of training tokens.**


<!-- ===== removed running headers / footers / margin notes ===== -->

<!-- p1-m1 margin_note: arXiv:2203.15556v1 [cs.CL] 29 Mar 2022 -->

<!-- p2-g1 figure_text: 1017 1019 1021 1023 1025 FLOPs -->

<!-- p2-g2 figure_text: 10M -->

<!-- p2-g3 figure_text: 100M -->

<!-- p2-g4 figure_text: 1.0B -->

<!-- p2-g5 figure_text: 10B -->

<!-- p2-g6 figure_text: 100B -->

<!-- p2-g7 figure_text: 1T -->

<!-- p2-g8 figure_text: Parameters -->

<!-- p2-g9 figure_text: Approach 1 Approach 2 Approach 3 Kaplan et al (2020) Chinchilla (70B) Gopher (280B) GPT-3 (175B) Megatron-Turing NLG (53 -->

<!-- p2-m1 footer: 2 -->

<!-- p3-m1 footer: 3 -->

<!-- p4-m1 footer: 4 -->

<!-- p5-g1 figure_text: 1017 1018 1019 1020 1021 1022 FLOPS -->

<!-- p5-g2 figure_text: 2.0 -->

<!-- p5-g3 figure_text: 2.5 -->

<!-- p5-g4 figure_text: 3.0 -->

<!-- p5-g5 figure_text: 3.5 -->

<!-- p5-g6 figure_text: 4.0 -->

<!-- p5-g7 figure_text: 4.5 -->

<!-- p5-g8 figure_text: 5.0 5.5 6.0 -->

<!-- p5-g9 figure_text: Training loss -->

<!-- p5-g10 figure_text: 75M -->

<!-- p5-g11 figure_text: 250M -->

<!-- p5-g12 figure_text: 500M -->

<!-- p5-g13 figure_text: 1B -->

<!-- p5-g14 figure_text: 2.5B -->

<!-- p5-g15 figure_text: 5B -->

<!-- p5-g16 figure_text: 10B -->

<!-- p5-g17 figure_text: 1017 1019 1021 1023 1025 FLOPs -->

<!-- p5-g18 figure_text: 109 -->

<!-- p5-g19 figure_text: 1010 -->

<!-- p5-g20 figure_text: 1011 -->

<!-- p5-g21 figure_text: 1012 -->

<!-- p5-g22 figure_text: Tokens -->

<!-- p5-g23 figure_text: 1.5T -->

<!-- p5-g24 figure_text: 1017 1019 1021 1023 1025 FLOPs -->

<!-- p5-g25 figure_text: 100M -->

<!-- p5-g26 figure_text: 1.0B -->

<!-- p5-g27 figure_text: 10B -->

<!-- p5-g28 figure_text: 100B -->

<!-- p5-g29 figure_text: 1T -->

<!-- p5-g30 figure_text: Parameters -->

<!-- p5-g31 figure_text: 67B -->

<!-- p5-m1 footer: 5 -->

<!-- p6-g1 figure_text: 100M 300M 1B 3B 6B 30B Parameters -->

<!-- p6-g2 figure_text: 2.0 -->

<!-- p6-g3 figure_text: 2.2 -->

<!-- p6-g4 figure_text: 2.4 -->

<!-- p6-g5 figure_text: 2.6 -->

<!-- p6-g6 figure_text: 2.8 -->

<!-- p6-g7 figure_text: 3.0 -->

<!-- p6-g8 figure_text: 3.2 -->

<!-- p6-g9 figure_text: Training Loss -->

<!-- p6-g10 figure_text: 6e18 1e19 3e19 6e19 1e20 3e20 6e20 1e21 3e21 -->

<!-- p6-g11 figure_text: 1017 1019 1021 1023 1025 FLOPs -->

<!-- p6-g12 figure_text: 100M -->

<!-- p6-g13 figure_text: 1B -->

<!-- p6-g14 figure_text: 10B -->

<!-- p6-g15 figure_text: 100B -->

<!-- p6-g16 figure_text: 1T -->

<!-- p6-g17 figure_text: Parameters -->

<!-- p6-g18 figure_text: 63B -->

<!-- p6-g19 figure_text: 1017 1019 1021 1023 1025 FLOPs -->

<!-- p6-g20 figure_text: 100M -->

<!-- p6-g21 figure_text: 1B -->

<!-- p6-g22 figure_text: 10B -->

<!-- p6-g23 figure_text: 100B -->

<!-- p6-g24 figure_text: 1T -->

<!-- p6-g25 figure_text: 10T -->

<!-- p6-g26 figure_text: Tokens -->

<!-- p6-g27 figure_text: 1.4T -->

<!-- p6-m1 footer: 6 -->

<!-- p7-g1 figure_text: 1018 1019 1020 1021 1022 1023 Gopher budget Training FLOPs -->

<!-- p7-g2 figure_text: 100M -->

<!-- p7-g3 figure_text: 1B -->

<!-- p7-g4 figure_text: 10B -->

<!-- p7-g5 figure_text: 40B -->

<!-- p7-g6 figure_text: 100B -->

<!-- p7-g7 figure_text: Model size -->

<!-- p7-g8 figure_text: IsoLoss contours -->

<!-- p7-g9 figure_text: Efficient frontier Empirical data IsoFLOPs slice -->

<!-- p7-g10 figure_text: 2.00 -->

<!-- p7-g11 figure_text: 3.00 -->

<!-- p7-g12 figure_text: 4.00 -->

<!-- p7-g13 figure_text: 5.00 -->

<!-- p7-g14 figure_text: Loss -->

<!-- p7-g15 figure_text: 100M 1B 10B 40B -->

<!-- p7-g16 figure_text: Model size -->

<!-- p7-g17 figure_text: IsoFLOPs slices -->

<!-- p7-g18 figure_text: Train. FLOPs -->

<!-- p7-g19 figure_text: 6e+18 1e+19 3e+19 6e+19 1e+20 3e+20 6e+20 1e+21 3e+21 Gopher -->

<!-- p7-m1 footer: 7 -->

<!-- p8-m1 footer: 8 -->

<!-- p9-m1 footer: 9 -->

<!-- p10-g1 figure_text: pubmed_abstracts -->

<!-- p10-g2 figure_text: nih_exporter -->

<!-- p10-g3 figure_text: uspto_backgrounds -->

<!-- p10-g4 figure_text: pubmed_central -->

<!-- p10-g5 figure_text: pile_cc -->

<!-- p10-g6 figure_text: bookcorpus2 -->

<!-- p10-g7 figure_text: stackexchange -->

<!-- p10-g8 figure_text: opensubtitles -->

<!-- p10-g9 figure_text: openwebtext2 -->

<!-- p10-g10 figure_text: hackernews -->

<!-- p10-g11 figure_text: dm_mathematics -->

<!-- p10-g12 figure_text: arxiv -->

<!-- p10-g13 figure_text: freelaw -->

<!-- p10-g14 figure_text: books3 -->

<!-- p10-g15 figure_text: philpapers -->

<!-- p10-g16 figure_text: github -->

<!-- p10-g17 figure_text: ubuntu_irc -->

<!-- p10-g18 figure_text: europarl -->

<!-- p10-g19 figure_text: gutenberg_pg_19 -->

<!-- p10-g20 figure_text: 0.00 -->

<!-- p10-g21 figure_text: 0.02 -->

<!-- p10-g22 figure_text: 0.04 -->

<!-- p10-g23 figure_text: 0.06 -->

<!-- p10-g24 figure_text: 0.08 -->

<!-- p10-g25 figure_text: 0.10 -->

<!-- p10-g26 figure_text: Decrease in bpb -->

<!-- p10-g27 figure_text: compared to Gopher -->

<!-- p10-m1 footer: 10 -->

<!-- p11-m1 footer: 11 -->

<!-- p12-g1 figure_text: college_mathematics -->

<!-- p12-g2 figure_text: econometrics -->

<!-- p12-g3 figure_text: moral_scenarios -->

<!-- p12-g4 figure_text: formal_logic -->

<!-- p12-g5 figure_text: medical_genetics -->

<!-- p12-g6 figure_text: machine_learning -->

<!-- p12-g7 figure_text: public_relations -->

<!-- p12-g8 figure_text: global_facts -->

<!-- p12-g9 figure_text: business_ethics -->

<!-- p12-g10 figure_text: electrical_engineering -->

<!-- p12-g11 figure_text: college_computer_science -->

<!-- p12-g12 figure_text: world_religions -->

<!-- p12-g13 figure_text: high_school_us_history -->

<!-- p12-g14 figure_text: high_school_psychology -->

<!-- p12-g15 figure_text: management -->

<!-- p12-g16 figure_text: high_school_computer_science -->

<!-- p12-g17 figure_text: marketing -->

<!-- p12-g18 figure_text: high_school_physics -->

<!-- p12-g19 figure_text: high_school_macroeconomics -->

<!-- p12-g20 figure_text: sociology -->

<!-- p12-g21 figure_text: high_school_government_and_politics -->

<!-- p12-g22 figure_text: high_school_european_history -->

<!-- p12-g23 figure_text: nutrition -->

<!-- p12-g24 figure_text: college_medicine -->

<!-- p12-g25 figure_text: astronomy -->

<!-- p12-g26 figure_text: logical_fallacies -->

<!-- p12-g27 figure_text: professional_psychology -->

<!-- p12-g28 figure_text: miscellaneous -->

<!-- p12-g29 figure_text: jurisprudence -->

<!-- p12-g30 figure_text: clinical_knowledge -->

<!-- p12-g31 figure_text: high_school_geography -->

<!-- p12-g32 figure_text: high_school_biology -->

<!-- p12-g33 figure_text: college_biology -->

<!-- p12-g34 figure_text: college_chemistry -->

<!-- p12-g35 figure_text: high_school_world_history -->

<!-- p12-g36 figure_text: us_foreign_policy -->

<!-- p12-g37 figure_text: virology -->

<!-- p12-g38 figure_text: philosophy -->

<!-- p12-g39 figure_text: moral_disputes -->

<!-- p12-g40 figure_text: human_aging -->

<!-- p12-g41 figure_text: computer_security -->

<!-- p12-g42 figure_text: security_studies -->

<!-- p12-g43 figure_text: international_law -->

<!-- p12-g44 figure_text: high_school_microeconomics -->

<!-- p12-g45 figure_text: high_school_statistics -->

<!-- p12-g46 figure_text: professional_accounting -->

<!-- p12-g47 figure_text: professional_medicine -->

<!-- p12-g48 figure_text: prehistory -->

<!-- p12-g49 figure_text: high_school_chemistry -->

<!-- p12-g50 figure_text: elementary_mathematics -->

<!-- p12-g51 figure_text: abstract_algebra -->

<!-- p12-g52 figure_text: anatomy -->

<!-- p12-g53 figure_text: professional_law -->

<!-- p12-g54 figure_text: human_sexuality -->

<!-- p12-g55 figure_text: college_physics -->

<!-- p12-g56 figure_text: high_school_mathematics -->

<!-- p12-g57 figure_text: conceptual_physics -->

<!-- p12-g58 figure_text: 10 -->

<!-- p12-g59 figure_text: 0 -->

<!-- p12-g60 figure_text: 10 -->

<!-- p12-g61 figure_text: 20 -->

<!-- p12-g62 figure_text: 30 -->

<!-- p12-g63 figure_text: Relative Improvement -->

<!-- p12-g64 figure_text: over Gopher -->

<!-- p12-m1 footer: 12 -->

<!-- p13-g1 figure_text: crash_blossom -->

<!-- p13-g2 figure_text: dark_humor_detection -->

<!-- p13-g3 figure_text: mathematical_induction -->

<!-- p13-g4 figure_text: logical_args -->

<!-- p13-g5 figure_text: general_knowledge_json -->

<!-- p13-g6 figure_text: Human_organs_senses_multiple_choice -->

<!-- p13-g7 figure_text: formal_fallacies_syllogisms_negation -->

<!-- p13-g8 figure_text: known_unknowns -->

<!-- p13-g9 figure_text: navigate -->

<!-- p13-g10 figure_text: sentence_ambiguity -->

<!-- p13-g11 figure_text: moral_permissibility -->

<!-- p13-g12 figure_text: intent_recognition -->

<!-- p13-g13 figure_text: irony_identification -->

<!-- p13-g14 figure_text: entailed_polarity -->

<!-- p13-g15 figure_text: hyperbaton -->

<!-- p13-g16 figure_text: misconceptions -->

<!-- p13-g17 figure_text: evaluating_information_essentiality -->

<!-- p13-g18 figure_text: similarities_abstraction -->

<!-- p13-g19 figure_text: epistemic_reasoning -->

<!-- p13-g20 figure_text: fantasy_reasoning -->

<!-- p13-g21 figure_text: movie_dialog_same_or_different -->

<!-- p13-g22 figure_text: winowhy -->

<!-- p13-g23 figure_text: novel_concepts -->

<!-- p13-g24 figure_text: discourse_marker_prediction -->

<!-- p13-g25 figure_text: strategyqa -->

<!-- p13-g26 figure_text: causal_judgment -->

<!-- p13-g27 figure_text: hindu_knowledge -->

<!-- p13-g28 figure_text: phrase_relatedness -->

<!-- p13-g29 figure_text: alignment_questionnaire -->

<!-- p13-g30 figure_text: reasoning_about_colored_objects -->

<!-- p13-g31 figure_text: date_understanding -->

<!-- p13-g32 figure_text: penguins_in_a_table -->

<!-- p13-g33 figure_text: figure_of_speech_detection -->

<!-- p13-g34 figure_text: disambiguation_q -->

<!-- p13-g35 figure_text: implicatures -->

<!-- p13-g36 figure_text: SNARKS -->

<!-- p13-g37 figure_text: ruin_names -->

<!-- p13-g38 figure_text: logical_fallacy_detection -->

<!-- p13-g39 figure_text: anachronisms -->

<!-- p13-g40 figure_text: logic_grid_puzzle -->

<!-- p13-g41 figure_text: riddle_sense -->

<!-- p13-g42 figure_text: analytic_entailment -->

<!-- p13-g43 figure_text: question_selection -->

<!-- p13-g44 figure_text: nonsense_words_grammar -->

<!-- p13-g45 figure_text: physics_mc -->

<!-- p13-g46 figure_text: empirical_judgments -->

<!-- p13-g47 figure_text: sports_understanding -->

<!-- p13-g48 figure_text: crass_ai -->

<!-- p13-g49 figure_text: physical_intuition -->

<!-- p13-g50 figure_text: timedial -->

<!-- p13-g51 figure_text: implicit_relations -->

<!-- p13-g52 figure_text: english_proverbs -->

<!-- p13-g53 figure_text: presuppositions_as_nli -->

<!-- p13-g54 figure_text: movie_recommendation -->

<!-- p13-g55 figure_text: understanding_fables -->

<!-- p13-g56 figure_text: metaphor_boolean -->

<!-- p13-g57 figure_text: temporal_sequences -->

<!-- p13-g58 figure_text: logical_sequence -->

<!-- p13-g59 figure_text: identify_odd_metaphor -->

<!-- p13-g60 figure_text: gre_reading_comprehension -->

<!-- p13-g61 figure_text: odd_one_out -->

<!-- p13-g62 figure_text: analogical_similarity -->

<!-- p13-g63 figure_text: 20 -->

<!-- p13-g64 figure_text: 0 -->

<!-- p13-g65 figure_text: 20 -->

<!-- p13-g66 figure_text: 40 -->

<!-- p13-g67 figure_text: 60 -->

<!-- p13-g68 figure_text: 80 -->

<!-- p13-g69 figure_text: 100 -->

<!-- p13-g70 figure_text: 120 -->

<!-- p13-g71 figure_text: Relative Improvement -->

<!-- p13-g72 figure_text: over Gopher -->

<!-- p13-m1 footer: 13 -->

<!-- p14-m1 footer: 14 -->

<!-- p15-m1 footer: 15 -->

<!-- p16-m1 footer: 16 -->

<!-- p17-m1 footer: 17 -->

<!-- p18-m1 footer: 18 -->

<!-- p19-m1 footer: 19 -->

<!-- p20-m1 footer: 20 -->

<!-- p21-m1 footer: 21 -->

<!-- p22-m1 footer: 22 -->

<!-- p23-g1 figure_text: 0.0 -->

<!-- p23-g2 figure_text: 0.2 -->

<!-- p23-g3 figure_text: 0.4 -->

<!-- p23-g4 figure_text: 0.6 -->

<!-- p23-g5 figure_text: 0.8 -->

<!-- p23-g6 figure_text: 1.0 -->

<!-- p23-g7 figure_text: Learning Rate/Max LR -->

<!-- p23-g8 figure_text: 2.70 -->

<!-- p23-g9 figure_text: 2.75 -->

<!-- p23-g10 figure_text: 2.80 -->

<!-- p23-g11 figure_text: 2.85 -->

<!-- p23-g12 figure_text: 2.90 -->

<!-- p23-g13 figure_text: 2.95 -->

<!-- p23-g14 figure_text: 3.00 -->

<!-- p23-g15 figure_text: Training Loss -->

<!-- p23-g16 figure_text: 2.80 -->

<!-- p23-g17 figure_text: 2.85 -->

<!-- p23-g18 figure_text: 2.90 -->

<!-- p23-g19 figure_text: 2.95 -->

<!-- p23-g20 figure_text: 3.00 -->

<!-- p23-g21 figure_text: 3.05 -->

<!-- p23-g22 figure_text: 3.10 -->

<!-- p23-g23 figure_text: 3.15 -->

<!-- p23-g24 figure_text: 3.20 -->

<!-- p23-g25 figure_text: C4 Loss -->

<!-- p23-g26 figure_text: Cosine Cycle Length -->

<!-- p23-g27 figure_text: 2.0× num. steps -->

<!-- p23-g28 figure_text: 5.0× num. steps -->

<!-- p23-g29 figure_text: 0.0 -->

<!-- p23-g30 figure_text: 0.2 -->

<!-- p23-g31 figure_text: 0.4 -->

<!-- p23-g32 figure_text: 0.6 -->

<!-- p23-g33 figure_text: 0.8 -->

<!-- p23-g34 figure_text: 1.0 -->

<!-- p23-g35 figure_text: Learning Rate/Max LR -->

<!-- p23-g36 figure_text: 2.70 -->

<!-- p23-g37 figure_text: 2.75 -->

<!-- p23-g38 figure_text: 2.80 -->

<!-- p23-g39 figure_text: 2.85 -->

<!-- p23-g40 figure_text: 2.90 -->

<!-- p23-g41 figure_text: 2.95 -->

<!-- p23-g42 figure_text: 3.00 -->

<!-- p23-g43 figure_text: Training Loss -->

<!-- p23-g44 figure_text: 2.80 -->

<!-- p23-g45 figure_text: 2.85 -->

<!-- p23-g46 figure_text: 2.90 -->

<!-- p23-g47 figure_text: 2.95 -->

<!-- p23-g48 figure_text: 3.00 -->

<!-- p23-g49 figure_text: 3.05 -->

<!-- p23-g50 figure_text: 3.10 -->

<!-- p23-g51 figure_text: 3.15 -->

<!-- p23-g52 figure_text: 3.20 -->

<!-- p23-g53 figure_text: C4 Loss -->

<!-- p23-g54 figure_text: 2.0 -->

<!-- p23-g55 figure_text: 2.2 -->

<!-- p23-g56 figure_text: 2.4 -->

<!-- p23-g57 figure_text: 2.6 -->

<!-- p23-g58 figure_text: 2.8 -->

<!-- p23-g59 figure_text: 3.0 -->

<!-- p23-g60 figure_text: 3.2 -->

<!-- p23-g61 figure_text: C4 Training Loss -->

<!-- p23-g62 figure_text: 1e19 1e20 6e20 1e21 -->

<!-- p23-g63 figure_text: 100M -->

<!-- p23-g64 figure_text: 1B -->

<!-- p23-g65 figure_text: 10B -->

<!-- p23-g66 figure_text: 100B -->

<!-- p23-g67 figure_text: 1T -->

<!-- p23-g68 figure_text: Parameters -->

<!-- p23-g69 figure_text: 73B -->

<!-- p23-g70 figure_text: 100M -->

<!-- p23-g71 figure_text: 1B -->

<!-- p23-g72 figure_text: 10B -->

<!-- p23-g73 figure_text: 100B -->

<!-- p23-g74 figure_text: 1T -->

<!-- p23-g75 figure_text: 10T -->

<!-- p23-g76 figure_text: Tokens -->

<!-- p23-g77 figure_text: 1.3T -->

<!-- p23-g78 figure_text: 0.2 -->

<!-- p23-g79 figure_text: 0.3 -->

<!-- p23-g80 figure_text: 0.4 -->

<!-- p23-g81 figure_text: 0.5 -->

<!-- p23-g82 figure_text: 0.6 -->

<!-- p23-g83 figure_text: 0.7 -->

<!-- p23-g84 figure_text: 0.8 -->

<!-- p23-g85 figure_text: 0.9 -->

<!-- p23-g86 figure_text: 1.0 -->

<!-- p23-g87 figure_text: GitHub Training Loss -->

<!-- p23-g88 figure_text: 1e19 1e20 6e20 1e21 -->

<!-- p23-g89 figure_text: 100M -->

<!-- p23-g90 figure_text: 1B -->

<!-- p23-g91 figure_text: 10B -->

<!-- p23-g92 figure_text: 100B -->

<!-- p23-g93 figure_text: 1T -->

<!-- p23-g94 figure_text: Parameters -->

<!-- p23-g95 figure_text: 59B -->

<!-- p23-g96 figure_text: 100M -->

<!-- p23-g97 figure_text: 1B -->

<!-- p23-g98 figure_text: 10B -->

<!-- p23-g99 figure_text: 100B -->

<!-- p23-g100 figure_text: 1T -->

<!-- p23-g101 figure_text: 10T -->

<!-- p23-g102 figure_text: Tokens -->

<!-- p23-g103 figure_text: 1.6T -->

<!-- p23-m1 footer: 23 -->

<!-- p24-m1 footer: 24 -->

<!-- p25-m1 footer: 25 -->

<!-- p26-g1 figure_text: 1010 1011 1012 1013 Tokens -->

<!-- p26-g2 figure_text: 108 -->

<!-- p26-g3 figure_text: 109 -->

<!-- p26-g4 figure_text: 1010 -->

<!-- p26-g5 figure_text: 1011 -->

<!-- p26-g6 figure_text: 1012 -->

<!-- p26-g7 figure_text: Parameters -->

<!-- p26-g8 figure_text: 1e+18 -->

<!-- p26-g9 figure_text: 1e+19 -->

<!-- p26-g10 figure_text: 1e+20 -->

<!-- p26-g11 figure_text: 1e+21 -->

<!-- p26-g12 figure_text: 1e+22 -->

<!-- p26-g13 figure_text: 1e+23 -->

<!-- p26-g14 figure_text: 1e+24 -->

<!-- p26-g15 figure_text: 1e+25 -->

<!-- p26-g16 figure_text: 1e+26 Approach 1 Approach 2 Approach 3 Chinchilla Gopher GPT-3 Megatron-Turing NLG -->

<!-- p26-m1 footer: 26 -->

<!-- p27-g1 figure_text: 0 1 2 Sequences 1e7 -->

<!-- p27-g2 figure_text: 2.2 -->

<!-- p27-g3 figure_text: 2.3 -->

<!-- p27-g4 figure_text: 2.4 -->

<!-- p27-g5 figure_text: 2.5 -->

<!-- p27-g6 figure_text: 2.6 -->

<!-- p27-g7 figure_text: 2.7 -->

<!-- p27-g8 figure_text: 2.8 -->

<!-- p27-g9 figure_text: Training Loss -->

<!-- p27-g10 figure_text: 2.2 -->

<!-- p27-g11 figure_text: 2.3 -->

<!-- p27-g12 figure_text: 2.4 -->

<!-- p27-g13 figure_text: 2.5 -->

<!-- p27-g14 figure_text: 2.6 -->

<!-- p27-g15 figure_text: 2.7 -->

<!-- p27-g16 figure_text: 2.8 -->

<!-- p27-g17 figure_text: Training Loss -->

<!-- p27-m1 footer: 27 -->

<!-- p28-g1 figure_text: 1017 1018 1019 1020 1021 1022 -->

<!-- p28-g2 figure_text: FLOPS -->

<!-- p28-g3 figure_text: 2.0 -->

<!-- p28-g4 figure_text: 2.5 -->

<!-- p28-g5 figure_text: 3.0 -->

<!-- p28-g6 figure_text: 3.5 -->

<!-- p28-g7 figure_text: 4.0 -->

<!-- p28-g8 figure_text: 4.5 -->

<!-- p28-g9 figure_text: 5.0 -->

<!-- p28-g10 figure_text: 5.5 -->

<!-- p28-g11 figure_text: 6.0 -->

<!-- p28-g12 figure_text: Training loss -->

<!-- p28-g13 figure_text: 75 -->

<!-- p28-g14 figure_text: 250 -->

<!-- p28-g15 figure_text: 500 -->

<!-- p28-g16 figure_text: 1000 -->

<!-- p28-g17 figure_text: 2500 -->

<!-- p28-g18 figure_text: 5000 -->

<!-- p28-g19 figure_text: 10000 -->

<!-- p28-g20 figure_text: Million Parameters -->

<!-- p28-m1 footer: 28 -->

<!-- p29-g1 figure_text: 0 5 10 15 20 25 30 Million Sequences -->

<!-- p29-g2 figure_text: 2.45 -->

<!-- p29-g3 figure_text: 2.50 -->

<!-- p29-g4 figure_text: 2.55 -->

<!-- p29-g5 figure_text: 2.60 -->

<!-- p29-g6 figure_text: 2.65 -->

<!-- p29-g7 figure_text: 2.70 -->

<!-- p29-g8 figure_text: Training Loss -->

<!-- p29-g9 figure_text: 0 5 10 15 20 25 30 Million Sequences -->

<!-- p29-g10 figure_text: 17 -->

<!-- p29-g11 figure_text: 18 -->

<!-- p29-g12 figure_text: 19 -->

<!-- p29-g13 figure_text: 20 -->

<!-- p29-g14 figure_text: 21 -->

<!-- p29-g15 figure_text: 22 -->

<!-- p29-g16 figure_text: 23 -->

<!-- p29-g17 figure_text: 24 -->

<!-- p29-g18 figure_text: 25 -->

<!-- p29-g19 figure_text: 26 -->

<!-- p29-g20 figure_text: Wikitext103 Perplexity -->

<!-- p29-g21 figure_text: 0 5 10 15 20 25 30 Million Sequences -->

<!-- p29-g22 figure_text: 2.60 -->

<!-- p29-g23 figure_text: 2.65 -->

<!-- p29-g24 figure_text: 2.70 -->

<!-- p29-g25 figure_text: 2.75 -->

<!-- p29-g26 figure_text: 2.80 -->

<!-- p29-g27 figure_text: 2.85 -->

<!-- p29-g28 figure_text: 2.90 -->

<!-- p29-g29 figure_text: 2.95 -->

<!-- p29-g30 figure_text: 3.00 -->

<!-- p29-g31 figure_text: C4 Loss -->

<!-- p29-g32 figure_text: Training Setup Adam w/ High Precision AdamW w/ High Precision Adam No High Precision AdamW No High Precision -->

<!-- p29-g33 figure_text: 0 25 50 75 100 125 150 Million Sequences -->

<!-- p29-g34 figure_text: 2.3 -->

<!-- p29-g35 figure_text: 2.4 -->

<!-- p29-g36 figure_text: 2.5 -->

<!-- p29-g37 figure_text: 2.6 -->

<!-- p29-g38 figure_text: 2.7 -->

<!-- p29-g39 figure_text: 2.8 -->

<!-- p29-g40 figure_text: C4 Loss -->

<!-- p29-g41 figure_text: 0 25 50 75 100 125 150 Million Sequences -->

<!-- p29-g42 figure_text: 10.0 -->

<!-- p29-g43 figure_text: 12.5 -->

<!-- p29-g44 figure_text: 15.0 -->

<!-- p29-g45 figure_text: 17.5 -->

<!-- p29-g46 figure_text: 20.0 -->

<!-- p29-g47 figure_text: 22.5 -->

<!-- p29-g48 figure_text: 25.0 -->

<!-- p29-g49 figure_text: 27.5 -->

<!-- p29-g50 figure_text: 30.0 -->

<!-- p29-g51 figure_text: Wikitext103 Perplexity -->

<!-- p29-g52 figure_text: 0 25 50 75 100 125 150 Million Sequences -->

<!-- p29-g53 figure_text: 0.0 -->

<!-- p29-g54 figure_text: 0.1 -->

<!-- p29-g55 figure_text: 0.2 -->

<!-- p29-g56 figure_text: 0.3 -->

<!-- p29-g57 figure_text: 0.4 -->

<!-- p29-g58 figure_text: 0.5 -->

<!-- p29-g59 figure_text: 0.6 -->

<!-- p29-g60 figure_text: LAMBADA Accuracy -->

<!-- p29-g61 figure_text: 417M, Adam 417M, AdamW 1.4B, Adam 1.4B, AdamW -->

<!-- p29-m1 footer: 29 -->

<!-- p30-m1 footer: 30 -->

<!-- p31-m1 footer: 31 -->

<!-- p32-g1 figure_text: Factors -->

<!-- p32-g2 figure_text: Metrics -->

<!-- p32-g3 figure_text: Evaluation Data -->

<!-- p32-m1 footer: 32 -->

<!-- p33-g1 figure_text: Datasets -->

<!-- p33-m1 footer: 33 -->

<!-- p34-m1 footer: 34 -->

<!-- p35-m1 footer: 35 -->

<!-- p36-m1 footer: 36 -->
