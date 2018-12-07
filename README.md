# vdnn-plus-plus
vDNN++ is an improvement in design over vDNN (https://arxiv.org/abs/1602.08124). This repository contain my implementation of
vDNN and vDNN++. Currently, this supports linear networks.

vDNN++ removes synchronization at the end of computation of each layer. 
It support different heuristics and reduces memory fragmentation.
It demonstrates the feasibility of compression on CPU-side for reducing pinned memory usage.
