---
aliases:
  - DCN
tags:
  - recsys
  - architecture
  - neural-network
---
Deep & Cross Network (DCN) is a neural network architecture developed by Google that combines a cross network with a deep network to efficiently learn explicit and implicit feature interactions. The original DCN was introduced in 2017, with an improved version, DCN v2, released in 2020.
## Architecture
* Cross Network: takes the input embedding vector and computes cross-products with itself, each layer increases the polynomial degree by one. The input dimension is maintained throughout the network.
* Deep Network: MLP with [[ReLU]] activation function
* Combination: the outputs are concatenated and passed through a final dense layer to produce predictions (usually probability)
![[Pasted image 20250303082436.png]]

### DCN v2 Improvements
* Replaces the vector-based cross operation with the matrix-based using low rank decomposition
* Uses Mixture of Experts to decompose the learned matrix into sub-spaces, which are then aggregated with gates

![[Pasted image 20250303082534.png]]

==

## Advantages
- Automatically learns feature interactions without manual feature engineering
- Lower computational complexity than comparable models

## Disadvantages
- Original cross network limited in the types of interactions it can learn
- Hyperparameter tuning can be challenging

## Links

- [Original DCN Paper (2017)](https://arxiv.org/abs/1708.05123)
- [DCN v2 Paper (2020)](https://arxiv.org/abs/2008.13535)
- [TensorFlow Implementation](https://www.tensorflow.org/recommenders/examples/dcn)
