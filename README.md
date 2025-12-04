This repository implements various network pruning techniques to optimize VGG16-BN models for edge devices and evaluated them based off of Model Size (MB), Peak/Avg GPU Memory, Latency (ms), Energy Consumption (mJ), and MACs.
The techniques included:
- Unstructured Pruning: individual weights are zeroed out regardless of structure
- Magnitude-Based Pruning: A standard baseline technique where weights with the lowest absolute values ($|W|$) are pruned first.
- GrasP (Gradient Signal Preservation): A saliency-based approach that iteratively prunes connections by analyzing the Hessian-gradient product. This preserves gradient flow rather than just weight magnitude, theoretically allowing for higher sparsity at better accuracy.
- Sparse Inference: Implemented custom forward passes using torch.sparse COO tensors to handle sparse-dense matrix multiplication.
- Structured Pruning (Channel Pruning): We implemented the regression-based method from He et al. (2017) to physically remove entire convolutional filters (channels), resulting in actual hardware acceleration without specialized sparse kernels.
  - LASSO Regression (Channel Selection): Formulated filter selection as a LASSO optimization problem to identify the most representative channels that minimize reconstruction error.
  - Least Squares (Weight Reconstruction): After pruning channels, we solved a standard Least Squares problem to adjust the remaining weights and restore feature map fidelity.
