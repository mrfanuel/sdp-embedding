# Positive semi-definite embedding for dimensionality reduction
This is the code associated to the following paper:

Michaël Fanuel, Antoine Aspeel, Jean-Charles Delvenne, Johan A.K. Suykens, [Positive semi-definite embedding for dimensionality reduction and out-of-sample extensions](https://doi.org/10.1137/20M1370653), published in SIAM journal of mathematics for data science, [https://arxiv.org/abs/1711.07271](https://arxiv.org/abs/1711.07271)

### Datasets

- The [HTRU](https://archive.ics.uci.edu/ml/datasets/HTRU2) dataset was preprocessed and saved in .mat format. It is available in the Data folder.
- For the [MNIST](http://yann.lecun.com/exdb/mnist/) data, please download in the Data folder the following files:
  - train-images-idx3-ubyte.gz
  - train-labels-idx1-ubyte.gz
  - t10k-images-idx3-ubyte.gz
  - t10k-labels-idx1-ubyte.gz
### Running demos
In the Demos folder, you can run the following scripts:
- `interval_oos.m` (Fig. 2 and Fig. 3)
- `MNIST_embed_45.m`  (Fig. 4)
- `wine_embed.m` (Fig. 5)
- `quasar_embed_classifier.m` (Fig. 6)
- `two_moons_plots.m` (Fig. 7)
- `two_moons_benchmark.m`(Fig. 8)
  
Note that the eigenvalue decomposition of the diffusion embedding in `two_moons_benchmark.m` may fail to converge for a very small kernel bandwidth.
### Dependencies
- MATLAB R2019b
- Statistics and Machine Learning Toolbox
