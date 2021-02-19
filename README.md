# Evaluating quantised Bayesian neural networks

by Martin Ferianc (martin.ferianc.19@ucl.ac.uk), Partha Maji, Matthew Mattina and Miguel Rodrigues

The paper can be found at: <LINK>

- [Evaluating quantised Bayesian neural networks](#evaluating-quantised-bayesian-neural-networks)
  - [Description](#description)
    - [Bayesian methods](#bayesian-methods)
  - [Structure](#structure)
    - [Experiments](#experiments)
    - [Requirements](#requirements)
    - [Metrics](#metrics)
  - [Authors](#authors)
  - [Credits](#credits)

## Description

Neural processing units use reduced precision for computation to save resources (memory, compute, MACs, OPs etc.). However, neural networks usually work with 32-bit floating-point. There has been recently a shift towards 8-bit inference as well as training. **Nevertheless, there has not been an investigation into whether it is possible to introduce *Bayesianity* with respect to quantisation and in general perform uncertainty quantification reliably in a quantised regime**. The investigation of this project is into whether this is possible, or how to make it possible.

### Bayesian methods

Standard pointwise deep learning tools have gained tremendous attention in applied machine learning. However, such tools for regression and classification do not capture model uncertainty. In comparison, Bayesian models offer a mathematically grounded framework to reason about model uncertainty, but usually come with a prohibitive computational cost [\[Gal et al., 2015\]](https://arxiv.org/abs/1506.02142).

In this repository, we demonstrate capabilities of multiple methods that introduce Bayesanity and uncertainty quantification to standard neural networks on multiple tasks. The tasks include *regression* on UCI datasets and synthetic data, *classification of MNIST digits* and *classification of CIFAR-10 images*. Each method/architecture is benchmarked against its pointwise counterpart. All hyperparameters were hand-tuned for best performance.

The architectures for *each method/experiment* include: 

- *Regression*: X inputs, 100 nodes, ReLU, 100 nodes, ReLU, 100 nodes, ReLU, 100 nodes for variance output (1), 100 nodes for mean output (1)
- *MNIST digit classification*: 1x28x28 inputs, 2D Convolution, 2D Max-pool, 2D Convolution, 2D Max-pool, 2450 nodes, ReLU, 500 nodes, 10 outputs
- *CIFAR-10 image classification*: 3x32x32 inputs, [ResNet-18](https://arxiv.org/abs/1512.03385) with enabled Batch normalization as well as skip-connections, 10 outputs

The same architecture is used to benchmark all the methods, including pointwise as well as Bayesian.

**Bayesian Implementation details**:

The implemented Bayesian methods are [Bayes-by-backprop](https://arxiv.org/abs/1505.05424) (with an added lower-variance [local reparametrization trick](https://arxiv.org/abs/1506.02557)), during training, however, during evaluation the local reparametrization trick is not used.

[Monte Carlo Dropout](https://arxiv.org/abs/1506.02142), which is applied before each computational layer (convolution, linear) except the input convolution/linear and the output layers. The dropout rate for each layer in the network is uniform across all operations.

[Stochastic Gradient Hamiltonian Monte Carlo](https://arxiv.org/abs/1402.4102), which is implemented as an ensemble with respect to architectures sampled at different times during the training. Note, that the architectures are sampled every-other epoch (every 2 epochs), to reduce correlation between the samples.

Each method has its own separate model definition or an optimiser for clarity and they are benchmarked under the same settings. The settings were hand-tuned, so if you find better ones definitely let us know.

**Quantisation Implementation details**:

Quantisation is developed with the help of PyTorch quantisation functionality [link](https://pytorch.org/docs/stable/quantization.html).

In general, we consider quantization-aware training, however, post-training quantisation can also be performed. We consider quantization of *weights* (including standard deviation for Bayes-by-backprop) as well as *activations* into different precisions, excluding biases that were introduced by folding batch-nomalisation into the convolutions. *All random number generation* is in 32-bit floating point. The hardware backend that has been assumed for the quantisation has been [FBGEMM](https://github.com/pytorch/FBGEMM). 

## Structure

```
   .
   |-experiments            # Where all the experiments are located
   |---data                 # Where all the data is located
   |---scripts              # Pre-configured scripts for running the experiments
   |-src                    # Main source folder, also containing the model descriptions
   |---models               # Model implementations for both pointwise architectures as well as Bayesian, under stochastic and pointwise respectively
   |-----pointwise
   |-----stochastic
```

### Experiments

There are in total three different experiments, regression, MNIST digit classification, CIFAR-10 image classification. The default runners for the experiments are under the `experiments` folder, where the scripts for easy pre-configured runs can be found under `experiments/scripts/`.

To run all the scripts at once make sure that you have installed all the requirements and you can simply run:

```
cd experiments/scripts && chmod +x run_all_float.sh && ./run_all_float.sh 0 # For the GPU id
# For the runs that you are happy with with respect to floating point you should then put them under a `default` directory 
# Under the same experiment folder (`float`) and then run to obtain the quantised results
cd experiments/scripts && chmod +x run_all_quant.sh && ./run_all_quant.sh 0 # For the GPU id
```

To be able to run the experiments individually simply navigate to the `experiments` folder and after activating the virtual environment, simply run any of the preconfigured scripts, such that:

```
cd experiments/scripts/stochastic/mcdropout/float/ && python3 mcdropout_mnist.py
```

Be careful, it is expected that by default you have a GPU available! If no GPU is available pass in a flag: `--gpu -1`. Also be careful that you need to first train floating-point models which only then can be quantized, and you can quantize them by running the appropriate script and then using the `--load` option to load in the floating-point module which will be then quantized. 

### Requirements

To be able to run this project and install the requirements simply execute (will work for GPUs or CPUs):

```
git clone <repository url>
python3 -m venv /path/to/new/virtual/environment
source /path/to/new/virtual/environment/bin/activate
pip3 install -r requirements.txt
```

The main requirement is PyTorch==1.7.0, because it supports custom quantisation modules, which need to be implemented for each Bayesian method.

Additionally, there is an existing bug in PyTorch where the quantisation bounds are not passed to the siumlated quantisation module and it is necessary to modify its constructor such that: 

```python
# In torch/quantization/fake_quantize.py
class FakeQuantize(Module):
    def __init__(self, observer=MovingAverageMinMaxObserver, quant_min=0, quant_max=255, **observer_kwargs):
        super(FakeQuantize, self).__init__()
        assert quant_min <= quant_max, \
            'quant_min must be less than or equal to quant_max'
        self.quant_min = quant_min
        self.quant_max = quant_max
        self.register_buffer('fake_quant_enabled', torch.tensor([1], dtype=torch.uint8))
        self.register_buffer('observer_enabled', torch.tensor([1], dtype=torch.uint8))
        # This line
        self.activation_post_process = observer(quant_min=quant_min, quant_max=quant_max, **observer_kwargs)
        # Instead of self.activation_post_process = observer(**observer_kwargs)
```
Without this it is not possible to simulate below 8-bit quantisation for weights or activations. Also notice the clamping that needs to be performed with respect to all quantised tensors to avoid overflow or underflow because PyTorch does not check this at all at runtime. 

### Metrics

We used a combined set of metrics to measure both the accuracy and the quality of uncertainty estimation under quantsation. For *regression*, we focused on the root-mean-squared-error (RMSE) and average negative-log-likelihood (NLL). For both *MNIST* and *CIFAR-10*, we looked at the classification error, average negative-log-likelihood (NLL), average predictive entropy (aPE) and expected calibration error (ECE). 

## Authors

Martin Ferianc (martin.ferianc.19@ucl.ac.uk), Partha Maji, Matthew Mattina and Miguel Rodrigues

All source code is made available under a BSD 3-clause license. You can freely use and modify the code, without warranty, so long as you provide attribution to the authors. See LICENSE.md for the full license text.

The manuscript text is not open source. The authors reserve the rights to the article content. If you use ideas presented in this work please cite our work: 

```bibtex
<CITATION>

```


## Credits

- Pre-processing, loading of regression datasets and SGHMC implementation: https://github.com/JavierAntoran/Bayesian-Neural-Networks 
