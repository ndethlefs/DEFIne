# Code for "DEFIne: A Fluent Interface DSL for Deep Learning Applications". 

This repository contains the code for the following paper:

N Dethlefs and K Hawick (2017) DEFIne: A Fluent Interface DSL for Deep Learning Applications. Proceedings of the 2nd International Workshop on Real World Domain Specific Languages. ACM Digital Library International Conference Proceedings Series (ICPS). Austin, Texas, US.

See link here: https://dl.acm.org/doi/10.1145/3039895.3039898

The basic idea with this paper is to shorten the process from idea to implementation of well-understood standard neural
network architectures for new tasks or datasets. DEFIne provides a set of simple commands that can be combined and 
chained together to pre-process a new dataset, compare a variety of neural networks, generate benchmark results and 
visualisations. It is build in Python on top of deep  learning libraries Keras and Theano. 

<figure>
<img src="/img/architecture.png" alt="drawing" width="400"/>
<figcaption>
Architecture for code optimisation and generation.
 </figcaption>
</figure>


In the paper, we present a proof-of-concept results for heart disease diagnosis, hand-written digit recognition and 
weather forecast generation. Results in terms of accuracy, runtime and lines of code show that our domain-specific language (DSL) 
achieves equivalent accuracy and runtime to state-of-the-art models, while requiring only about 10 lines of code per application.

<figure>
<img src="/img/results.png" alt="drawing" width="400"/>
<figcaption>
<em>Performance results (including hyper-parameter optimisation and runtime comparison) in a number of domains.</em>
 </figcaption>
</figure>
</br></br></br>

# Code

DEFIne can be run from the <code>test.py</code> example file. The <code>config</code> files (any format) contains default values. 

<figure>
<img src="/img/code.png" alt="drawing" width="400"/>
<figcaption>
Minimal DEFIne code example for training a 2-layered multi-layer perceptron. 
 </figcaption>
 </figuer>

The <code>data_repository</code> contains example datasets for a variety of classification and regression tasks.

Models can be trained from scratch (using <code>text.py</code>) or from prior knowledge. For the latter, please
use <code>from_prior.py</code>. Json files <code>knowledge.json</code> and <code>knowledge_regression.json</code> contain pre-trained models to facilitate 
and accelerate training of new models, but need not be adapted.

This is a later version of code that adds hyper-parameter optimisation and model definition from configuration files beyond what 
was presented in the paper.

