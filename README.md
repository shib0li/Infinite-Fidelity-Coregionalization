# *IFC-ODE$*: Infinite-Fidelity Coregionalization for Physical Simulation

by [Shibo Li](https://imshibo.com), Wang Zheng, [Mike Kirby](https://www.cs.utah.edu/~kirby/) and [Shandian Zhe](https://www.cs.utah.edu/~zhe/)

<!-- <p align="center">
    <br>
    <img src="images/THIS-ODE.png" width="500" />
    <br>
<p>

<h4 align="center">
    <p>
        <a href="https://proceedings.mlr.press/v162/li22i.html">Paper</a> |
        <a href="https://github.com/shib0li/THIS-ODE/blob/main/images/slides-v2.pdf">Slides</a> |
        <a href="https://github.com/shib0li/THIS-ODE/blob/main/images/923-poster.png">Poster</a> 
    <p>
</h4> -->


Multi-fidelity modeling and learning is important in physical simulation related applications. It can leverage both low-fidelity and high-fidelity examples for training so as to reduce the cost of data generation yet still achieving good performance. While existing approaches only model finite, discrete fidelities, in practice, the feasible fidelity choice is often infinite, which can correspond to a continuous mesh spacing or finite element length.   In this paper, we propose Infinite Fidelity Coregionalization (IFC). Given the data, our method can extract and exploit rich information within infinite, continuous fidelities to bolster the prediction accuracy. Our model can interpolate and/or extrapolate the predictions to novel fidelities that are not covered by the training data. Specifically, we introduce a low-dimensional latent output as a continuous function of the fidelity and input, and multiple it with a basis matrix to predict high-dimensional solution outputs. We model the latent output as a neural Ordinary Differential Equation (ODE) to capture the complex relationships within and integrate information throughout the continuous fidelities.  We then use Gaussian processes or another ODE to estimate the fidelity-varying bases. For efficient inference, we reorganize the bases as a tensor, and use a tensor-Gaussian variational posterior approximation to develop a scalable inference algorithm for massive outputs. We show the advantage of our method in several benchmark tasks in computational physics. 


IFC$^2$

# System Requirements

We highly recommend to use Docker to run our code. We have attached the docker build file `env.Dockerfile`. Or feel free to install the packages with pip/conda that could be found in the docker file.

# Datasets

In our paper, we conduct our experiments on 4 real-world tasks and 1 synthetic task. For each task, we evaluate all the methods on two settings: ***extrapolation*** and ***interpolation***. 

You should download original raw datasets([FitRec](https://cseweb.ucsd.edu/~jmcauley/datasets.html#google_local), [Time series events](https://github.com/snudatalab/TATD)) into the `data/raw/(domain)` folders and use the notebooks provided to process the data. If propoerly processed, the data used for training and testing should appear as the `data/processed/BeijingAirExtrap.pickle` for example

```
data/
├── raw/
│   └── Beijing/
│       └──...
│   └── FitRec/
│       └──...
├── processed
├── (notebooks)
```

# Run

To run all the methods, use our provided `run.sh` script, this script takes several arguments

```
./run.sh $DOMAIN $METHOD $MAX_EPOCHS $RANK $DEVICE $FOLD $BATCH_SIZE $TEST_INTERVAL 

```

* `$DOMAIN` is a concat string that specifies the task and setting, for example ***BeijingAirExtrap*** means run with BeijingAir data on extrapolation setting.
* `$METHOD` learning algorithms
* `$MAX_EPOCHS` an integer of number of epochs
* `$RANK` dimension used for decomposition representation
* `$DEVICE` where to run, for example ***cuda:0*** or ***cpu***
* `$FOLD` fold index after splitting the datasets
* `$BATCH_SIZE` mini-batch size used for training
* `$TEST_INTERVAL` frequency for saving the results

for example

```
bash run.sh BeijingAirExtrap Neural_time 10 8 cuda:0 0 100 1 
```

runs the experiment of BeijingAir on the extraplation setting with the algorihtm of neural_time for 10 epochs. The dimensional of representation is 8 and the program runs on the first GPU. The minibatch size is 100 and save the reulst every 1 update.

Here are the name mappings of the names we used in the paper and the methods we used in the code.

| Names in the paper | Names in this repo |
|:------------------:|:------------------:|
|       CP-Time      |      CPTF_time     |
|       CP-DTL       |     CPTF_linear    |
|       CP-DTN       |      CPTF_rnn      |
|      GPTF-Time     |      GPTF_time     |
|      GPTF-DTL      |     GPTF_linear    |
|      GPTF-DTN      |      GPTF_rnn      |
|      NTF-Time      |     Neural_time    |
|       NTF-DTL      |    Neural_linear   |
|       NTF-DTN      |     Neural_rnn     |
|       PTucker      |       Tucker       |
|      THIS-ODE      |     NODE_noise     |

# License

IFC is released under the MIT License, please refer the LICENSE for details

# Getting Involved
Feel free to submit Github issues or pull requests. Welcome to contribute to our project!

To contact us, never hestitate to send an email to `shibo@cs.utah.edu` or `shiboli.cs@gmail.com` 
<br></br>


# Citation
Please cite our paper if you find it helpful :)

```

@inproceedings{
li2022infinitefidelity,
title={Infinite-Fidelity Coregionalization  for Physical Simulation},
author={Shibo Li and Zheng Wang and Robert Kirby and Shandian Zhe},
booktitle={Advances in Neural Information Processing Systems},
editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
year={2022},
url={https://openreview.net/forum?id=dUYLikScE-}
}

```
<br></br>
