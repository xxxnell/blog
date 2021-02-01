---
layout: post
title: "Temporal Smoothing for Efficient Uncertainty Estimation"
description: "We show that temporal smoothing is an efficient method to estimate uncertainty in video stream processing."
issue: 10
tags: [nn, bnn]
lang: en
---


## TL; DR

* Bayesian neural networks (BNNs) predict not only predictive results but also uncertainties, so it is an effective method to build a trustworthy AI system. However, Bayesian neural network inference is significantly slow. To tackle this problem, we proposed a novel method called *vector quantized Bayesian neural network* (VQ-BNN) inference that uses previously memorized predictions. 
* We propose *temporal smoothing* of predictions with exponentially decaying importance or *exponential moving average* by applying VQ-BNN to data streams.
* Temporal smoothing is an easy-to-implement method that performs significantly faster than BNNs while estimating predictive results comparable to or even superior to the results of BNNs.

Translation: [Korean](/ko/posts/temporal-smoothing)


## Recall: vector quantized Bayesian neural network improves inference speed by using previously memorized predictions

Bayesian neural networks (BNNs) predict not only predictive results but also uncertainties. However, in the previous post ["Vector Quantized Bayesian Neural Network for Efficient Inference"](/posts/vqbnn), we raised the problem that BNNs are significantly slower than traditional neural networks or deterministic NNs. To solve this problem, we also proposed vector quantized Bayesian neural network (VQ-BNN) that improves the inference speed by using previously memorized predictions.

{% include image.html path="documentation/vqbnn/diagrams/bnn-inference" path-detail="documentation/vqbnn/diagrams/bnn-inference" alt="bnn-inference" %}

The figure above represents BNN inference. In short, BNN inference is Bayesian neural net ensemble average. BNN need iterative NN executions to predict a result for one data, and it gives raise to prohivitive comptuational cost. 


{% include image.html path="documentation/vqbnn/diagrams/vqbnn-inference" path-detail="documentation/vqbnn/diagrams/vqbnn-inference" alt="vqbnn-inference" %}

The figure above represents VQ-BNN inference. VQ-BNN inference makes a prediction for an input data *only once*, and compensates the predictive result with previously memorized predictions. Here, the importance is defined as the similarity between the observed and memorized data. VQ-BNN is an efficient method since it only needs one newly calculated prediction.

VQ-BNN inference needs *prototype* and *importance*. For computational efficiency, they have to meet the following requirements:

1. **Prototype** should consist of proximate datasets.
2. **Importance** should be easy to calculate.

Data stream analysis, especially video analysis, is an area where latency is important. Videos are large and video analysis sometimes requires real-time processing. Therefore, it is not practical to use BNN in this area because BNN inference is too slow. Instead, this post shows that VQ-BNN can process video streams easily and efficiently.



## Real-world data streams are continuously chaning

In order to use VQ-BNN as the approximation theory of BNN for data streams, we exploits the property that most real-world data streams change continuously. We call it *temporal consistency* or *temporal proximity* of data streams.

  {% include image.html path="documentation/temporal-smoothing/diagrams/consistency" path-detail="documentation/temporal-smoothing/diagrams/consistency" alt="consistency" %}

The figure above shows an example of the temporal consistency of a video sequence. In this video stream, a car is moving slowly and continuously from right to left as timestamp $$t$$ increases. Here, let the most recent frame $$t=0$$ be the observed input data.

Thanks to this temporal consistency, we simply take $$K$$ recent data as prototypes $$\mathcal{S}$$: 

$$
\mathcal{S} = \{ \textbf{x}_t \vert 0 \geq t \geq -K \}
$$

Similary, we propose an simple importance model which is defined as the similarity between the latest and memorized data. As shown below, it decreases exponentially over time:

$$
\pi(\textbf{x}_{t} \vert \mathcal{S}) = \frac{\exp(- t / \tau)}{\sum_{t=0}^{-K} \exp(- t / \tau)}
$$

where hyperparameter $$\tau$$ is decaying rate. The denominator is a normalizing constant for importance.

Taken together, VQ-BNN inference for data streams is just *temporal smoothing* or *exponential moving average* (EMA) of recent NN predictions $$ p(\textbf{y} \vert \textbf{x}_t, \textbf{w}_t) $$ at time $$t$$:

$$
p(\textbf{y} \vert \mathcal{S}, \mathcal{D}) \simeq  \sum_{t=0}^{-K} \alpha \exp(- t / \tau) \, p(\textbf{y} \vert \textbf{x}_t, \textbf{w}_t) 
$$

where $$ \alpha = \left({\sum_{t=0}^{-\infty} \exp(- t / \tau)} \right)^{-1} $$ is a normalizing constant.

In order to calculate VQ-BNN inference, we have to determine the prediction $$ p(\textbf{y} \vert \textbf{x}_t, \textbf{w}_t) $$ parameterized by NN. For classification tasks, we set $$ p(\textbf{y} \vert \textbf{x}_t, \textbf{w}_t) $$ as a categorical distribution parameterized by the `Softmax` of NN logit:

$$
p(\textbf{y} \vert \mathcal{S}, \mathcal{D}) \simeq \sum_{t=0}^{-K} \alpha \exp(- t / \tau) \, \texttt{Softmax}(\text{NN} (\textbf{x}_{t}, \textbf{w}_{t}))
$$

where $$\text{NN} (\textbf{x}_{t}, \textbf{w}_{t})$$ is NN logit, e.g. a prediction of NN with MC dropout layers, at time $$t$$. 



<!--
$$
p(\textbf{y} \vert \mathcal{S}, \mathcal{D}) \simeq  \sum_{t=0}^{-\infty} \alpha \exp(- t) \, p(\textbf{y} \vert \textbf{x}_t, \textbf{w}_t) 
$$

$$
p(\textbf{y} \vert \mathcal{S}, \mathcal{D}) = \alpha \, p(\textbf{y} \vert \textbf{x}_{0}, \textbf{w}_{0}) + (1 - \alpha) \, p_{-1}(\textbf{y} \vert \mathcal{S}, \mathcal{D})
$$
-->



## Temporal smoothing for semantic segmentation


We have previously shown that VQ-BNN for a data stream is a temporal smoothing of recent NN predictions. From now on, let's apply VQ-BNN inference---i.e., temporal smoothing---to the real-world data streams. In this section, we use [MC dropout](https://arxiv.org/abs/1506.02142) as a BNN approximation. 


  {% include image.html path="documentation/temporal-smoothing/diagrams-bnn" path-detail="documentation/temporal-smoothing/diagrams-bnn" alt="bnn-inference" %}

Let's take an example of semantic segmentation, which is a pixel-wise classification, on real-world video sequence. MC dropout predicts the `Softmax` (not `Argmax`) probability multiple times for one video frame, and averages them. It generally requires 30 samples to achieve high predictive performance, so the inference speed is decreased by 30 times accordingly.

One more thing we would like to mention is that BNN is highly dependent on input data. The input frame can be an outlier, by motion blur or video defocus or anything else. When the input frame is noisy, BNN may give an erroneous result. In this example, the window of the car is a difficult part to classify. So, most predictions of BNN are incorrect results in this case.


  {% include image.html path="documentation/temporal-smoothing/diagrams-vqbnn" path-detail="documentation/temporal-smoothing/diagrams-vqbnn" alt="vqbnn-inference" %}

In contrast, VQ-BNN predicts the result for the latest frame only once, and compensates it with previously memorized predictions. It is easy to memorize the sequence of NN predictions, so the computational performance of VQ-BNN is almost the same as that of deterministic NN.

Previously, we mentioned that BNN is overly dependent on input. What about VQ-BNN? In this example, the prediction for the most recent frame is incorrect. So far, it is the same as that of BNN. However, VQ-BNN smoothen the result by using past predictions, and past predictions classify the window of the car correctly. It implies that the results of VQ-BNN are robust to the noise of data such as motion blur.



### Qualitative results

<div style="width:85%;margin:auto;">
  {% include image.html path="documentation/temporal-smoothing/diagrams/result" path-detail="documentation/temporal-smoothing/diagrams/result" alt="semantic-segmentation-result" %}
</div>

Let's check the results. The second row shows the predictive results---i.e., `Argmax` of the predictive distributions---of BNN and VQ-BNN. As we expected, the result of BNN is incorrectly classified. In contrast, VQ-BNN gives a more accurate result than BNN.

The third row in this figure shows the predictive confidence---i.e., `Max` of the predictive distributions. The brighter the background, the higher confidence, that is the lower uncertainty. According to these confidences, VQ-BNN is less likely to be overconfident than BNN. This is because VQ-BNN uses both NN weight distribution and a data distribution $$p(\textbf{x} \vert \mathcal{S})$$ at the same time.

<div style="width:45%;margin:auto;">
  {% include image.html path="documentation/temporal-smoothing/semantic-segmentation/sequence/input-seq1.gif" path-detail="documentation/temporal-smoothing/semantic-segmentation/sequence/input-seq1.gif" alt="input-seq" %}
</div>

For a better understanding, let's compare the predictive results of each method for the above video sequence. We use vanilla deterministic neural network (DNN) and BNN (MC dropout) as baselines. And we compare them to the temporal smoothing of DNN and BNN, called VQ-DNN and VQ-BNN respectively.


  {% include image.html path="documentation/temporal-smoothing/semantic-segmentation/sequence.gif" path-detail="documentation/temporal-smoothing/semantic-segmentation/sequence.gif" alt="vqbnn-unc-seq" %}

<!--
<table cellspacing="3" style="width:100%;text-align:center;">
  <tr>
    <th style="font-size:18px">DNN <div style="font-size:16px"></div></th>
    <th style="font-size:18px">VQ-DNN <div style="font-size:16px"></div></th>
    <th style="font-size:18px">BNN <div style="font-size:16px"></div></th>
    <th style="font-size:18px">VQ-BNN <div style="font-size:16px"></div></th>
  </tr>
  <tr>
    <td>
  {% include image.html path="documentation/temporal-smoothing/semantic-segmentation/sequence/dnn-res-seq1.gif" path-detail="documentation/temporal-smoothing/semantic-segmentation/sequence/dnn-res-seq1.gif" alt="dnn-res-seq" %}
    </td>
    <td>
  {% include image.html path="documentation/temporal-smoothing/semantic-segmentation/sequence/vqdnn-res-seq1.gif" path-detail="documentation/temporal-smoothing/semantic-segmentation/sequence/vqdnn-res-seq1.gif" alt="vqdnn-res-seq" %}
    </td>
    <td>
  {% include image.html path="documentation/temporal-smoothing/semantic-segmentation/sequence/bnn-res-seq1.gif" path-detail="documentation/temporal-smoothing/semantic-segmentation/sequence/bnn-res-seq1.gif" alt="bnn-res-seq" %}
    </td>
    <td>
  {% include image.html path="documentation/temporal-smoothing/semantic-segmentation/sequence/vqbnn-res-seq1.gif" path-detail="documentation/temporal-smoothing/semantic-segmentation/sequence/vqbnn-res-seq1.gif" alt="vqbnn-res-seq" %}
    </td>
  </tr>
  <tr>
    <td>
  {% include image.html path="documentation/temporal-smoothing/semantic-segmentation/sequence/dnn-unc-seq1.gif" path-detail="documentation/temporal-smoothing/semantic-segmentation/sequence/dnn-unc-seq1.gif" alt="dnn-unc-seq" %}
    </td>
    <td>
  {% include image.html path="documentation/temporal-smoothing/semantic-segmentation/sequence/vqdnn-unc-seq1" path-detail="documentation/temporal-smoothing/semantic-segmentation/sequence/vqdnn-unc-seq1.gif" alt="vqdnn-unc-seq1" %}
    </td>
    <td>
  {% include image.html path="documentation/temporal-smoothing/semantic-segmentation/sequence/bnn-unc-seq1.gif" path-detail="documentation/temporal-smoothing/semantic-segmentation/sequence/bnn-unc-seq1.gif" alt="bnn-unc-seq" %}
    </td>
    <td>
  {% include image.html path="documentation/temporal-smoothing/semantic-segmentation/sequence/vqbnn-unc-seq1.gif" path-detail="documentation/temporal-smoothing/semantic-segmentation/sequence/vqbnn-unc-seq1.gif" alt="vqbnn-unc-seq" %}
    </td>
  </tr>
</table> 
-->

These are animations of predictive results. The first column is the results of deterministic NN and BNN, and the second column is their temporal smoothings.

In these videos, the predictive results of deterministic NN and BNN are *noisy*. Their classification results change irregularly and randomly. This phenomenon is widely observed not only in semantic segmentation, but also in image processing using deep learning. For example, in object detection, consider a case where the size of the bounding box changes discontinuously and sometimes disappears. In contrast, temporal smoothing of deterministic NN’s and BNN’s results are *stabilized*. They change smoothly. So, we might get more natural results by using temporal smoothing.


### Quantitative results

We previously mentioned that VQ-BNN may give a more accurate result than BNN, when the inputs are noisy. The quantitative results support the speculation.

<style>
.styled-table {
    border-collapse: collapse;
    margin: 30px auto;
    font-size: 17x;
}

.styled-table thead tr {
    background-color: #009879;
    color: #ffffff;
    text-align: left;
}

.styled-table th,
.styled-table td {
    padding: 10px 1%;
}

.styled-table tbody tr:nth-of-type(even) {}

.styled-table tbody tr:first-of-type {
    border-top: 2px solid black;
    border-bottom: 1px solid black;
}

.styled-table tbody tr:last-of-type {
    border-bottom: 2px solid black;
}

.styled-table tbody tr.active-row {
    font-weight: bold;
    color: #009879;
}
</style>

<table cellspacing="3" style="width:90%;text-align:center;" class="styled-table">
  <tr>
    <th>Method</th>
    <th>Rel Thr<div>(%, ↑)</div></th>
    <th>NLL <div>(↓)</div></th>
    <th>Acc <div>(%, ↑)</div></th>
    <th>ECE <div>(%, ↓)</div></th>
  </tr>
  <tr>
    <td>DNN</td><td><b>100</b></td><td>0.314</td><td>91.1</td><td>4.31</td>
  </tr>
  <tr>
    <td>BNN</td><td>2.99</td><td>0.276</td><td>91.8</td><td>3.71</td>
  </tr>
  <tr>
    <td>VQ-DNN</td><td>98.2</td><td>0.284</td><td>91.2</td><td>3.00</td>
  </tr>
  <tr>
    <td>VQ-BNN</td><td>92.7</td><td><b>0.253</b></td><td><b>92.0</b></td><td><b>2.24</b></td>
  </tr>
</table>

This table shows the performance of the methods with semantic segmentation task on the CamVid dataset. We use arrows to indicate which direction is better.

First of all, we measure relative throughput (*Rel Thr*), which is the relative number of video frames processed per second. In this experiment, we use MC dropout with 30 forward passes to predict results, so the throughput of BNN is only 1/30 of that of deterministic NN. In contrast, the inference speed of VQ-BNN is comparable to that of deterministic NN, and 30✕ higher than that of BNN.

We also measure a trio for this semantic segmentation task. One is a *NLL* which is a proper scoring rule, two is a global pixel accuracy (*Acc*), and the last one is an expected calibration error (*ECE*) to measure the uncertainty reliability. In terms of these metrics, the predictive performance of BNN is obviously better than that of deterministic NN. More important thing is, VQ-BNN predicts more accurate results than BNN. Similarly, the results of VQ-DNN show that temporal smoothing improves predictive performance even without using BNN.

When we use [deep ensemble](https://arxiv.org/abs/1612.01474) instead of MC dropout, we obtain similar results. NLL of deep ensemble with 5 models is 0.216, and NLL of temporal smoothing with deep ensemble is 0.235 which is comparable to the result of the deep ensemble.



<div style="width:85%;margin:auto;">
  {% include image.html path="documentation/temporal-smoothing/semantic-segmentation/reliability-diagram-extended" path-detail="documentation/temporal-smoothing/semantic-segmentation/reliability-diagram-extended" alt="reliability-diagram" %}
</div>

This reliability diagram also shows consistent results that temporal smoothing is an effective method to calibrate results. As shown in this figure, deterministic NN is miscalibrated. In contrast, VQ-BNN is better calibrated than deterministic NN, and surprisingly better than BNN. Likewise, VQ-DNN is better calibrated than deterministic NN and BNN.  

In conclusion, *using knowledge from the previous time steps is useful* for improving predictive performance and estimating uncertainties. Temporal smoothing is an easy-to-implement method that significantly speeds up Bayesian NN inference without loss of accuracy.



## Further reading

* This post is based on the paper ["Vector Quantized Bayesian Neural Network Inference for Data Streams"](https://arxiv.org/abs/1907.05911). For more detailed information on VQ-BNN, please refer to the paper. For the implementation of VQ-BNN, please refer to [GitHub](https://github.com/xxxnell/temporal-smoothing). If you find this post or the paper useful, please consider citing the paper. Please contact me with any comments or feedback.
* For more qualitative results of semantic segmentation, please refer to [GitHub](https://github.com/xxxnell/temporal-smoothing/blob/master/resources/README.md).
* We have shown that predictive performance can be further improved by using *future predictions*---as well as past predictions. For more detailed informations, please refer to Appendix D.1 and Figure 11 in the [paper](https://arxiv.org/abs/1907.05911).
* The figure below shows an example of the results and uncertainties for each method with monochronic depth estimation task. In this example, we observe that the uncertainty represented by VQ-DNN differs from the uncertainty represented by BNN. VQ-BNN contains both types of uncertainties. For more detailed informations, please refer to Appendix D.2 in the [paper](https://arxiv.org/abs/1907.05911).

  {% include image.html path="documentation/temporal-smoothing/depth-estimation/visualize/result" path-detail="documentation/temporal-smoothing/depth-estimation/visualize/result" alt="depth-estimation-result" %}
  
  
  
  
  




