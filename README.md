# Automatic Detection and Classification of Rock Microstructures through Machine Learning

This repo accompanies my poster presentation at AGU Fall Meeting 2021.
Here, you will find the original poster, along with code to replicate the original experiments, or to adapt to other similar tasks.

To see the original presentation, see DOI [10.1002/essoar.10508800.1](https://doi.org/10.1002/essoar.10508800.1).

## Motivation.
Geologists need help classifying microscope rock images of sigma clasts; a microstructure widely used as kinematic indicators in rocks. Knowledge about the shear sense of a sigma clast during formation (either CCW or CW shearing) gives insights into rock formation history.  In this work, we report on early investigation of deep learning approaches for the automatic detection and classification of sigma clasts and their rotation from photomicrographs.

Moreover, since traditional deep learning approaches simply predict outcomes rather than providing an intuition behind the decisions leading to the outcome, we set out to make our sigma clast classification pipeline more interpretable, i.e. understand why it’s making the predictions it is making. So we set out to answer the following questions:

* Where is the network “looking” in the input image?
* Which series of neurons are activated during training or prediction?
* How does the network arrive at its final output?

<!--For, if we cannot answer these questions, how can we trust the decisions of our model?-->

## Methods.
We used Convolutional Neural Networks (CNNs) to extract and leverage defining features of sigma clasts, such as shape, color, texture, and tail direction to improve accuracy. Due to limited availability of geological data, we leverage existing models that are pre-trained on very large collections of images, and use transfer learning techniques to apply them to microstructure images. We experimented with large pre-trained models such as ResNet50, VGG19, Inceptionv3 with two additional layers trained specifically on our dataset. Additionally, we used YOLOv3, an object segmentation algorithm, to identify different sigma clasts in a given image.


## Interpretability.
To try to answer about interpretability, we implemented a Grad-CAM heatmap filter [[Selvaraju et al., 2016](https://arxiv.org/abs/1610.02391)]. Grad-CAM uses the gradients of the input image going into the final convolutional layer of a network to produce a heatmap highlighting the most salient regions in the image used for label prediction. Using Grad-CAM now allows us to visually validate where the network is “looking”, providing some human-interpretable intuition about what patterns in the input image the network thinks is important for classification. Most interestingly, we can look at false positive and false negative examples, and see why the network got confused. Was it due to poor data quality, different lighting conditions, or something else.

Next, to keep experimental results more organized, we’ve added TensorBoard support to our network training routines. TensorBoard now keeps track of all our training logs with: useful comparative metrics (e.g. training/validation loss, f1 score), visualizations of the network’s computational graph, training parameters and hyperparams, as well as the Grad-CAM heatmaps used for human-interpretable analysis. Furthermore, we are now able to view histograms of the weights, biases, and other tensors as they change over time during training, allowing for individual neuron analysis. This gives us a very fine picture of which neurons are activated during training or inference, and by which images. This is important in the transfer learning setting, where a large part of our model architecture is frozen, and we only control a few final classification layers. With this addition to the training routines, we are now able to better keep track of experimental results of each model, quickly quantify the performance of different approaches (e.g. architectures, hyperparams), and fine-tune models faster.

A final upshot of this generalized experimentation environment is it can be adapted and used in adjacent projects to improve ML model and pipeline interpretability.

## Citation
```
@article{iota2021microstructures,
	author = {Iota, Stephen and Liu, Junyi and Lyu, Ming and Pan, Bolong and Wang, Xiaoyu and Gil, Yolanda and Gill, Gurman and AbdAlmageed, Wael and Mookerjee, Matty},
	title = {Automatic Detection and Classification of Rock Microstructures through Machine Learning},
	journal = {Earth and Space Science Open Archive, AGU Fall Meeting},
	pages = {1},
	year = {2021},
	DOI = {10.1002/essoar.10508800.1},
	url = {https://doi.org/10.1002/essoar.10508800.1},
}
```
