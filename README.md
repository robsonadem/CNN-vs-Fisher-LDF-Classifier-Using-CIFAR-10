# CNN-vs-Fisher-LDF-Classifier-Using-CIFAR-10
Making use of the CIFAR-10dataset, I implemented the following two training
and classification methods and analyzed their respective performance.

## Fisher LDF:

Trained and tested a 10-class Fisher Linear Discriminant Function classifier
using the Mahalanobis distance as the metric for classification. 


## CNN:

Based on the above configurations, I trained 50,000 images from the CIFAR- 10
data using the convolutional neural net, and then used 10,000 images for
classification. For learning, iterations of 350, Epoch of 40, and learning rate of
0.001 were used. 

Following the learning process, I run a test on the 10,000 images from the
CIFAR-10 test batch. I have tabulated the confusion matrix as follows.


## Analysis:

From the above observations, we can conclude that the Fisher LDF algorithm
falls short when compared with the convolutional neural net.
