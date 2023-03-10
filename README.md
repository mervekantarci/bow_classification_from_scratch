# Bag-of-Words for Image Classification from Scratch

SIFT descriptors are used to collect image features.
Dictionary is built using k-means clustering with various dictionary sizes(k).
Support-Vector classifier is preferred.
SVC experiments include; various regularization factors, weighted regularization, and different kernels (chi-squared, linear).
Classification accuracy is evaluated using (class-based/mean) F1-score and confusion matrix.
Best performing model parameters are automatically chosen.
You can also display misclassified examples with this repository.

## Data
I used a subset of Caltech image dataset[[1]](#1).


<a id="1">[1]</a> 
Li, F.-F., Andreeto, M., Ranzato, M., & Perona, P. (2022). Caltech 101 (1.0) [Data set]. CaltechDATA. https://doi.org/10.22002/D1.20086
