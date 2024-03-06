## Leaf Classification with CNN and Random Forest Ensemble.

The inspiration for the project idea and dataset came from [PUT Vision Lab](https://github.com/PUTvision/WDPOProject/tree/v2023?fbclid=IwAR0sf5s2HThpwEizT8nSRvGK55OU4nRmtPnd7vs46LFmc6yeXXMa-wp2MCc).

## Workflow

![](https://github.com/01Cramer/classification-cnn-and-random-forest-ensemble/blob/main/image.png)

The image above illustrates how we performed leaf detection on the image. The extracted leaves were then sorted and divided into 5 different categories based on the species. In the next step, the extracted leaves were preprocessed to facilitate their effective handover to the neural network. The model was built using convolutional neural networks tasked with extracting the most relevant features from the image, which were then passed to a random forest classifier. This classifier made decisions and predicted the class of the given leaf.
