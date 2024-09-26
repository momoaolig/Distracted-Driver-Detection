# Deep learning on Image Classification: Hybrid Model for Distracted Driver Detection
![Poster](Poster.pdf)

Mo Zhou, Joanna Wang  
We leverage pre-trained CNN models as the baseline and explore whether alternative hybrid models of pre-trained CNN  and transformers can have superior results. The findings indicate that pre-trained CNNs perform the best, followed by a hybrid model with ResNet and Vit. This showcases the effectiveness of CNN in capturing local features when classifying images. Another finding is that ResNet50 and ResNet50+ViT models exhibit confusion between the label safe driving and the label talking to passengers. Future research could modify the transition layer of the hybrid model and involve augmenting data samples for classes 0 and 9 to enhance accuracy
# Dataset
The dataset we used is State Farm Distracted Driver Detection from Kaggle. It includes driver images taken in a car with a driver doing something in the car (texting, eating, talking on the phone, makeup, reaching behind, etc). The goal is to predict the likelihood of what the driver is doing in each picture.
# Model
In this experiment, we test the performance of different models on this image classification task, including CNN-based models and transformers. We also combine both CNN and transformers together to see if it can result in a better performance. The reason for the combination is because CNN models capture local patterns of an image better, and are efficient at processing raw pixel data and extracting low-level features like edges, and textures, while transformers utilize self-attention mechanisms to capture global dependencies within the image. The CNN-based models we used are ResNet50 and VGG16. The transformer architectures we used are ViT and Swin.

