# deeplearning-bootcamp

Projects related to Complete Tensorflow 2 and Keras Deep Learning Bootcamp. 

### [(Basic) Artificial Neural Networks Project](anns.ipynb)

In this project, we will use a subset of the LendingClub dataset obtained from Kaggle. The goal is to build a model that can predict whether or not a borrower will repay their loan.

During the case study, we will do some exploratory data analysis and data pre-processing to finally create a dense multi-layered ANN. We compile the model using binary_crossentropy as the loss function, and perform a performance evaluation by plotting validation losses against training losses and looking at the classification report (accuracy, recall, f1 score, and support) and the confusion matrix.

<p align="center">
  <img src="https://user-images.githubusercontent.com/12433425/172470890-0d6555c3-7fbd-4bd1-bdbf-4645a8849be0.png">
</p>

### [Convolutional Neural Network Project](cnn.ipynb)

In this project, we will use the Fashion MNIST dataset and build an image classifier with a Convolutional Neural Network. During the case study, we will understand the data, preprocess it, and create an ImageDataGenerator to expand the dataset. Finally, we will create a CNN using convolutional and dense layers, along with pooling and flattening layers. We will also add an early stop to prevent overfitting.

Performance evaluation is done by plotting validation losses against training losses and looking at the classification report (accuracy, recall, f1-score, and support) and the confusion matrix.

<p align="center">
  <img src="https://user-images.githubusercontent.com/12433425/172478879-e8b6a0c4-603f-4242-b9ee-28c433ea4b99.png">
</p>

### [Recurrent Neural Network Project](rnn.ipynb)

For this project, we will use the Frozen Dessert Production dataset and create an RNN model to predict future values. We first do a data scan, then scale the data, and finally create an RNN model. This model will have LSTM and Dense layers, and will use a TimeSeriesGenerator in training. We evaluate the model on the test data, forecasting new predictions and plotting them over the true predictions.

<p align="center">
  <img src="https://user-images.githubusercontent.com/12433425/172479137-6f1c6686-4436-4705-9662-25dea89772df.png">
</p>

### [Natural Language Processing Project](rnn.ipynb)

In this project, we will process the work of Charles Dickens and create a model capable of generating text based on the author's style. To do this, we perform text processing and encoding, and create batches (training sequences) for the model. The model (RNN) will have Embedding, GRU and Dense layers.

<p align="center">
  <img src="https://user-images.githubusercontent.com/12433425/172611990-525a14d6-c108-4269-b570-7af783ea1529.png">
</p>

### [Autoencoder Project](rnn.ipynb)

The aim of the project is to determine whether, given a data set of average eating habits across UK countries, any particular country stands out as different. For this we will use autoencoder to reduce the dimensionality and help us identify the country with particular eating habits. We will explore the data and then create two models (encoder and decoder) using dense layers. We then run the encoder on the scaled data and plot the results.

<p align="center">
  <img src="https://user-images.githubusercontent.com/12433425/172612737-f9169d00-4c92-4842-94fb-14268e9a17fb.png">
</p>

### [Generative Adversarial Networks Project](rnn.ipynb)

In this project, we will use the MNIST dataset to train a model capable of creating new images similar to those in the dataset. We'll create the generator and discriminator (using dense layers), then set up the training batches, and finally train the model. Then, we will visualize an image created by the generator.

In the second part of the project, we will use convolutional layers for our GAN model and repeat the same steps as before, observing the new results.

<p align="center">
  <img src="https://user-images.githubusercontent.com/12433425/172614106-b3e9ccee-d855-4c08-8dd6-d00c69512558.png">
</p>
