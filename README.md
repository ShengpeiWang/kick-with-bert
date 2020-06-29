# kick-with-bert
Applying Bert (RoBERTa) for Better Text Classification 

## Motivation
In another project, I created an app to predict Kickstarter project success (amount raised) using NLP, but I was not satisfied with the model accuracy. After exposing myself to more modern NLP techniques, i.e. deep recurrent neural network and transformer models (BERT), I decided to use these more complex techniques to improve model performance of my old project.

Specifically, I used the pretrained RoBERTa model to build a classifier and fine tuned the model with my own data on Kickstarter proposals. To build the model, I downloaded the pretrained weights of the RoBERTa model, used it to initiate the first 13 layers of a neural network, and added a linear layer on top to perform the classification task. I then progressively trained last layers of this model using Kickstarter proposal text, predicting project success (amount raised binned into 4 buckets).     

Among the many pretrained BERT models available, RoBERTa tends to perform better than other models on most language tasks. Even though RoBERTa is not the most memory efficient or fastest to train, I am not constrained by memory use or training speed, so I decided to chose RoBERTa over other models. 

## Notes of interests
I have heard that the hidden outputs of BERT models are not better word embeddings than classic methods such as Word2vec. This means that training only the classification layer but not the encoder layers would not give great model performance. Since I trained the different layers progressively from the last layer, I was able to observe this effect during training. When I only trained the classification head (the last layer), the model performance improved only minimally with more training. But when I unfroze the last two layers of the encoder layers, more training led to about 10% imcrease in accuracy, a marked improvement in model performance.

## Built with
-	Python
-	Hugging Face
-	PyTorch
-	fast.ai

## Features
-	Built a classification model with pretrained RoBERTa weights
-	Fine tuned the model using triangular learning rates, using fast.ai’s functionality
-	Applied pretrained tokenizer to integrate with fast.ai’s utilities

## Credits
I consulted heavily to other projects that have done similar analysis. Especially [Melissa Rajaram’s post](https://www.kaggle.com/melissarajaram/roberta-fastai-huggingface-transformers#Training-the-Model) with code that applies to many other BERT models as well. I also learned a lot about using PyTorch’s utilities for my purpose from code examples in [Venelin Valkov’s post](https://www.curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/). Lots of thanks to Hugging Face’s work on making so many BERT models so easy to apply to different situations. And to fast.ai for applying the latest deep learning research that makes model training both easy and fast. 
