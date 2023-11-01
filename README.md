# Sentiment-Analysis
Sentiment analysis comparisons between Naive Bayes, Perceptron, and BERT. 

This project performs sentiment analysis by classifying an entire movie review as positive or negative. We used the imdb dataset as used in the paper below.
2002. Thumbs up? Sentiment Classi cation using Machine Learning Techniques. Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 79-86.

1. Naive Bayes: Naive Bayes is based on Bayes' theorem and assumes that the presence of each word in a document is conditionally independent of the presence of any other word in that document. Naive Bayes assumes conditional independence of features (fi) given the document's class (c), and the training involves relative-frequency estimation of P(c) and P(fi|c) using add-one smoothing. 

2. Perceptron: The Perceptron classification algorithm for sentiment analysis is a simple binary classifier that assigns sentiment labels, such as positive or negative, to text data. It represents text numerically, initializes weights, and iteratively updates them during training to find a linear boundary that separates the sentiment classes. If a document is misclassified, the algorithm adjusts the weights to correct the error. Once trained, it can classify new text data based on the learned linear separation. We train the perceptron for 4000 iterations. 

4. BERT. We fine-tune BERT (Bidirectional Encoder Representations from Transformers) on the imdb dataset to perform sentiment analysis of movie review classification. We train the model for 10 epochs on the training dataset.


 | Model | Average 10 fold Accuracy |
|----------|----------|
| Naive Bayes | 81.6 |
| Perceptron | 82.2 |
| BERT | 83.0 | 

