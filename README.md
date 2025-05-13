TruthSentry: Fake News Detection

TruthSentry is a machine learning project designed to detect fake news articles using natural language processing (NLP) and classification algorithms. The project processes news article titles and author names, applies text preprocessing techniques (stemming, stopword removal, TF-IDF vectorization), and trains models like Logistic Regression, Random Forest, and Gradient Boosting to classify news as real or fake.

Dataset

The project uses a dataset (fake_news_dataset.csv) containing 20,000 news articles with the following columns:





title: Article title



text: Article content



date: Publication date



source: News source



author: Author name



category: News category



label: Real (0) or Fake (1)

Features





Text Preprocessing: Removes non-alphabetic characters, applies stemming, and removes stopwords.



Feature Extraction: Uses TF-IDF vectorization to convert text data into numerical features.



Models: Implements Logistic Regression, Random Forest, and Gradient Boosting classifiers.



Evaluation: Provides accuracy scores and classification reports for model performance.



Predictive System: Classifies new articles as real or fake based on trained models.
Usage





Open the Fake_news_detection_new.ipynb notebook in Jupyter.



Execute the cells sequentially to:





Load and preprocess the dataset.



Train and evaluate the models (Logistic Regression, Random Forest, Gradient Boosting).



Test the predictive system on sample data.



To classify a new article, use the trained model as shown in the notebook's predictive system section.

Model Performance





Logistic Regression:





Training Accuracy: ~64.91%



Test Accuracy: ~49.45%



Random Forest (on subset):





Training Accuracy: ~84.19%



Test Accuracy: ~51.12%



Gradient Boosting (on subset):





Training Accuracy: ~70.31%



Test Accuracy: ~48.50%

Note: The models show signs of overfitting, particularly Random Forest. Further tuning (e.g., increasing max_features, adjusting hyperparameters, or using a larger dataset) could improve performance.
Future Improvements





Enhance model performance by tuning hyperparameters or using advanced models like BERT.



Incorporate additional features (e.g., article text, source credibility).



Address dataset imbalance and improve preprocessing for better generalization.



Deploy the model as a web application for real-time fake news detection.

Contributing

Contributions are welcome! Please fork the repository, create a new branch, and submit a pull request with your changes.
