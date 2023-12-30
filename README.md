# Project_Hate_Offensive_Speech_detection

## About this dataset
This dataset, named hate_speech_offensive, is a meticulously curated collection of annotated tweets with the specific purpose of detecting hate speech and offensive language. The dataset primarily consists of English tweets and is designed to train machine learning models or algorithms in the task of hate speech detection. It should be noted that the dataset has not been divided into multiple subsets, and only the train split is currently available for use.

The dataset includes several columns that provide valuable information for understanding each tweet's classification. The column count represents the total number of annotations provided for each tweet, whereas hate_speech_count signifies how many annotations classified a particular tweet as hate speech. On the other hand, offensive_language_count indicates the number of annotations categorizing a tweet as containing offensive language. Additionally, neither_count denotes how many annotations identified a tweet as neither hate speech nor offensive language.

For researchers and developers aiming to create effective models or algorithms capable of detecting hate speech and offensive language on Twitter, this comprehensive dataset offers a rich resource for training and evaluation purposes.

Data Source; https://huggingface.co/datasets/hate_speech_offensive

## About Training

The process started with understanding the features and distribution with the target using data exploration techniques.
Feature Engineering and model selection was done in 4 experiments where each experiment has a different combination of text preprocessing and resampling.
The model selection phase utilized text processing techniques like: 
 - Lowercasing
 - Replacing Puctuations
 - Replacing numbers and hastags
 - Tokenizing
 - Lemmatization
 - Stemming
 - Vectorization

and models like:
 - MultinomialNB
 - RandomForest
 - KNN
 - Xgboost

After initial training, the best model -- MultinomialNB was selected and placed in a pipeline with CountVectorizer

Hyperparameter tuning was done using Grid search to select the best tuning parameters.

The best experiment's feature preprocessing combination was selected to preprocess the train set and the model pipeline was fit to the preprocessed data.

