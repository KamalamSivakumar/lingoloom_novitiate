# lln - lingo(nlp) loom(models) novitiae(beginning)

Welcome to my repository, featuring all the foundational projects I've completed while learning about NLP and LLMs. 

This readme file is more like a blog, where I mention each of the project links along with what I've learnt from working on the same. 

I've honed my skills by diving into blogs and documentation, and I'll be including those references as well.

Hope this is useful and enjoyable. 

-------------------------------------------------------------------------------------------------------------------------------

### [NLP for analyzing disaster tweets](https://github.com/KamalamSivakumar/lingoloom_novitiae/blob/main/NLP%20with%20Disaster%20tweets.ipynb)

Tried out preprocessing with both NLTK and spaCy libraries.
1.	Preprocessing text to remove unnecessary characters.
2.	Preprocessing text to tokenize, remove stop words and stem/lemmatize.

Used Multinomial NB and a pretrained BERT Model for Text Classification.

For Multinomial NB, used the preprocessed input as TF-IDF vectors and applied Multinomial NB to get the predicitons.

For BERT Model, used the transformers library from huggingface, used the preprocessed input as encoded tensors.

Configured training and classification used “BertForSequenceClassification” which has a linear classification layer, classifying as disaster or not.

Using PyTorch libraries, ran training and validation, to get the predicitons.

Nuances learnt: 
1.	How the batch training and validation sets are configured,
2.	How the hyperparameters (learning rate) can be fine-tuned or scheduled.

[Kaggle reference link](https://www.kaggle.com/competitions/nlp-getting-started)

----------------------------------------------------------------------------------------------------------------------------------------

### [Sentiment Analysis](https://github.com/KamalamSivakumar/lingoloom_novitiae/blob/main/Sentiment%20Analysis.ipynb)

Learnt about the basics of Sentiment Analysis:
1. Components of a review (Opinion, Subject, Entity), sentiments can be classified based on document (entire text), sentence (given sentence) or aspect (which part of sentence, eg a phone review, “the display is good, but the battery strength is poor”)
2. Valence of a sentence (polarity and subjectivity)
3. Rule based and ML based classification
              
Worked on “Sentiment Analysis, classify a review as positive or negative” problem using Rule based and ML based approaches.

Rule based approach: VADER

ML based approach: NB Classifier, Logistic Regression and KNN

Nuances learnt:
1. Text Classification vs Sentiment Analysis:
               While Sentiment Analysis is a Text Classification problem, it aims at understanding the tone of the document being analyzed. With the Disaster Classification problem, I did not aim at finding how the tone of the tweet was, while the Sentiment Analysis problem revolves around finding the tone of the said document.
2. It’s very important to pre-process the text effectively, before applying any type of model to get effective results.
3. As VADER is mainly trained on social media (tweets/comments etc), it did not classify movie_reviews effectively, tried to compute the VADER scores for each sentence in a movie review, with a compound measure that indicates the normalized or aggregated polarity score for that sentence. The result did come out better. 
4. Extracted features that can be used to train the ML models, positive scores and compound scores, the feature extraction can also be extended to most frequent words used in a positive/negative review.

----------------------------------------------------------------------------------------------------------------------------------------------------------

