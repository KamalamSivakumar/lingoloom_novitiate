# lln - lingo(nlp) loom(models) novitiate(novice)

Welcome to my repository, featuring all the foundational projects I've completed while learning about nlp and llms. 

This readme file is more like a blog, where I mention each of the project's link along with what I've learnt from working on the same. 

I've honed my skills by diving into blogs and documentation, and I'll be including those references as well.  
Feel free to reach out if you spot any mistakes or have suggestions, as I'm still learning and appreciate any feedback.

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

References:

https://github.com/cjhutto/vaderSentiment

https://towardsdatascience.com/a-guide-to-text-classification-and-sentiment-analysis-2ab021796317

----------------------------------------------------------------------------------------------------------------------------------------------------------

### [Aspect Based Analysis](https://github.com/KamalamSivakumar/lingoloom_novitiae/blob/main/ASBA%20using%20spacy%20and%20VADER.ipynb)

Why ASBA?

Often, a business requirement is to understand and analyze customer reviews. When we can comprehend the reviews the “product”, “component” or “aspect” wise, we can make more use of the reviews.

ASBA has 3 steps (in an overview):
1. Extract aspect candidates.
2. Classify the aspect candidates as aspects or non-aspects.
3. Associate a polarity to each extracted aspect.

Performing ASBA with spaCy and VADER:  
With spaCy’s inbuilt language models, it is easier to extract the token’s Part-Of-Speech (POS) tag.

An assumption made: Usually the features of products and services are mostly nouns and compound nouns.

Accordingly, their BOI tags are considered for identifying the “target”/”aspect” and its adjective would be the corresponding “description”/”opinion”.

Once identified, the polarity scores can either be assigned by employing TextBlob/VADER libraries. In case of VADER, the threshold for classifying is considered as follows: pos-> > 0.5 & neg-> <=0.5.

Spaces for improvement: As of now, in regard with the aspect extraction, only the final occurrence of adj is considered. With regards to assigning polarity scores, non-lexical models can be experimented with. 

References:

https://medium.com/nlplanet/quick-intro-to-aspect-based-sentiment-analysis-c8888a09eda7

----------------------------------------------------------------------------------------------------------------------------------------------------------

### [Named Entity Recognition](https://github.com/KamalamSivakumar/lingoloom_novitiae/blob/main/Named%20Entity%20Recognition.ipynb)

Goal of NER: Identifying all textual mentions of the named entities. Involves two aspects namely:
1. Identifying the boundaries of the named entity.
2. Identifying the type of the entity.
     
Chunking/Shallow Parsing/Light Parsing: Inside, Others and Before (IOB Components). 

Used spaCy’s displacy for viewing the dependencies/hierarchies between the chunks. Could be done through nltk as well but couldn't install ghostscripts/standford nlp parser for viewing the hierarchy in the chunks. (admin rights required to install the same)

Tagging: Part of Speech Tagging, that could be done through in-built methods using nltk/spacy or by configuring custom POS tagger using existing methods, for example by defining a regex tagger or by defining an NGramTagger.

Used ClassifierBasedPOSTagger from nltk to build a POS Tagger that learns through a classification model (NaiveBayesClassifier)

Used spaCy’s Named Entity Recognition package and viewed the results. Used spaCy’s displacy to render the results better. 

Learnt to build a NER model by using Conditonal Random Fields (CRF). CRF is used for prediction tasks where contextual information in our case, state of the neighboring words affects the current prediction. Trained using Gradient Descent approach, (MLE) where the parameters are optimized to maximise the probability of the correct output sequence.

Nuances learnt:
1. backoff concept in taggers: backoff is nothing but when any of the words in the input sequence, don't have a corresponding tag, it's assigned None, when with backoff mentioned, the backup tagger method mentioned is used. This improves the performance of the corresponding taggers significantly. 
2. In order to use CRF for NER, the features must be defined. The following features were considered: the word, the last 3 characters, the last 2 characters, its POS tag, if it’s a digit. Additionally, [BOS] (beginning of a sentence) and [EOS] (end of a sentence) were added as well.
3. The crf_suite from scikit-learn is easier to implement with, the desired MLE algorithm can be mentioned, and if we need to consider all possible options in the CRF. The model performed quite well, further improvements could be made by extending the feature engineering and fine tuning the hyperparameters.

References:

Notebooks on NER from [Dipanjan Sarkar](https://github.com/dipanjanS)

-----------------------------------------------------------------------------------------------------------------------------------------------------

### [Custom NER](https://github.com/KamalamSivakumar/lingoloom_novitiae/blob/main/Custom%20NER.ipynb)

Key points:
1. Config File: Aids in setting the parameters and values for the respective spacy pipeline. Base config file (template) can be downloaded from the documentation: https://spacy.io/usage/training based on the task at hand. Auto-filling of other details can be done for ease of use. We can set the parameters of the pipeline in a bespoke manner as well. 
2. Annotating the data for validation and saved as a list of [text, entites_dict]. Various custom text annotators are available over the internet, I have used: https://tecoholic.github.io/ner-annotator/
3. Using DocBin object from spacy to load our newly annotated data. The DocBin class is used to efficiently serialize the information from a collection of Doc (spacy) objects.
4. Then, the custom spaCy NER model can be trained, and used to give annotations.
   
Commands list that enabled custom NER using spaCy:
1. Auto filling the config file: !python -m spacy init fill-config ner_config.cfg (template) config.cfg (Auto filling based on the template)
2. To train: !python -m spacy train "C:/Users/kamalam.s/Desktop/kamalam's/nlp dev/config.cfg" (path to config)
…  --output "C:/Users/kamalam.s/Desktop/kamalam's/nlp dev/trained_models/output" (path to output folder for the model to save)
…  --paths.train "C:/Users/kamalam.s/Desktop/kamalam's/nlp dev/trained_models/train_data.spacy" (path to train data)
… --paths.dev "C:/Users/kamalam.s/Desktop/kamalam's/nlp dev/trained_models/test_data.spacy" (path to test data)

The commands can be customised to include parameters to our requirement. References: spaCy Documentation: https://spacy.io/usage/training

The model’s performance can be improved by training on appropriate custom annotated data.

Creating a custom annotated data on a larger scale is a greater task. 

Understood the general flow of how Custom NER works. 

References:

https://medium.com/@mjghadge9007/building-your-own-custom-named-entity-recognition-ner-model-with-spacy-v3-a-step-by-step-guide-15c7dcb1c416

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

### [BERT for question answering](https://github.com/KamalamSivakumar/lingoloom_novitiae/blob/main/BERT%20for%20Question%20Answering.ipynb)

Basics of BERT: (based on its documentation)
1. Bidirectional Encoder Representations from Transformers. The pre-trained BERT model can be fine-tuned with just an additional output layer to create models for a wide range of tasks, such as question answering, without substantial task-specific architecture modifications.
2. Considers context from both right and left.
3. BERT is suitable for Text Classification, Question Answering, Summarization, but not for Text Generation. This is because it was modelled with Masked Language Modelling and Next Sentence
Prediction objectives.
4. Corrupts the inputs by using random masking (making it not optimal for text generation).

Before applying BERT for a task at hand, the data must be manipulated to help BERT in achieving the objectives. The manipulation depends on the task at hand. 

Text Classification:
1. Tokenize the sentences.
2. Add the [CLS] and [SEP] tags for each sentence. 
3. Pad and truncate sentences to the maximum length. (padded on the right by default)
4. Construct attention masks. 
5. Trained and Classified using “BertForSequenceClassification”.
              
Question Answering:
1. Tokenize the question and context.
2. Chunk or split the context sentences to overlap.
3. Add the [CLS] and [SEP] tags for each sentence.
4. Truncate only the context based on the maximum length context.
5. Set doc_stride, sets the pace for considering context windows. 
6. Set “True” for overlapping token chunks, ensures our answer doesn't get missed.
7. Set “True” for offsets, returns the mapping between tokens and position in the original context.
8. Sequence Ids are used to identify if question (0) or context (1).
9. Find the answer token_start and token_end position.
10. Using datasets.map(prepare_train_features, batched=True, remove_columns=datasets["train"].column_names) to get our data is manipulated to apply. (Basically, tokenizing the dataset with the above-mentioned components in place.)
11. Trained and Classified using “AutoModelForQuestionAnswering”/ “BertForQuestionAnswering”

Nuances learnt:

Training procedure for Question-Answering:
1. The training arguments are defined using “TrainingArguments” from the transformers library. Defines the following:

    >>#folder path to save the model checkpoints.
    >>
    >>#evaluation strategy, done at the end of each epoch.
    >>
    >>#defining the learning rate
    >>
    >>#defining the train size
    >>
    >>#defining the batch size
    >>
    >>#no.of epochs
2. The “Trainer” method considers the training arguments and the manipulated dataset for training.

p.s I keyboard interrupted the runtime, as I didn't have the resources and time to run the training. 

References: [BERT documentation](https://huggingface.co/docs/transformers/en/model_doc/bert#overview)

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

### [Question-Answering and Text-Generation](https://github.com/KamalamSivakumar/lingoloom_novitiate/blob/main/text-generation%20%26%20question-answering.ipynb)

Question Answering by an llm based on a context. 

Context documents stored as vector databases. 

Employing Q&A through text-generation - a work around basically as HuggingFace Pipeline Only supports text-generation, text2text-generation, summarization and translation for now.

------------------------------------------------------------------------------------------------------------------------------------------------------------------------
