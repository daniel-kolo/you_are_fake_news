# Team You are fake news!'s Homework Project for the Deep Learning in Practice with Python and LUA course

## Team Members:
- Kolozsvári Dániel
- Domonkos Bálint

## The Project:
Building an agent that can classify an english language news article as real news, or fake news.

## Data used for the project:
### Documentation for the dataset:
https://github.com/several27/FakeNewsCorpus

### Data can be downloaded at:
https://storage.googleapis.com/researchably-fake-news-recognition/news_cleaned_2018_02_13.csv.zip

## How to prepare data
The notebooks, and other scripts are in the scr directory. For more information about the code, please see the comments. 

| File name | Usage |
| ------------- | ------------- |
| Data analysis.ipynb  | Analyzing text for stopwords and input length.  |
| Download_data.ipynb  | Extracting a smaller database from the original database.  |
| Preprocess_data.ipynb  | Processing data to be used by model.  |
| text_preprocess.py  | Functions for filtering stopwords, punctuation, and lemmatizing words.  |
| training_preprocess.py  | Functions for spilitting and vectorizing the data.  |
| BiLSTM network.ipynb  | Initial training and methods of the network  |
| BiLSTM_ELMo.ipynb | Bilateral LSTM with Elmo model training  |
| data_slicing.ipynb | Shareding the data to comply with github size restrictions  |
| final_network.h5 | The final billateral Neural Network |
| hyper_param_optimalization_final_training.ipynb | Hyperparameter optimalization and training of the final Bilateral LSTM network |
| state_of_the_art/Bert.ipynb | Bert model usage |
| state_of_the_art/GPT-2_generator.ipynb | GPT2 network usage + grover |
| state_of_the_art/GPT-2_detector.ipynb | GPT2 network usage for detection + Roberta |
| state_of_the_art/GPT-2_generator.ipynb | GPT2 network usage + grover |


