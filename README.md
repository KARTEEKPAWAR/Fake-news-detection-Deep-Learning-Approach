# Fake-news-detection-Deep-Learning-Approach
pip install --upgrade tensorflow

!pip install plotly

 NLTK( natural language toolkit), PUNKT(parameters from a corpus in an unsupervised) is an unsupervised trainable model, which means it can be trained on unlabeled data Data that has not been tagged with information identifying its characteristics, properties, or categories is referred to as unlabeled data.

import nltk
nltk.download('punkt')

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import nltk
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Input, LSTM, Conv1D, MaxPool1D, Bidirectional
from tensorflow.keras.models import Model

What is stemming in NLTK?
Stemming is a technique used to reduce an inflected word down to its word stem

df_fake = pd.read_csv("Fake.csv")
df_fake.head()

df_fake.tail()

df_true = pd.read_csv("True.csv")
df_true.head()

df_true.tail()

Performing Exploratory Data Analysis
Note:
- Target Column isfake is created
- isfake is set to 1 for real news and 0 for fake news

# Adding A Target Class Column To Indicate Whether The News Is Real Or Fake
df_true['isfake'] = 1
df_true.head()

df_fake['isfake'] = 0
df_fake.head()

Notes:
- By using Concat() function two dataframe df_true and df_fake convert into df dataframe
- reset_index() method resets the index back to the default 0, 1, 2 etc indexes. By default, this method will keep the "old" idexes in a column named "index". To avoid this, drop parameter is used.

# Concatenating Real And Fake News
df = pd.concat([df_true, df_fake]).reset_index(drop = True)
df

#Checking if dataset is class-imbalanced or not
df['isfake'].value_counts()

 Notes:
- No of samples in 2 classes are 23481 and 21417
- So, dataset is almost balanced
- used drop() function to remove unnecessary column

df.drop(columns = ['date'], inplace = True)

df['original'] = df['title'] + ' ' + df['text']
df.head()

Performing Data Cleaning
Notes:
- stopwords imported from nltk.corpus and extending using some words
- STOPWORDS imported from gensim.parsing.preprocessing is also used to remove stopwords
- NLTK library has 179 words in the stopword collection, whereas Gensim has 337 words
- Any tokens shorter than min_len=2 characters and greater than max_len=15 characters are discarded using simple_preprocess()
- 9276947 words are found by traversing through each word of each sentence
- 108704 unique words are obtained using set keyword
- join() method is used to combine words to sentence

nltk.download("stopwords")

# Obtaining Additional Stopwords From nltk
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

# Removing Stopwords And Remove Words With 2 Or Less Characters
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in stop_words:
            result.append(token)
            
    return result

# Applying The Function To The Dataframe
df['clean'] = df['original'].apply(preprocess)
# Showing Original News
df['original'][0]

# Showing Cleaned Up News After Removing Stopwords
print(df['clean'][0])

df

# Obtaining The Total Words Present In The Dataset
list_of_words = []
for i in df.clean:
    for j in i:
        list_of_words.append(j)
list_of_words

len(list_of_words)

# Obtaining The Total Number Of Unique Words
total_words = len(list(set(list_of_words)))
total_words

# Joining The Words Into A String
df['clean_joined'] = df['clean'].apply(lambda x: " ".join(x))
df

df['clean_joined'][0]

Visualizing Cleaned Up Dataset
💡 Notes:
- Two countplot are plotted, one for Subject wise News Count, another for Real-Fake wise News Count
- The most frequent words are among the given text are visualized using WordCloud

# Plotting The Number Of Samples In 'subject'
plt.figure(figsize = (8, 8))
plt.title("Subject Wise News Count")
sns.countplot(y = "subject", data = df)
plt.show()

Notes:
- politicsNews is in largest amount
- US_News, Middle-east are in shortest amount

plt.figure(figsize = (8, 8))
plt.title("No of Samples based on Real Fake")
sns.countplot(y = "isfake", data = df)
plt.show()

plt.figure(figsize = (20,20)) 
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = stop_words).generate(" ".join(df[df.isfake == 1].clean_joined))
plt.imshow(wc, interpolation = 'bilinear')
plt.show()

# Plotting The Word Cloud For Text That Is Fake
plt.figure(figsize = (20,20)) 
wc = WordCloud(max_words = 2000 , width = 1600 , height = 800 , stopwords = stop_words).generate(" ".join(df[df.isfake == 0].clean_joined))
plt.imshow(wc, interpolation = 'bilinear')
plt.show()

# Determining The Maximum Number Of Words In Any Document Required To Create Word Embeddings 
maxlen = -1
for doc in df.clean_joined:
    tokens = nltk.word_tokenize(doc)
    if(maxlen<len(tokens)):
        maxlen = len(tokens)
print("The maximum number of words in any document is =", maxlen)

# Visualizing The Distribution Of Number Of Words In A Text
import plotly.express as px
fig = px.histogram(x = [len(nltk.word_tokenize(x)) for x in df.clean_joined], nbins = 100)
fig.show()

Preparing The Data By Performing Tokenization And Padding
 Notes:
- A text string is tokenized into individual words using word_tokenize
- maxlen is set to the largest no of tokens among texts
- Text Data is splitted into train-test as 80:20 ratio
- A set of text data is converted into a sequence of integers using texts_to_sequences
- All sequences have been converted to the same length using pad_sequences

# Splitting Data Into Test And Train 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df.clean_joined, df.isfake, test_size = 0.2)

from nltk import word_tokenize

# Creating A Tokenizer To Tokenize The Words And Create Sequences Of Tokenized Words
tokenizer = Tokenizer(num_words = total_words)
tokenizer.fit_on_texts(x_train)
train_sequences = tokenizer.texts_to_sequences(x_train)
test_sequences = tokenizer.texts_to_sequences(x_test)

print("The encoding for document\n",df.clean_joined[0],"\n is : \n\n",train_sequences[0])

# Adding Padding
padded_train = pad_sequences(train_sequences,maxlen = 40, padding = 'post', truncating = 'post')
padded_test = pad_sequences(test_sequences,maxlen = 40, truncating = 'post') 
for i,doc in enumerate(padded_train[:2]):
     print("The padded encoding for document",i+1," is : ",doc)

Building And Training The Model
Notes:
- Sequential model is built
- Integer inputs, which represent the index of words, are converted to dense vector of fixed size using Embedding
- Input Sequences are processed in both directions using Bidirectional
- A Bidirectional LSTM layer is created with 128 units

# Sequential Model
model = Sequential()

# Embeddidng layer
model.add(Embedding(total_words, output_dim = 128))


# Bi-Directional RNN and LSTM
model.add(Bidirectional(LSTM(128)))

# Dense layers
model.add(Dense(128, activation = 'relu'))
model.add(Dense(1,activation= 'sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.summary()

total_words

y_train = np.asarray(y_train)
# Training the model
model.fit(padded_train, y_train, batch_size = 64, validation_split = 0.1, epochs = 5)

Assessing Trained Model Performance
 Notes: Model Performance is assessed using-
- accuracy_score
- confusion_matrix
- classification_report

# Making prediction
pred = model.predict(padded_test)

# If The Predicted Value Is >0.95 (i.e., More Than 95%), It Is Real Else It Is Fake
prediction = []
for i in range(len(pred)):
    if pred[i].item() > 0.95:
        prediction.append(1)
    else:
        prediction.append(0)

# Getting The Accuracy
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(list(y_test), prediction)

print("Model Accuracy : ", accuracy)

# Getting The Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(list(y_test), prediction)
plt.figure(figsize = (4, 3))
sns.heatmap(cm, annot = True)
plt.show()

Notes:
- False Positive and False Negative is very less comapare to True Positive and True Negative So, confusion_matix indicates model is performing good

# Getting The classification_report
from sklearn.metrics import classification_report
print(classification_report(list(y_test), prediction))

Notes:
- precision, recall, f1-score indicates model's performance is pretty well
