import pandas as pd
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize
import numpy as np
import re

stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

df = pd.read_csv("Merged_MIMIC.csv")

# print(df.columns)

mat = df['TEXT']
mat = np.asarray(mat)

corpus = mat

# Start of Preprocessing
for i,row in enumerate(corpus):
	# Remove everything except spaces, numbers and letters
	corpus[i] = re.sub('[^a-zA-Z0-9 ]','',corpus[i])
	# Remove spaces at beginning if any
	corpus[i] = re.sub('^ ','',corpus[i])
	# Remove spaces at end if any
	corpus[i] = re.sub(' $','',corpus[i])
	# Combine 2 or more successive spaces into one
	corpus[i] = re.sub('  ',' ',corpus[i])
	# Convert all words to lowercase (for consistency)
	corpus[i] = corpus[i].lower()
	# Tokenize
	words = word_tokenize(corpus[i])
	# Lemmatize
	lemmatized_words = []
	for word in words:
		lemmatized_words.append(lemmatizer.lemmatize(word))
	# Stopword Removal
	words_without_stopwords = []
	for word in lemmatized_words:
		if word not in stop_words:
			words_without_stopwords.append(word)
	# Recreate the entry after preprocessing
	corpus[i] = ' '.join(words_without_stopwords)

print(df.shape)

df['PREPROCESSED_TEXT'] = corpus

print(df.shape)

df.to_csv('Preprocessed_MIMIC.csv',index=False,header=True)

print('CSV Creation Done')