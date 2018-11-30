import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("Preprocessed_MIMIC.csv")

class_labels = df['class_labels']

# print(df.columns)

mat = df['PREPROCESSED_TEXT']
mat = np.asarray(mat)

corpus = mat

# Applying TfIDF to the corpus

vectorizer = TfidfVectorizer()

x = vectorizer.fit_transform(corpus)

daf = pd.DataFrame(x.todense(),columns=vectorizer.get_feature_names(),dtype=float)

print(daf.shape)

daf['class_labels'] = class_labels

print(daf.shape)


daf.to_csv('MIMIC_TFIDF.csv',index=False,header=False)

print('CSV Creation Done')