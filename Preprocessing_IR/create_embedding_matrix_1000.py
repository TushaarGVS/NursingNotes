import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec,TaggedDocument

df = pd.read_csv("Preprocessed_MIMIC.csv")

class_labels = df['class_labels']

# print(df.columns)

mat = df['PREPROCESSED_TEXT']
mat = np.asarray(mat)

corpus = mat

# Applying Doc2Vec embedding model to the corpus

# Converting each document as list of words 
corpus_as_list = []

for doc in corpus:
	lis = doc.split()
	corpus_as_list.append(lis)

documents = [TaggedDocument(doc,[i]) for i,doc in enumerate(corpus_as_list)]
model = Doc2Vec(documents,vector_size=1000,workers=4,min_count=1,epochs=50)

embedding_matrix = []

for i in range(len(corpus_as_list)):
	prediction = model.docvecs[i]
	embedding_matrix.append(prediction)

embedding_matrix = np.asarray(embedding_matrix)

my_df = pd.DataFrame(embedding_matrix)

print(my_df.shape)

my_df['Class_Label'] = class_labels

print(my_df.shape)

my_df.to_csv('MIMIC_EMBEDDING_1000.csv',index=False,header=False)

print('Creating CSV Done')
