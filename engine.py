import os
import time
import re
import sys

from logic import df
#
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from rank_bm25 import BM25Okapi

stop_words = set(stopwords.words('english'))

# iterate through all file
corpus = [(body+' '+title).lower() for title, body in zip(df['title'], df['keywords'])]
print(len(corpus))
# for i in range(10):
#     print(corpus[i])
#     print('\n')

tokenized_corpus = [doc.split(" ") for doc in corpus]

bm25 = BM25Okapi(tokenized_corpus)

query = "sieve of eratosthenes"
tokenized_query = query.split(" ")

doc_scores = bm25.get_scores(tokenized_query)
results = bm25.get_top_n(tokenized_query, corpus)
for result in results:
    print(result+'\n\n')
print(sorted(doc_scores)[::-1][:5])

#
# file = open('spoj/titleeees.txt', 'a')
# all_ques = [x for x,y in [z.split('.') for z in os.listdir('spoj/questions')]]
# all_urls = [x[30::] for x in urls][:-1:]
# for que in all_ques:
#     if que not in all_urls:
#         print(que)

# for i in range(0,len(all_ques)):
#     if all_urls[i]!=all_ques[i]:
#         print(all_urls[i], all_ques[i])

# for url in all_urls:
#     if url not in all_ques:
#         print(url)
# print(len(all_ques),len(all_urls))
