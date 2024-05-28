import numpy as np
from tabulate import tabulate

vocab={}
documents=[]
vocab_size=0
document_size=0
dtm = np.zeros((100000,1731))

with open("reviews_limited_vocab.txt", "r") as file:
    for row in file:
      words = row.split()
      for word in words:
        if word not in vocab:
          vocab[word] = vocab_size
          dtm[document_size,vocab_size]+=1
          vocab_size+=1
        else:
          reference = vocab[word]
          dtm[document_size,reference]+=1
      document_size+=1

def apply_lsa(A, k):
    U, S, VT = np.linalg.svd(A, full_matrices=False)
    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    VT_k = VT[:k, :]
    A_k = np.dot(U_k, np.dot(S_k, VT_k))
    return A_k

words_to_compare = ['excellent', 'amazing', 'delicious', 'fantastic', 'gem',
                   'perfectly', 'incredible', 'worst', 'mediocre', 'bland', 'meh', 'awful',
                   'horrible', 'terrible']

k_values = [2, 4, 8]
cosine_similarities = {}

from scipy.spatial.distance import cosine

for k in k_values:
    A_k = apply_lsa(dtm, k)
    cosine_similarities[k] = np.zeros((len(words_to_compare), len(words_to_compare)))

    for i, word1 in enumerate(words_to_compare):
        for j, word2 in enumerate(words_to_compare):
            vector1 = A_k[:, vocab[word1]]
            vector2 = A_k[:, vocab[word2]]
            cosine_similarities[k][i, j] = 1 - cosine(vector1, vector2)



print("Cosine Similarities for k=2")
print(tabulate(cosine_similarities[2],headers=words_to_compare,tablefmt="fancy_grid",showindex=words_to_compare))
print("Cosine Similarities for k=4")
print(tabulate(cosine_similarities[4],headers=words_to_compare,tablefmt="fancy_grid",showindex=words_to_compare))
print("Cosine Similarities for k=8")
print(tabulate(cosine_similarities[8],headers=words_to_compare,tablefmt="fancy_grid",showindex=words_to_compare))

def apply_lsa_better(A, k):
    U, S, VT = np.linalg.svd(A, full_matrices=False)
    U_k = U[:, 1:k+1]
    S_k = np.diag(S[1:k+1])
    VT_k = VT[1:k+1, :]
    A_k = np.dot(U_k, np.dot(S_k, VT_k))
    return A_k

for k in k_values:
    A_k = apply_lsa_better(dtm, k)
    cosine_similarities[k] = np.zeros((len(words_to_compare), len(words_to_compare)))

    for i, word1 in enumerate(words_to_compare):
        for j, word2 in enumerate(words_to_compare):
            vector1 = A_k[:, vocab[word1]]
            vector2 = A_k[:, vocab[word2]]
            cosine_similarities[k][i, j] = 1 - cosine(vector1, vector2)

print("Cosine Similarities for k=2")
print(tabulate(cosine_similarities[2],headers=words_to_compare,tablefmt="fancy_grid",showindex=words_to_compare))
print("Cosine Similarities for k=4")
print(tabulate(cosine_similarities[4],headers=words_to_compare,tablefmt="fancy_grid",showindex=words_to_compare))
print("Cosine Similarities for k=8")
print(tabulate(cosine_similarities[8],headers=words_to_compare,tablefmt="fancy_grid",showindex=words_to_compare))