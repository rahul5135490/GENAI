from chromadb.utils import embedding_functions 

default_ef = embedding_functions.DefaultEmbeddingFunction()

name = "Rahul Srivastava"

emb = default_ef(name)

print(emb)