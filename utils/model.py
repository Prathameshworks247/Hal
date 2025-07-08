# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('all-MiniLM-L6-v2')
# model.save('offline_models/all-MiniLM-L6-v2')

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')  # Download happens here
model.save('./all-MiniLM-L6-v2')  # This creates a complete folder locally
