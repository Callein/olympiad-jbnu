import pickle

with open('../data/embeddings.pkl', 'rb') as f:
    knowledge_db = pickle.load(f)

if __name__ == "__main__":
    print(knowledge_db.head())