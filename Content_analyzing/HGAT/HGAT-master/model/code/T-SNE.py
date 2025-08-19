from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
 
    #embeddings = get_embeddings(model, G)
filePath = r"embeddings/test_v3.emb"
with open(filePath,'r',encoding='utf-8') as file:
    embedding_list = file.readlines()

model = TSNE(n_components=2)
compress_embedding = model.fit_transform(embedding_list)
keys = list(embeddings.keys()) #如何给列表加一个序号作为key
 
plt.scatter(compress_embedding[:, 0], compress_embedding[:, 1], s=10)
for x, y, key in zip(compress_embedding[:, 0], compress_embedding[:, 1], keys):
    plt.text(x, y, key, ha='left', rotation=0, c='black', fontsize=8)
plt.title("T-SNE")
plt.show()