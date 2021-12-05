import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_tsne(config, embedding, labels):
    figure = plt.figure(figsize=(8, 8), dpi=120)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels.ravel())
    plt.axis("off")
    return figure


def tsne(config, features):
    embedding = TSNE().fit_transform(features)
    return embedding
