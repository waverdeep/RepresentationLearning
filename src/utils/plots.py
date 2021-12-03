import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_tsne(config, embedding, labels, epochp):
    # fp = os.path.join(args.out_dir, "tsne", "{}-{}.png".format(epoch, step))
    # if not os.path.exists(os.path.dirname(fp)):
    #     os.makedirs(os.path.dirname(fp))

    figure = plt.figure(figsize=(8, 8), dpi=120)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels.ravel())
    plt.axis("off")
    # plt.savefig(fp, bbox_inches="tight")
    return figure


def tsne(config, features):
    embedding = TSNE().fit_transform(features)
    return embedding
