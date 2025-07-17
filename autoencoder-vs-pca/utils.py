import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_reconstructions(model, images, n_images=5):
    reconstructions = np.clip(model.predict(images[:n_images]), 0, 1)
    fig = plt.figure(figsize=(n_images * 1.5, 3))
    for i in range(n_images):
        plt.subplot(2, n_images, 1 + i)
        plt.imshow(images[i].reshape(28, 28), cmap="binary")
        plt.axis("off")
        if i == 0:
            plt.title("Original", fontsize=14)
        plt.subplot(2, n_images, 1 + n_images + i)
        plt.imshow(reconstructions[i].reshape(28, 28), cmap="binary")
        plt.axis("off")
        if i == 0:
            plt.title("Reconstruction", fontsize=14)
    plt.tight_layout()
    plt.show()


def visualize_compressed(x_compressed, y, images, result='PCA_Results', class_names=None):
    plt.figure(figsize=(8, 6))
    cmap = plt.cm.tab10
    Z = (x_compressed - x_compressed.min()) / (x_compressed.max() - x_compressed.min())
    scatter = plt.scatter(Z[:, 0], Z[:, 1], c=y, s=10, cmap=cmap)

    image_positions = np.array([[1., 1.]])
    for index, position in enumerate(Z):
        dist = ((position - image_positions) ** 2).sum(axis=1)
        if dist.min() > 0.02:
            image_positions = np.r_[image_positions, [position]]
            imagebox = mpl.offsetbox.AnnotationBbox(
                mpl.offsetbox.OffsetImage(images[index].reshape(28, 28), cmap="binary", zoom=0.6),
                position,
                bboxprops={"edgecolor": cmap(y[index]), "lw": 1}
            )
            plt.gca().add_artist(imagebox)

    if class_names:
        handles = [plt.Line2D([0], [0], marker='o', color='w', label=class_names[i],
                              markerfacecolor=cmap(i), markersize=6) for i in np.unique(y)]
        plt.legend(handles=handles, loc='best', fontsize=10)

    plt.axis("off")
    plt.title(f"{result}: 2D Latent Space", fontsize=14)
    plt.tight_layout()
    plt.show()
