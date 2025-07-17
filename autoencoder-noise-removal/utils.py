import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def plot_noise_images(X_clean, X_noisy, title):
    plt.figure(figsize=(14, 3))
    plt.suptitle(title, fontsize=16, y=1.05)
    num_images = min(10, len(X_clean))
    for i in range(num_images):
        plt.subplot(2, 10, i + 1)
        plt.imshow(X_clean[i], cmap='binary')
        plt.title('Original', fontsize=8)
        plt.axis('off')

        plt.subplot(2, 10, i + 11)
        plt.imshow(X_noisy[i], cmap='binary')
        plt.title('Noisy', fontsize=8)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def model_reconstruct_images(model, X_noisy, model_status='before'):
    x = X_noisy[:10]
    y = model(X_noisy[:10])
    plt.figure(figsize=(14, 3))
    plt.suptitle(f"{model_status.capitalize()} Model Reconstructed Images", fontsize=16, y=1.05)
    for i in range(len(x)):
        plt.subplot(2, 10, i + 1)
        plt.imshow(x[i], cmap='binary')
        plt.title('Noisy Input', fontsize=8)
        plt.axis('off')

        plt.subplot(2, 10, i + 11)
        plt.imshow(y[i], cmap='binary')
        plt.title('Reconstructed', fontsize=8)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def visualize_compressed(x_compressed, y, images, result='PCA_Results', class_names=None):
    plt.figure(figsize=(8, 6))
    cmap = plt.cm.tab10
    Z = (x_compressed - x_compressed.min()) / (x_compressed.max() - x_compressed.min())  # normalize
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
        handles = [plt.Line2D([0], [0], marker='o', color='w',
                              label=class_names[i],
                              markerfacecolor=cmap(i), markersize=6)
                   for i in np.unique(y)]
        plt.legend(handles=handles, loc='best', fontsize=10)

    plt.axis("off")
    plt.title(f"{result}: 2D Latent Space with Class Labels and Sample Images", fontsize=14)
    plt.tight_layout()
    plt.show()
