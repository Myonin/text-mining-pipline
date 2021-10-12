import numpy as np
import matplotlib.pyplot as plt
import umap

SEED = 123


def process_umap(embd_df, title, alpha):
    np.random.seed(SEED)
    x_umap_store = umap.UMAP(n_components=2).fit_transform(embd_df)
    x_umap_store_3d = umap.UMAP(n_components=3).fit_transform(embd_df)
    plt.figure(figsize=[15, 11])
    plt.title('UMAP -- {}'.format(title))
    plt.scatter(x_umap_store[:, 0], x_umap_store[:, 1], marker='.', alpha=alpha)
    plt.show()
    plt.figure(figsize=[15, 15])
    plt.subplot(111, projection='3d')
    plt.scatter(x_umap_store_3d[:, 2], x_umap_store_3d[:, 0], x_umap_store_3d[:, 1], )
    plt.show()