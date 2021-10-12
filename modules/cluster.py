from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def calculate_number_clusters(data, max_k, max_iter, n_init):
    iters = range(1, max_k)
    sse = []
    for k in tqdm(iters):
        cls = KMeans(n_clusters=k, max_iter=max_iter, n_init=n_init).fit(data)
        sse.append(cls.inertia_)
    plt.figure(figsize=[9, 5])
    plt.plot(iters, sse, marker='.')
    plt.xlabel('Cluster Centers')
    plt.ylabel('SSE')
    plt.title('SSE by Cluster Center Plot')
    plt.show()