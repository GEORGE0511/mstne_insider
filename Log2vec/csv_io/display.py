import pandas as pd
import numpy as np
from sklearn import manifold, datasets
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
MSTNE=np.loadtxt(r"../emb/dblp_htne_attn_3.emb",dtype=np.float32)

def plot_2D(data):
    #n_samples, n_features = data.shape
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0) #使用TSNE对特征降到二维
    #t0 = time()
    result = tsne.fit_transform(data) #降维后的数据
    #print(result.shape)
    #画图
    plt.figure(figsize=(10, 8))
    for i in range(result.shape[0]):
        plt.scatter(result[i,0], result[i,1])
    # plt.legend()
    # fig = plot_embedding(result, label,
    #                      't-SNE embedding of the digits (time %.2fs)')
    #                      #% (time() - t0))
    # fig.subplots_adjust(right=0.8)  #图例过大，保存figure时无法保存完全，故对此参数进行调整
    plt.show()
    plt.savefig("cluster.png")
plot_2D(MSTNE)