import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import seaborn as sns
from scipy.spatial.distance import cdist
import warnings

warnings.filterwarnings('ignore')

# 1. 传统K-means聚类算法实现（随机初始化质心）
class Traditional_Kmeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):
        """
        初始化传统K-means聚类器（随机初始化质心）

        参数:
        n_clusters: 聚类数量
        max_iter: 最大迭代次数
        tol: 收敛容忍度
        random_state: 随机种子
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia_ = None
        self.n_iter_ = 0

    def _initialize_centroids(self, X):
        """传统随机初始化质心：从样本中随机选择k个样本作为初始质心"""
        n_samples, n_features = X.shape

        if self.random_state is not None:
            np.random.seed(self.random_state)

        # 随机选择k个不同的样本作为初始质心
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        centroids = X[random_indices]

        return centroids

    def _assign_clusters(self, X, centroids):
        """将样本分配到最近的质心"""
        distances = cdist(X, centroids, metric='euclidean')
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        """更新质心位置"""
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            if np.sum(labels == k) > 0:
                centroids[k] = X[labels == k].mean(axis=0)
            else:
                # 如果某个簇没有样本，重新随机初始化该质心
                centroids[k] = X[np.random.randint(X.shape[0])]
        return centroids

    def fit(self, X):
        """
        训练传统K-means模型

        参数:
        X: 输入数据，形状为(n_samples, n_features)
        """
        # 初始化质心（传统随机方式）
        self.centroids = self._initialize_centroids(X)

        for i in range(self.max_iter):
            # 分配簇标签
            labels = self._assign_clusters(X, self.centroids)

            # 更新质心
            new_centroids = self._update_centroids(X, labels)

            # 检查收敛
            centroid_shift = np.sum(np.sqrt(np.sum((new_centroids - self.centroids) ** 2, axis=1)))

            if centroid_shift < self.tol:
                self.centroids = new_centroids
                self.labels = labels
                self.n_iter_ = i + 1
                break

            self.centroids = new_centroids
            self.labels = labels
            self.n_iter_ = i + 1

        # 计算惯性（簇内平方和）
        distances = cdist(X, self.centroids, metric='euclidean')
        min_distances = np.min(distances, axis=1)
        self.inertia_ = np.sum(min_distances ** 2)

        return self

    def predict(self, X):
        """预测新样本的簇标签"""
        return self._assign_clusters(X, self.centroids)

# 2. 优化的K-means++聚类算法实现（保留原有实现）
class Optimized_Kmeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):
        """
        初始化K-means聚类器（K-means++初始化）

        参数:
        n_clusters: 聚类数量
        max_iter: 最大迭代次数
        tol: 收敛容忍度
        random_state: 随机种子
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia_ = None
        self.n_iter_ = 0

    def _initialize_centroids(self, X):
        """初始化质心 - 使用K-means++改进"""
        n_samples, n_features = X.shape

        if self.random_state is not None:
            np.random.seed(self.random_state)

        # 第一个质心随机选择
        centroids = np.zeros((self.n_clusters, n_features))
        centroids[0] = X[np.random.randint(n_samples)]

        # 使用K-means++算法选择剩余质心
        for i in range(1, self.n_clusters):
            # 计算每个样本到最近质心的距离
            distances = cdist(X, centroids[:i], metric='euclidean')
            min_distances = np.min(distances, axis=1)

            # 将距离平方作为概率分布
            probabilities = min_distances ** 2
            probabilities /= probabilities.sum()

            # 根据概率选择下一个质心
            cumulative_prob = np.cumsum(probabilities)
            r = np.random.rand()
            for j, p in enumerate(cumulative_prob):
                if r < p:
                    centroids[i] = X[j]
                    break

        return centroids

    def _assign_clusters(self, X, centroids):
        """将样本分配到最近的质心"""
        distances = cdist(X, centroids, metric='euclidean')
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        """更新质心位置"""
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            if np.sum(labels == k) > 0:
                centroids[k] = X[labels == k].mean(axis=0)
            else:
                # 如果某个簇没有样本，重新随机初始化该质心
                centroids[k] = X[np.random.randint(X.shape[0])]
        return centroids

    def fit(self, X):
        """
        训练K-means模型

        参数:
        X: 输入数据，形状为(n_samples, n_features)
        """
        # 初始化质心（K-means++方式）
        self.centroids = self._initialize_centroids(X)

        for i in range(self.max_iter):
            # 分配簇标签
            labels = self._assign_clusters(X, self.centroids)

            # 更新质心
            new_centroids = self._update_centroids(X, labels)

            # 检查收敛
            centroid_shift = np.sum(np.sqrt(np.sum((new_centroids - self.centroids) ** 2, axis=1)))

            if centroid_shift < self.tol:
                self.centroids = new_centroids
                self.labels = labels
                self.n_iter_ = i + 1
                break

            self.centroids = new_centroids
            self.labels = labels
            self.n_iter_ = i + 1

        # 计算惯性（簇内平方和）
        distances = cdist(X, self.centroids, metric='euclidean')
        min_distances = np.min(distances, axis=1)
        self.inertia_ = np.sum(min_distances ** 2)

        return self

    def predict(self, X):
        """预测新样本的簇标签"""
        return self._assign_clusters(X, self.centroids)

# 3. 加载和预处理wine数据集
def load_and_preprocess_data():
    """加载wine数据集并进行预处理"""
    # 加载数据集
    wine_data = load_wine()
    X = wine_data.data
    y = wine_data.target
    feature_names = wine_data.feature_names
    target_names = wine_data.target_names

    print("数据集信息:")
    print(f"样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
    print(f"特征名: {feature_names}")
    print(f"类别名: {target_names}")
    print(f"真实标签分布: {np.bincount(y)}")
    print("\n" + "=" * 50 + "\n")

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, feature_names, target_names

# 4. 肘部法则确定最佳K值（兼容两种K-means）
def elbow_method(X, max_k=10, kmeans_type='traditional'):
    """
    使用肘部法则确定最佳聚类数
    参数:
        X: 输入数据
        max_k: 最大尝试的k值
        kmeans_type: 'traditional' 或 'optimized'，选择使用的K-means类型
    """
    inertias = []
    k_values = range(1, max_k + 1)

    for k in k_values:
        if k == 1:
            # 当k=1时，质心是数据的均值
            inertia = np.sum((X - X.mean(axis=0)) ** 2)
            inertias.append(inertia)
        else:
            if kmeans_type == 'traditional':
                kmeans = Traditional_Kmeans(n_clusters=k, random_state=42)
            else:
                kmeans = Optimized_Kmeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)

    # 计算每个k值的下降率变化
    rates_of_change = []
    for i in range(1, len(inertias) - 1):
        rate = (inertias[i - 1] - inertias[i]) / (inertias[i] - inertias[i + 1])
        rates_of_change.append(rate)

    # 找到肘部点（下降率变化最大的点）
    if rates_of_change:
        elbow_k = np.argmax(rates_of_change) + 2  # +2因为从k=2开始计算
    else:
        elbow_k = 3

    return k_values, inertias, elbow_k

# 5. 聚类性能评估
def evaluate_clustering(X, y_pred, y_true=None):
    """评估聚类结果"""
    evaluation = {}

    # 轮廓系数（不需要真实标签）
    silhouette_avg = silhouette_score(X, y_pred)
    evaluation['silhouette_score'] = silhouette_avg

    # Calinski-Harabasz指数
    calinski_harabasz = calinski_harabasz_score(X, y_pred)
    evaluation['calinski_harabasz_score'] = calinski_harabasz

    # Davies-Bouldin指数（值越小越好）
    davies_bouldin = davies_bouldin_score(X, y_pred)
    evaluation['davies_bouldin_score'] = davies_bouldin

    # 如果有真实标签，计算外部指标
    if y_true is not None:
        # 计算调整兰德指数（需要真实标签）
        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score(y_true, y_pred)
        evaluation['adjusted_rand_score'] = ari

        # 计算互信息
        from sklearn.metrics import normalized_mutual_info_score
        nmi = normalized_mutual_info_score(y_true, y_pred)
        evaluation['normalized_mutual_info_score'] = nmi

    return evaluation

# 6. 可视化函数
def plot_elbow_method(k_values, inertias, elbow_k, title_suffix=""):
    """绘制肘部法则图"""
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of clusters (k)', fontsize=12)
    plt.ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
    plt.title(f'Elbow Method for Optimal k {title_suffix}', fontsize=14, fontweight='bold')
    plt.axvline(x=elbow_k, color='r', linestyle='--', alpha=0.7, label=f'Elbow point: k={elbow_k}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_cluster_distribution(labels, title_suffix=""):
    """绘制簇分布图"""
    unique, counts = np.unique(labels, return_counts=True)

    plt.figure(figsize=(8, 5))
    bars = plt.bar(unique, counts, color=plt.cm.tab10(np.arange(len(unique)) / len(unique)))
    plt.xlabel('Cluster', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title(f'Cluster Distribution {title_suffix}', fontsize=14, fontweight='bold')
    plt.xticks(unique)

    # 在每个柱子上显示数量
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{count}', ha='center', va='bottom')

    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

def plot_pca_visualization(X, labels, centroids=None, title_suffix=""):
    """使用PCA降维可视化聚类结果"""
    # 使用PCA降维到2维
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(10, 7))

    # 绘制数据点
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels,
                          cmap='tab10', s=50, alpha=0.7, edgecolors='k', linewidth=0.5)

    # 如果提供了质心，绘制质心
    if centroids is not None:
        centroids_pca = pca.transform(centroids)
        plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
                    c='red', s=300, marker='X', edgecolors='k', linewidth=2, label='Centroids')
        plt.legend()

    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    plt.title(f'PCA Visualization of Clusters {title_suffix}', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return X_pca, pca

def plot_pairwise_features(X, labels, feature_names, title_suffix="", n_features=4):
    """绘制特征对之间的散点图矩阵"""
    if n_features > len(feature_names):
        n_features = len(feature_names)

    # 选择最重要的特征（基于方差）
    variances = np.var(X, axis=0)
    important_features = np.argsort(variances)[-n_features:][::-1]

    fig, axes = plt.subplots(n_features, n_features, figsize=(15, 15))

    for i, idx_i in enumerate(important_features):
        for j, idx_j in enumerate(important_features):
            ax = axes[i, j]

            if i == j:
                # 对角线：显示直方图
                ax.hist(X[:, idx_i], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax.set_title(feature_names[idx_i], fontsize=10)
            else:
                # 非对角线：显示散点图
                scatter = ax.scatter(X[:, idx_j], X[:, idx_i], c=labels,
                                     cmap='tab10', s=30, alpha=0.6, edgecolors='k', linewidth=0.3)
                if i == n_features - 1:
                    ax.set_xlabel(feature_names[idx_j], fontsize=9)
                if j == 0:
                    ax.set_ylabel(feature_names[idx_i], fontsize=9)

            ax.tick_params(labelsize=8)

    plt.suptitle(f'Pairwise Feature Scatter Matrix with Clusters {title_suffix}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

def plot_silhouette_analysis(X, labels, n_clusters, title_suffix=""):
    """绘制轮廓分析图"""
    from sklearn.metrics import silhouette_samples

    silhouette_vals = silhouette_samples(X, labels)
    y_lower = 10

    plt.figure(figsize=(10, 8))

    for i in range(n_clusters):
        # 获取第i个簇的轮廓值
        cluster_silhouette_vals = silhouette_vals[labels == i]
        cluster_silhouette_vals.sort()

        size_cluster_i = cluster_silhouette_vals.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.tab10(i / n_clusters)
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                          0, cluster_silhouette_vals,
                          facecolor=color, edgecolor=color, alpha=0.7)

        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    plt.axvline(x=np.mean(silhouette_vals), color="red", linestyle="--",
                label=f'Average silhouette: {np.mean(silhouette_vals):.3f}')

    plt.xlabel("Silhouette coefficient values", fontsize=12)
    plt.ylabel("Cluster label", fontsize=12)
    plt.title(f"Silhouette Analysis for KMeans (k={n_clusters}) {title_suffix}", fontsize=14, fontweight='bold')
    plt.yticks([])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# 7. 主函数
def main():
    print("=" * 50)
    print("K-means聚类算法在Wine数据集上的实现与验证")
    print("=" * 50)

    # 加载和预处理数据
    X, y_true, feature_names, target_names = load_and_preprocess_data()

    # --------------------- 第一步：传统K-means分析 ---------------------
    print("\n" + "-" * 50)
    print("1. 传统K-means（随机初始化）分析")
    print("-" * 50)

    # 使用肘部法则确定最佳K值（传统K-means）
    print("\n1.1 使用肘部法则确定最佳聚类数...")
    k_values_trad, inertias_trad, elbow_k_trad = elbow_method(X, max_k=10, kmeans_type='traditional')
    print(f"传统K-means建议的聚类数 (肘部法则): k = {elbow_k_trad}")

    # 绘制肘部法则图
    plot_elbow_method(k_values_trad, inertias_trad, elbow_k_trad, "(Traditional K-means)")

    # 使用建议的K值进行聚类（传统K-means）
    print(f"\n1.2 使用传统K-means进行聚类 (k={elbow_k_trad})...")
    kmeans_trad = Traditional_Kmeans(n_clusters=elbow_k_trad, random_state=42)
    kmeans_trad.fit(X)
    y_pred_trad = kmeans_trad.labels

    print(f"迭代次数: {kmeans_trad.n_iter_}")
    print(f"簇内平方和 (Inertia): {kmeans_trad.inertia_:.4f}")
    print(f"簇大小分布: {np.bincount(y_pred_trad)}")

    # 评估聚类结果（传统K-means）
    print("\n1.3 评估传统K-means聚类结果...")
    evaluation_trad = evaluate_clustering(X, y_pred_trad, y_true)

    print("传统K-means聚类性能指标:")
    for metric, value in evaluation_trad.items():
        print(f"  {metric}: {value:.4f}")

    # 可视化传统K-means聚类结果
    print("\n1.4 可视化传统K-means聚类结果...")
    plot_cluster_distribution(y_pred_trad, "(Traditional K-means)")
    X_pca_trad, pca_model_trad = plot_pca_visualization(X, y_pred_trad, kmeans_trad.centroids, "(Traditional K-means)")
    plot_silhouette_analysis(X, y_pred_trad, elbow_k_trad, "(Traditional K-means)")
    plot_pairwise_features(X, y_pred_trad, feature_names, "(Traditional K-means)", n_features=4)

    # --------------------- 第二步：优化K-means++分析 ---------------------
    print("\n" + "-" * 50)
    print("2. 优化K-means++（K-means++初始化）分析")
    print("-" * 50)

    # 使用肘部法则确定最佳K值（优化K-means++）
    print("\n2.1 使用肘部法则确定最佳聚类数...")
    k_values_opt, inertias_opt, elbow_k_opt = elbow_method(X, max_k=10, kmeans_type='optimized')
    print(f"优化K-means++建议的聚类数 (肘部法则): k = {elbow_k_opt}")

    # 绘制肘部法则图
    plot_elbow_method(k_values_opt, inertias_opt, elbow_k_opt, "(Optimized K-means++)")

    # 使用建议的K值进行聚类（优化K-means++）
    print(f"\n2.2 使用优化K-means++进行聚类 (k={elbow_k_opt})...")
    kmeans_opt = Optimized_Kmeans(n_clusters=elbow_k_opt, random_state=42)
    kmeans_opt.fit(X)
    y_pred_opt = kmeans_opt.labels

    print(f"迭代次数: {kmeans_opt.n_iter_}")
    print(f"簇内平方和 (Inertia): {kmeans_opt.inertia_:.4f}")
    print(f"簇大小分布: {np.bincount(y_pred_opt)}")

    # 评估聚类结果（优化K-means++）
    print("\n2.3 评估优化K-means++聚类结果...")
    evaluation_opt = evaluate_clustering(X, y_pred_opt, y_true)

    print("优化K-means++聚类性能指标:")
    for metric, value in evaluation_opt.items():
        print(f"  {metric}: {value:.4f}")

    # 可视化优化K-means++聚类结果
    print("\n2.4 可视化优化K-means++聚类结果...")
    plot_cluster_distribution(y_pred_opt, "(Optimized K-means++)")
    X_pca_opt, pca_model_opt = plot_pca_visualization(X, y_pred_opt, kmeans_opt.centroids, "(Optimized K-means++)")
    plot_silhouette_analysis(X, y_pred_opt, elbow_k_opt, "(Optimized K-means++)")
    plot_pairwise_features(X, y_pred_opt, feature_names, "(Optimized K-means++)", n_features=4)

    # --------------------- 第三步：两种算法对比 ---------------------
    print("\n" + "-" * 50)
    print("3. 传统K-means vs 优化K-means++ 对比")
    print("-" * 50)

    # 对比不同K值的性能
    print("\n3.1 对比不同K值的聚类性能...")
    results = []
    for k in [2, 3, 4, 5, 6]:
        # 传统K-means
        kmeans_trad_k = Traditional_Kmeans(n_clusters=k, random_state=42)
        kmeans_trad_k.fit(X)
        sil_trad = silhouette_score(X, kmeans_trad_k.labels)
        cal_trad = calinski_harabasz_score(X, kmeans_trad_k.labels)
        dav_trad = davies_bouldin_score(X, kmeans_trad_k.labels)

        # 优化K-means++
        kmeans_opt_k = Optimized_Kmeans(n_clusters=k, random_state=42)
        kmeans_opt_k.fit(X)
        sil_opt = silhouette_score(X, kmeans_opt_k.labels)
        cal_opt = calinski_harabasz_score(X, kmeans_opt_k.labels)
        dav_opt = davies_bouldin_score(X, kmeans_opt_k.labels)

        results.append({
            'k': k,
            '传统K-means Inertia': kmeans_trad_k.inertia_,
            'K-means++ Inertia': kmeans_opt_k.inertia_,
            '传统K-means 轮廓系数': sil_trad,
            'K-means++ 轮廓系数': sil_opt,
            '传统K-means CH指数': cal_trad,
            'K-means++ CH指数': cal_opt,
            '传统K-means DB指数': dav_trad,
            'K-means++ DB指数': dav_opt
        })

    # 创建结果DataFrame
    results_df = pd.DataFrame(results)
    print("\n不同K值的聚类性能对比:")
    print(results_df.to_string(index=False))

    # 可视化不同K值的性能对比
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Inertia对比
    axes[0, 0].plot(results_df['k'], results_df['传统K-means Inertia'], 'bo-', label='Traditional K-means', linewidth=2, markersize=8)
    axes[0, 0].plot(results_df['k'], results_df['K-means++ Inertia'], 'ro-', label='K-means++', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Number of clusters (k)')
    axes[0, 0].set_ylabel('Inertia')
    axes[0, 0].set_title('Inertia vs. k')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # 轮廓系数对比
    axes[0, 1].plot(results_df['k'], results_df['传统K-means 轮廓系数'], 'bo-', label='Traditional K-means', linewidth=2, markersize=8)
    axes[0, 1].plot(results_df['k'], results_df['K-means++ 轮廓系数'], 'ro-', label='K-means++', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Number of clusters (k)')
    axes[0, 1].set_ylabel('Silhouette Score')
    axes[0, 1].set_title('Silhouette Score vs. k')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # CH指数对比
    axes[1, 0].plot(results_df['k'], results_df['传统K-means CH指数'], 'bo-', label='Traditional K-means', linewidth=2, markersize=8)
    axes[1, 0].plot(results_df['k'], results_df['K-means++ CH指数'], 'ro-', label='K-means++', linewidth=2, markersize=8)
    axes[1, 0].set_xlabel('Number of clusters (k)')
    axes[1, 0].set_ylabel('Calinski-Harabasz Index')
    axes[1, 0].set_title('Calinski-Harabasz Index vs. k')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # DB指数对比
    axes[1, 1].plot(results_df['k'], results_df['传统K-means DB指数'], 'bo-', label='Traditional K-means', linewidth=2, markersize=8)
    axes[1, 1].plot(results_df['k'], results_df['K-means++ DB指数'], 'ro-', label='K-means++', linewidth=2, markersize=8)
    axes[1, 1].set_xlabel('Number of clusters (k)')
    axes[1, 1].set_ylabel('Davies-Bouldin Index')
    axes[1, 1].set_title('Davies-Bouldin Index vs. k')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.suptitle('Clustering Performance Metrics: Traditional K-means vs K-means++', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # 与真实标签的对比
    if y_true is not None:
        print("\n3.2 聚类结果与真实标签对比...")

        # 创建对比可视化
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        # 真实标签的PCA可视化
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        ax1 = axes[0]
        scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y_true,
                               cmap='tab10', s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax1.set_title('True Labels - PCA Visualization')
        plt.colorbar(scatter1, ax=ax1, label='True Class')
        ax1.grid(True, alpha=0.3)

        # 传统K-means的PCA可视化
        ax2 = axes[1]
        scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred_trad,
                               cmap='tab10', s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax2.set_title(f'Traditional K-means (k={elbow_k_trad}) - PCA Visualization')
        plt.colorbar(scatter2, ax=ax2, label='Cluster')
        ax2.grid(True, alpha=0.3)

        # 优化K-means++的PCA可视化
        ax3 = axes[2]
        scatter3 = ax3.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred_opt,
                               cmap='tab10', s=50, alpha=0.7, edgecolors='k', linewidth=0.5)
        ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax3.set_title(f'Optimized K-means++ (k={elbow_k_opt}) - PCA Visualization')
        plt.colorbar(scatter3, ax=ax3, label='Cluster')
        ax3.grid(True, alpha=0.3)

        plt.suptitle('Comparison: True Labels vs Traditional K-means vs K-means++', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    print("\n" + "=" * 50)
    print("K-means聚类分析完成!")
    print("=" * 50)

# 8. 运行主函数
if __name__ == "__main__":
    main()