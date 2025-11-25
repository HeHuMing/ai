# noinspection PyPackageRequirements
import numpy as np
import pandas as pd
from collections import Counter
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置中文字体和图形样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


class DecisionTreeID3:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        self.feature_importance_ = None

    def _entropy(self, y):
        if len(y) == 0:
            return 0
        counts = np.bincount(y)
        probabilities = counts / len(y)
        entropy = 0
        for p in probabilities:
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    def _information_gain(self, X, y, feature_idx, threshold=None):
        parent_entropy = self._entropy(y)

        if threshold is not None:
            left_mask = X[:, feature_idx] <= threshold
            right_mask = X[:, feature_idx] > threshold
            n = len(y)
            n_left, n_right = np.sum(left_mask), np.sum(right_mask)

            if n_left == 0 or n_right == 0:
                return 0

            child_entropy = (n_left / n) * self._entropy(y[left_mask]) + \
                            (n_right / n) * self._entropy(y[right_mask])
        else:
            unique_values = np.unique(X[:, feature_idx])
            child_entropy = 0
            for value in unique_values:
                mask = X[:, feature_idx] == value
                if np.sum(mask) > 0:
                    weight = np.sum(mask) / len(y)
                    child_entropy += weight * self._entropy(y[mask])

        return parent_entropy - child_entropy

    def _find_best_split(self, X, y, feature_indices):
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature_idx in feature_indices:
            feature_values = np.unique(X[:, feature_idx])
            for i in range(len(feature_values) - 1):
                threshold = (feature_values[i] + feature_values[i + 1]) / 2
                gain = self._information_gain(X, y, feature_idx, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, y, feature_indices, depth=0):
        n_samples = len(y)
        n_classes = len(np.unique(y))

        if (len(np.unique(y)) == 1 or
                n_samples < self.min_samples_split or
                (self.max_depth is not None and depth >= self.max_depth) or
                len(feature_indices) == 0):
            return self._create_leaf_node(y)

        best_feature, best_threshold, best_gain = self._find_best_split(X, y, feature_indices)

        if best_gain == 0:
            return self._create_leaf_node(y)

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = X[:, best_feature] > best_threshold

        left_subtree = self._build_tree(X[left_mask], y[left_mask], feature_indices, depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], feature_indices, depth + 1)

        return {
            'feature_index': best_feature,
            'feature_name': self.feature_names[best_feature],
            'threshold': best_threshold,
            'gain': best_gain,
            'samples': n_samples,
            'depth': depth,
            'left': left_subtree,
            'right': right_subtree
        }

    def _create_leaf_node(self, y):
        counts = np.bincount(y)
        return {
            'class': np.argmax(counts),
            'confidence': np.max(counts) / len(y) if len(y) > 0 else 0,
            'samples': len(y),
            'class_distribution': counts
        }

    def fit(self, X, y):
        self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        feature_indices = list(range(X.shape[1]))
        self.tree = self._build_tree(X, y, feature_indices)
        self._calculate_feature_importance()
        return self

    def _calculate_feature_importance(self):
        """计算特征重要性"""
        self.feature_importance_ = np.zeros(len(self.feature_names))
        self._traverse_tree_for_importance(self.tree)
        # 归一化
        total_importance = np.sum(self.feature_importance_)
        if total_importance > 0:
            self.feature_importance_ /= total_importance

    def _traverse_tree_for_importance(self, node):
        if 'class' not in node:  # 非叶节点
            feature_idx = node['feature_index']
            self.feature_importance_[feature_idx] += node['gain'] * node['samples']
            self._traverse_tree_for_importance(node['left'])
            self._traverse_tree_for_importance(node['right'])

    def _predict_sample(self, x, node):
        if 'class' in node:
            return node['class'], node['confidence']
        if x[node['feature_index']] <= node['threshold']:
            return self._predict_sample(x, node['left'])
        else:
            return self._predict_sample(x, node['right'])

    def predict(self, X):
        if self.tree is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return np.array([self._predict_sample(x, self.tree)[0] for x in X])

    def predict_proba(self, X):
        """预测概率"""
        if self.tree is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return np.array([self._predict_sample(x, self.tree)[1] for x in X])

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)


class DecisionTreeVisualizer:
    def __init__(self, model, feature_names, target_names):
        self.model = model
        self.feature_names = feature_names
        self.target_names = target_names

    def plot_tree_structure(self, figsize=(15, 10)):
        """绘制决策树结构图"""
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('off')

        def plot_node(node, x, y, dx, dy):
            if 'class' in node:  # 叶节点
                class_name = self.target_names[node['class']]
                confidence = node['confidence']
                samples = node['samples']

                # 绘制叶节点
                ax.text(x, y, f'{class_name}\n置信度: {confidence:.2f}\n样本数: {samples}',
                        ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7),
                        fontsize=9)
            else:
                # 绘制决策节点
                feature_name = self.feature_names[node['feature_index']]
                threshold = node['threshold']
                gain = node['gain']
                samples = node['samples']

                ax.text(x, y, f'{feature_name} <= {threshold:.2f}\n信息增益: {gain:.3f}\n样本数: {samples}',
                        ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7),
                        fontsize=8)

                # 绘制连接线
                ax.plot([x, x - dx], [y - 0.1, y - dy + 0.1], 'k-', lw=1)
                ax.plot([x, x + dx], [y - 0.1, y - dy + 0.1], 'k-', lw=1)

                # 递归绘制子节点
                plot_node(node['left'], x - dx, y - dy, dx / 2, dy)
                plot_node(node['right'], x + dx, y - dy, dx / 2, dy)

                # 添加分支标签
                ax.text(x - dx / 2, y - dy / 2, '是', ha='center', va='center', fontsize=7)
                ax.text(x + dx / 2, y - dy / 2, '否', ha='center', va='center', fontsize=7)

        plot_node(self.model.tree, 0.5, 0.95, 0.2, 0.2)
        plt.title('决策树结构可视化', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        return fig

    def plot_feature_importance(self, figsize=(10, 6)):
        """绘制特征重要性图"""
        fig, ax = plt.subplots(figsize=figsize)

        if hasattr(self.model, 'feature_importance_'):
            indices = np.argsort(self.model.feature_importance_)[::-1]
            features = [self.feature_names[i] for i in indices]
            importance = self.model.feature_importance_[indices]

            bars = ax.barh(range(len(features)), importance, color='skyblue')
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features)
            ax.invert_yaxis()
            ax.set_xlabel('特征重要性')
            ax.set_title('决策树特征重要性排序', fontsize=14, fontweight='bold')

            # 在条形上添加数值
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                        f'{width:.3f}', ha='left', va='center')

        plt.tight_layout()
        return fig

    def plot_confusion_matrix(self, X_test, y_test, figsize=(8, 6)):
        """绘制混淆矩阵"""
        y_pred = self.model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.target_names,
                    yticklabels=self.target_names, ax=ax)
        ax.set_xlabel('预测标签')
        ax.set_ylabel('真实标签')
        ax.set_title('混淆矩阵', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig

    def plot_decision_boundaries(self, X, y, features_idx=(0, 1), figsize=(12, 4)):
        """绘制决策边界（选择两个特征）"""
        if len(features_idx) != 2:
            raise ValueError("请选择两个特征进行可视化")

        feat1, feat2 = features_idx
        feature1_name = self.feature_names[feat1]
        feature2_name = self.feature_names[feat2]

        # 创建网格点
        x_min, x_max = X[:, feat1].min() - 0.5, X[:, feat1].max() + 0.5
        y_min, y_max = X[:, feat2].min() - 0.5, X[:, feat2].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                             np.arange(y_min, y_max, 0.02))

        # 为网格点创建特征矩阵（使用训练集的均值填充其他特征）
        grid_points = np.tile(X.mean(axis=0), (len(xx.ravel()), 1))
        grid_points[:, feat1] = xx.ravel()
        grid_points[:, feat2] = yy.ravel()

        # 预测网格点
        Z = self.model.predict(grid_points)
        Z = Z.reshape(xx.shape)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # 左侧：决策边界
        contour = ax1.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
        scatter = ax1.scatter(X[:, feat1], X[:, feat2], c=y,
                              cmap=plt.cm.RdYlBu, edgecolor='black', s=50)
        ax1.set_xlabel(feature1_name)
        ax1.set_ylabel(feature2_name)
        ax1.set_title(f'决策边界 ({feature1_name} vs {feature2_name})')
        plt.colorbar(contour, ax=ax1)

        # 右侧：分类结果散点图
        y_pred = self.model.predict(X)
        correct = y_pred == y
        colors = ['green' if c else 'red' for c in correct]
        ax2.scatter(X[:, feat1], X[:, feat2], c=colors, alpha=0.6, s=50)
        ax2.set_xlabel(feature1_name)
        ax2.set_ylabel(feature2_name)
        ax2.set_title('分类结果（绿色:正确, 红色:错误）')

        plt.tight_layout()
        return fig

    def plot_training_history(self, X_train, y_train, X_test, y_test, figsize=(10, 6)):
        """绘制训练过程准确率变化"""
        # 模拟不同深度下的准确率变化
        depths = range(1, 8)
        train_scores = []
        test_scores = []

        for depth in depths:
            temp_model = DecisionTreeID3(max_depth=depth)
            temp_model.fit(X_train, y_train)
            train_scores.append(temp_model.score(X_train, y_train))
            test_scores.append(temp_model.score(X_test, y_test))

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(depths, train_scores, 'o-', label='训练集准确率', linewidth=2)
        ax.plot(depths, test_scores, 's-', label='测试集准确率', linewidth=2)
        ax.set_xlabel('决策树深度')
        ax.set_ylabel('准确率')
        ax.set_title('不同深度下的模型性能')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig


def main():
    """主函数：完整的测试和可视化"""
    print("=== 决策树ID3算法实现 - 鸢尾花数据集分类与可视化 ===\n")

    # 1. 加载数据
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    print("数据集信息:")
    print(f"- 样本数量: {X.shape[0]}")
    print(f"- 特征数量: {X.shape[1]}")
    print(f"- 特征名称: {list(feature_names)}")
    print(f"- 类别名称: {list(target_names)}")
    print(f"- 类别分布: {dict(zip(target_names, np.bincount(y)))}\n")

    # 2. 数据划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print("数据划分:")
    print(f"- 训练集大小: {X_train.shape[0]}")
    print(f"- 测试集大小: {X_test.shape[0]}")

    # 3. 训练模型
    print("\n训练决策树模型中...")
    dt = DecisionTreeID3(max_depth=4, min_samples_split=5)
    dt.feature_names = feature_names
    dt.fit(X_train, y_train)

    # 4. 模型评估
    train_accuracy = dt.score(X_train, y_train)
    test_accuracy = dt.score(X_test, y_test)

    print("\n模型性能评估:")
    print(f"- 训练集准确率: {train_accuracy:.3f}")
    print(f"- 测试集准确率: {test_accuracy:.3f}")

    # 5. 创建可视化器
    visualizer = DecisionTreeVisualizer(dt, feature_names, target_names)

    # 6. 生成所有可视化图表
    print("\n生成可视化图表...")

    # 特征重要性图
    fig1 = visualizer.plot_feature_importance()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')

    # 混淆矩阵
    fig2 = visualizer.plot_confusion_matrix(X_test, y_test)
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')

    # 决策边界图（选择最重要的两个特征）
    fig3 = visualizer.plot_decision_boundaries(X, y, features_idx=(2, 3))  # 花瓣长度 vs 花瓣宽度
    plt.savefig('decision_boundaries.png', dpi=300, bbox_inches='tight')

    # 训练历史图
    fig4 = visualizer.plot_training_history(X_train, y_train, X_test, y_test)
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')

    # 决策树结构图（由于较复杂，保存为高分辨率图片）
    fig5 = visualizer.plot_tree_structure(figsize=(20, 12))
    plt.savefig('tree_structure.png', dpi=300, bbox_inches='tight')

    # 7. 显示详细分类报告
    y_pred = dt.predict(X_test)
    print("\n详细分类报告:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    # 8. 显示预测示例
    print("\n随机预测示例:")
    sample_indices = np.random.choice(len(X_test), 8, replace=False)
    for i, idx in enumerate(sample_indices):
        actual = target_names[y_test[idx]]
        predicted = target_names[y_pred[idx]]
        confidence = dt.predict_proba(X_test[idx:idx + 1])[0]
        status = "✓" if actual == predicted else "✗"
        print(f"样本{i + 1}: {status} 实际={actual:12} 预测={predicted:12} 置信度={confidence:.3f}")

    print(f"\n所有可视化图表已保存为PNG文件")
    print("可以关闭图表窗口继续程序...")
    plt.show()


if __name__ == "__main__":
    main()