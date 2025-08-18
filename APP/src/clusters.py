from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
import numpy as np

class KNNCluster:
    def relabel_by_knn(self, embeddings, labels, k=5):
        """
        embeddings: np.array, shape (N, D)
        labels: list of str, e.g. ['SPEAKER_00', 'SPEAKER_01']
        k: number of neighbors
        return: new_labels (list of str)
        """
        embeddings = np.array(embeddings)
        labels = np.array(labels)

        if len(embeddings) <= k:
            print(f"[SKIP] Too few samples: {len(embeddings)} <= k={k}")
            return labels.tolist()

        # 문자열 → 정수 ID로 인코딩
        le = LabelEncoder()
        label_ids = le.fit_transform(labels)

        knn = NearestNeighbors(n_neighbors=k+1, metric='cosine')
        knn.fit(embeddings)
        _, indices = knn.kneighbors(embeddings)

        new_label_ids = []
        for i in range(len(embeddings)):
            neighbor_idx = indices[i][1:]  # 자기 자신 제외
            neighbor_label_ids = label_ids[neighbor_idx]
            majority_id = np.bincount(neighbor_label_ids).argmax()
            new_label_ids.append(majority_id)

        # 정수 ID → 원래 라벨로 복원
        return le.inverse_transform(new_label_ids).tolist()