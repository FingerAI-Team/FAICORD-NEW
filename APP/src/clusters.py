from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
import numpy as np


class KNNCluster:
    def relabel_by_knn(
        self,
        embeddings,
        labels,
        k: int = 5,
        *,
        weighted: bool = True,            # 거리 가중 다수결(권장)
        keep_original_on_tie: bool = True,# 동률/근소차면 원 라벨 유지
        sim_margin: float = 0.02,         # 승자-차점 가중치 마진(0.0~0.05 권장)
        exclude_labels: tuple[str, ...] = (),  # 투표에서 제외(예: ("UNKNOWN","filler"))
        return_confidence: bool = False   # 각 샘플의 신뢰도 반환
    ):
        """
        embeddings: (N,D) L2-정규화 상태 가정
        labels:     list[str], 길이 N
        k:          최근접 이웃 수 (자동으로 min(k, N-1)로 조정)

        반환:
          - return_confidence=False: new_labels (list[str])
          - return_confidence=True : (new_labels, confidences)
        """
        X = np.asarray(embeddings, dtype=np.float32)
        y = np.asarray(labels)
        N = len(X)

        if N == 0:
            return ([], []) if return_confidence else []
        if N == 1:
            return ([y[0]], [1.0]) if return_confidence else [y[0]]

        k = max(1, min(k, N - 1))

        # cosine distance = 1 - cosine_similarity
        nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute", n_jobs=-1)
        nn.fit(X)
        dists, idxs = nn.kneighbors(X)  # 포함된 0번째는 자기 자신

        new_labels = []
        confidences = []

        for i in range(N):
            neigh = idxs[i][1:]              # 자기 자신 제외
            dist  = dists[i][1:]
            neigh_labels = y[neigh]

            # 제외 라벨 필터
            if exclude_labels:
                mask = ~np.isin(neigh_labels, exclude_labels)
                neigh = neigh[mask]
                dist  = dist[mask]
                neigh_labels = neigh_labels[mask]

            # 투표할 이웃이 아예 없으면 원 라벨 유지
            if len(neigh) == 0:
                new_labels.append(y[i])
                confidences.append(1.0)
                continue

            if weighted:
                # sim = 1 - dist  (정규화 가정 → 코사인 유사도)
                sim = 1.0 - np.clip(dist, 0.0, 2.0)
                # 음수 유사도는 가중치 0으로 클립(안정성)
                sim = np.clip(sim, 0.0, 1.0)

                weights = {}
                for lab, s in zip(neigh_labels, sim):
                    weights[lab] = weights.get(lab, 0.0) + float(s)

                # 가중치 상위 2개로 승자/마진 계산
                ranked = sorted(weights.items(), key=lambda x: x[1], reverse=True)
                winner_lab, w1 = ranked[0]
                if len(ranked) > 1:
                    w2 = ranked[1][1]
                    margin = w1 - w2
                    if keep_original_on_tie and margin < sim_margin:
                        winner_lab = y[i]
                    conf = float(w1 / (w1 + w2)) if (w1 + w2) > 0 else 1.0
                else:
                    conf = 1.0
            else:
                # 단순 다수결
                unique, counts = np.unique(neigh_labels, return_counts=True)
                order = np.argsort(-counts)
                winner_lab = unique[order[0]]
                if keep_original_on_tie and len(order) > 1 and counts[order[0]] == counts[order[1]]:
                    winner_lab = y[i]
                conf = float(counts[order[0]] / counts.sum())

            new_labels.append(winner_lab)
            confidences.append(conf)
        return (new_labels, confidences) if return_confidence else new_labels