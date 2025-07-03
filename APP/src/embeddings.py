from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.inference.speaker import EncoderClassifier
from torchaudio.transforms import Resample
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from pydub import AudioSegment
from io import BytesIO
import torch.nn.functional as F
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
import torchaudio
import wespeaker
import random
import torch
import os 


class BaseEMB:
    def __init__(self):
        self.set_seed()
        self.set_gpu()

    def set_seed(self, seed=42):
        """랜덤 시드 설정"""
        self.seed = seed
        random.seed(self.seed)  
        np.random.seed(self.seed)  
        torch.manual_seed(self.seed)  
        torch.cuda.manual_seed_all(self.seed)    # GPU 연산을 위한 시드 설정
        torch.backends.cudnn.deterministic = True   # 연산 재현성을 보장
        torch.backends.cudnn.benchmark = False    # 성능 최적화 옵션 비활성화

    def set_gpu(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else "cpu"

    def prepare_embeddings(self, embeddings):
        """
        embeddings: list of torch.Tensor or np.ndarray
            - speechbrain: [torch.Tensor of shape (1, 1, 192)]
            - wespeaker: [torch.Tensor of shape (256,)] or [np.ndarray of shape (256,)]
        Returns:
            np.ndarray of shape (N, D)
        """
        processed = []
        for emb in embeddings:
            if isinstance(emb, torch.Tensor):
                emb = emb.view(-1).cpu().numpy()
            elif isinstance(emb, np.ndarray):
                emb = emb.reshape(-1)
            else:
                raise TypeError(f"[prepare_embeddings] Unsupported type: {type(emb)}")
            processed.append(emb)
        return np.stack(processed)

    def calc_similarity_matrix(self, embeddings):
        """
        embeddings: numpy array [N, D]
        return: numpy array [N, N] similarity matrix
        """
        N = embeddings.shape[0]
        sim_matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                sim_matrix[i, j] = 1 - cosine(embeddings[i], embeddings[j])
        return sim_matrix
    
    def get_emb_mean(self, embeddings):
        '''
        embeddings x N  -> np.mean(embeddings)
        화자별 대표 임베딩 구하는데 사용하는 함수 
        '''
        pass
    
    def calc_speaker_mean_embeddings(self, embeddings, labels):
        """
        embeddings: numpy array [N, D]
        labels: list of speaker labels (N개)
        return: dict {speaker: mean_embedding}
        """
        speaker_means = {}
        unique_speakers = set(labels)
        for spk in unique_speakers:
            spk_embs = embeddings[np.array(labels) == spk]
            mean_emb = spk_embs.mean(axis=0)
            speaker_means[spk] = mean_emb
        return speaker_means

    def calc_mean_similarity_matrix(self, speaker_means):
        """
        speaker_means: dict {speaker: mean_embedding}
        return: 2D numpy array (S, S) similarity matrix, S = number of speakers
        """
        speakers = list(speaker_means.keys())
        S = len(speakers)
        sim_matrix = np.zeros((S, S))
        for i in range(S):
            for j in range(S):
                emb_i = speaker_means[speakers[i]]
                emb_j = speaker_means[speakers[j]]
                sim = 1 - cosine(emb_i, emb_j)
                sim_matrix[i, j] = sim
        return sim_matrix, speakers


class WSEMB(BaseEMB):
    def __init__(self):
        super().__init__()
    
    def load_model(self, model_path=None, language='english'):
        if model_path == None: 
            model = wespeaker.load_model(language)
        else:
            model = wespeaker.load_model_local(model_path)
        model.set_device(self.device)
        return model

    def get_embedding(self, model, file_name):
        embedding = model.extract_embedding(file_name)
        return embedding 

    def get_embeddings_from_file(self, model, file_path, file_list):
        '''
        여러 파일들을 입력으로 받아 각 파일별 임베딩 리스트 반환 
        '''
        emb_list = []
        for file_name in file_list: 
            audio_file = os.path.join(file_path, file_name)
            if isinstance(audio_file, AudioSegment):
                buffer = BytesIO()
                audio_file.export(buffer, format='wav')
                buffer.seek(0)
                emb = model.extract_embedding(buffer)
            else:
                emb = model.extract_embedding(audio_file)
            emb_list.append(self.prepare_embeddings(emb).squeeze())
        return emb_list

    def get_embeddings_from_diar(self, model, file_name, diar_result, chunk_offset=0):
        '''
        diar_result: List of ((start_time, end_time), speaker)
        file_name: path to audio file (.wav)
        return: List of ((start_time, end_time), speaker, embedding)
        '''
        audio = AudioSegment.from_wav(file_name)
        emb_results = []
        for (time_s, time_e), speaker in diar_result:
            if speaker == 'filler':
                continue 

            start_ms = int(time_s + chunk_offset) * 1000 
            end_ms = int(time_e + chunk_offset) * 1000
            if (end_ms - start_ms) < 1000:  # 1초 미만
                # print(f"[SKIP] Short segment ({time_s:.2f}s ~ {time_e:.2f}s, {end_ms - start_ms}ms) skipped.")
                continue

            segment = audio[start_ms:end_ms]
            buffer = BytesIO()
            segment.export(buffer, format="wav")
            buffer.seek(0)
            
            emb = model.extract_embedding(buffer)
            emb = self.prepare_embeddings(emb).squeeze()  # (D,) 형태로 정리
            emb_results.append(((time_s, time_e), speaker, emb))
        return emb_results
    
    def get_embeddings_from_feask(self, model, file_name, diar_result, chunk_offset=0):
        audio = AudioSegment.from_wav(file_name)
        emb_results = []
        fbanks = []
        segments_meta = []

        for (time_s, time_e), speaker in diar_result:
            if speaker == 'filler':
                continue
            start_ms = int(time_s + chunk_offset) * 1000
            end_ms = int(time_e + chunk_offset) * 1000
            if (end_ms - start_ms) < 1000:
                continue

            # 오디오 segment → numpy array
            segment = audio[start_ms:end_ms]
            samples = segment.get_array_of_samples()
            pcm = torch.tensor(samples, dtype=torch.float32) / 32768.0  # normalize
            if segment.channels > 1:
                pcm = pcm.reshape(-1, segment.channels).mean(dim=1)  # mono 변환
            pcm = pcm.unsqueeze(0)  # (1, T)
            if segment.frame_rate != model.resample_rate:
                pcm = Resample(orig_freq=segment.frame_rate,
                            new_freq=model.resample_rate)(pcm)

            # fbank 추출 및 저장
            fbank = model.compute_fbank(pcm, sample_rate=model.resample_rate, cmn=False)
            fbanks.append(fbank)
            segments_meta.append(((time_s, time_e), speaker))

        max_len = max(f.shape[0] for f in fbanks)
        fbanks_padded = [F.pad(f, (0, 0, 0, max_len - f.shape[0])) for f in fbanks]  # pad time axis

        emb_array = model.extract_embedding_feats(
            fbanks_padded,
            batch_size=model.diar_batch_size,
            subseg_cmn=False   # CMN에 패딩 포함 방지
        )
        for ((time_s, time_e), speaker), emb in zip(segments_meta, emb_array):
            emb = emb.reshape(-1)
            emb_results.append(((time_s, time_e), speaker, emb))
        return emb_results

class SBEMB(BaseEMB):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def set_classifier(self):
        self.classifier = EncoderClassifier.from_hparams(
            source=self.config['model_name'],
            run_opts={"device":self.device}
        )

    def set_srmodel(self):
        self.srmodel = SpeakerRecognition.from_hparams(
            source=self.config['model_name'], 
            savedir=self.config['model_path'], 
            run_opts={"device":self.device}
        )
                                            
    def get_embedding(self, classifier, file_name):
        if isinstance(file_name, AudioSegment):
            buffer = BytesIO()
            file_name.export(buffer, format="wav")
            buffer.seek(0)    # 반드시 처음으로 포인터 옮기기
            signal, fs = torchaudio.load(buffer)
        else:
            signal, fs = torchaudio.load(file_name)
        embeddings = classifier.encode_batch(signal)
        return embeddings
    
    def get_embeddings(self, classifier, file_path, file_list):
        emb_list = []
        for file_name in file_list: 
            audio_file = os.path.join(file_path, file_name)
            if isinstance(audio_file, AudioSegment):
                buffer = BytesIO()
                audio_file.export(buffer, format='wav')
                buffer.seek(0)
                signal, fs = torchaudio.load(buffer) 
            else:
                signal, fs = torchaudio.load(audio_file)
            emb = classifier.encode_batch(signal)
            emb_list.append(self.prepare_embeddings(emb))
        return emb_list 

    def calc_emb_similarity(self, srmodel, audio_file1, audio_file2):
        score, pred = srmodel.verify_files(audio_file1, audio_file2)
        return score, pred 


class EMBVisualizer(BaseEMB):
    def __init__(self):
        super().__init__()

    def tsne_and_plot(self, embeddings, labels, title, file_path=None):
        """
        embeddings: numpy array [N, D]
        labels: list of str (len=N)
        title: plot title
        """
        tsne = TSNE(n_components=2, metric='cosine', perplexity=5, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)

        plt.figure(figsize=(8,6))
        unique_labels = list(set(labels))
        colors = plt.colormaps.get_cmap('tab10')
        for idx, label in enumerate(unique_labels):
            indices = [i for i, l in enumerate(labels) if l == label]
            plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=label, color=colors(idx))

        plt.title(title)
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend()
        plt.grid()
        if file_path != None: 
            plt.savefig(f"{os.path.join(file_path, title.replace(' ', '_'))}.png")
            print(f"Saved plot to {title.replace(' ', '_')}.png")

    def pca_and_plot(self, embeddings, labels, title, file_path=None):
        """
        embeddings: numpy array [N, D]
        labels: list of str (len=N)
        title: plot title
        """
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)

        plt.figure(figsize=(8,6))
        unique_labels = list(set(labels))
        colors = plt.colormaps.get_cmap('tab10')
        for idx, label in enumerate(unique_labels):
            indices = [i for i, l in enumerate(labels) if l == label]
            plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=label, color=colors(idx))

        plt.title(title)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        plt.grid()
        if file_path != None: 
            plt.savefig(f"{os.path.join(file_path, title.replace(' ', '_'))}.png")
            print(f"Saved plot to {title.replace(' ', '_')}.png")

    def plot_similarity_heatmap(self, sim_matrix, speakers, title, cmap='Blues', file_path=None):
        plt.figure(figsize=(6,5))
        sns.heatmap(sim_matrix, xticklabels=speakers, yticklabels=speakers, cmap=cmap, annot=True, fmt=".2f", square=True, cbar=True, vmin=0, vmax=1)
        plt.title(title)
        plt.tight_layout()
        if file_path != None:
            plt.savefig(f"{os.path.join(file_path, title.replace(' ', '_'))}.png")
            print(f"Saved heatmap: {title.replace(' ', '_')}.png")