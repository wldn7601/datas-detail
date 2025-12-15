# AI 영화 추천 모델 데이터 저장소

카카오클라우드에서 학습한 모델 데이터 및 전처리 결과물을 로컬로 복사한 저장소입니다.

## 📦 데이터 다운로드

**Google Drive 공유 폴더**: [AI 모델 데이터 다운로드](https://drive.google.com/drive/folders/1RIEx7ExMuJ3Vx-yg_8mJnUgY1HazYETj)

---

## 📁 디렉토리 구조

```
models-data/
├── originam-data/          # 원본 데이터 (전처리 전)
├── sbert-data/             # CBF 모델 데이터 (Content-Based Filtering v2)
├── sbert-index/            # CBF 모델 인덱스 (FAISS)
├── lightgcn-data/          # LightGCN 모델 학습 데이터
└── lightgcn-checkpoints/   # LightGCN 모델 체크포인트
```

---

## 📂 1. originam-data/ (원본 데이터)

전처리 전 원본 데이터 파일들입니다.

### 파일 목록

| 파일명                       | 설명                                                         |
| ---------------------------- | ------------------------------------------------------------ |
| `final_movies_processed.pkl` | 최종 전처리된 영화 메타데이터 (태그, OTT, 장르 등 포함)      |
| `ratings.csv`                | 사용자-영화 평점 데이터 (userId, movieId, rating, timestamp) |
| `tagdl.csv`                  | Tag Genome 데이터 (영화별 태그 점수)                         |
| `tmdb_ott_raw.csv`           | TMDB OTT 제공 정보 원본 데이터                               |

### 데이터 출처

- **ratings.csv**: MovieLens 데이터셋
- **tagdl.csv**: Tag Genome 데이터
- **tmdb_ott_raw.csv**: TMDB API를 통해 수집한 OTT 제공 정보
- **final_movies_processed.pkl**: 위 데이터들을 통합/전처리한 최종 결과물

---

## 📂 2. sbert-data/ (CBF 모델 데이터)

Content-Based Filtering v2 모델의 데이터 파일들입니다.
Sentence-BERT (`multilingual-e5-large`) 임베딩을 사용합니다.

### 파일 목록

| 파일명                           | 설명                                                       | 생성 스크립트          |
| -------------------------------- | ---------------------------------------------------------- | ---------------------- |
| `pre_final_movies_processed.pkl` | 임베딩 생성 전 정제 데이터 (결측치 제거, 불필요 컬럼 삭제) | `run_preprocess.py`    |
| `pre_final_movies_processed.csv` | 위 pkl 파일의 CSV 버전 (확인용)                            | `run_preprocess.py`    |
| `movies_with_embeddings.pkl`     | 영화 메타데이터 + multilingual-e5-large 임베딩 벡터        | `create_embeddings.py` |
| `movies_embeddings_whitened.pkl` | Whitening 기법 적용 임베딩 (구조적 편향 제거, 사용 안 함)  | `run_whitening.py`     |

### 데이터 전처리 과정

1. **run_preprocess.py**

   - `final_movies_processed.pkl`에서 시작
   - 결측치 제거
   - `text_input`, `embedding` 컬럼 삭제 (재생성 위해)

2. **create_embeddings.py**

   - Sentence-BERT 모델: `intfloat/multilingual-e5-large`
   - 입력 형식:
     - 태그 있음: `"tags: {태그}. {overview}"`
     - 태그 없음: `"{overview}"`
     - overview 10자 미만: `"title: {제목}. tags: {태그}. {overview}"`
   - FP16 최적화 적용 (GPU 학습 속도 향상)
   - 정규화된 임베딩 벡터 생성 (차원: 1024)

3. **run_whitening.py** (선택적, 미사용)
   - Vector Space Collapse 문제 해결용
   - Zero-centering + Decorrelation
   - 최종 모델에서는 사용하지 않음

---

## 📂 3. sbert-index/ (FAISS 인덱스)

빠른 유사도 검색을 위한 FAISS 인덱스 파일들입니다.

### 파일 목록

| 파일명          | 설명                                                 | 생성 스크립트     |
| --------------- | ---------------------------------------------------- | ----------------- |
| `movies.faiss`  | FAISS IndexFlatIP 인덱스 (내적 기반 유사도 검색)     | `create_index.py` |
| `movie_ids.pkl` | 인덱스 순서에 매핑되는 movieId 리스트 (MovieLens ID) | `create_index.py` |

### 인덱스 생성 과정

- **create_index.py**
  - `movies_with_embeddings.pkl`에서 임베딩 추출
  - movieId 컬럼 존재 여부 검증
  - FAISS `IndexFlatIP` (Inner Product) 사용
  - GPU 사용 가능 시 GPU 인덱싱 후 CPU로 변환
  - **movieId** 매핑 정보 저장 (LightGCN과 동일한 ID 체계 사용)

---

## 📂 4. lightgcn-data/ (LightGCN 학습 데이터)

협업 필터링 모델인 LightGCN의 학습/평가 데이터입니다.

### 파일 목록

| 파일명               | 설명                                                     |
| -------------------- | -------------------------------------------------------- |
| `train_ratings.csv`  | Train 분할 평점 데이터 (원본 형식)                       |
| `test_ratings.csv`   | Test 분할 평점 데이터 (원본 형식)                        |
| `train_implicit.csv` | Implicit Feedback 변환 Train 데이터                      |
| `test_implicit.csv`  | Implicit Feedback 변환 Test 데이터                       |
| `train_remapped.csv` | ID 재매핑된 Train 데이터 (user_idx, item_idx 추가)       |
| `test_remapped.csv`  | ID 재매핑된 Test 데이터 (user_idx, item_idx 추가)        |
| `train_matrix.npz`   | Train Interaction Matrix (Sparse CSR 형식)               |
| `test_matrix.npz`    | Test Interaction Matrix (Sparse CSR 형식)                |
| `edge_index.pt`      | PyTorch Geometric Graph Edge Index (양방향)              |
| `id_mappings.pkl`    | User/Item ID 매핑 정보 (user2id, item2id 등)             |
| `metadata.pkl`       | 데이터셋 메타정보 (사용자 수, 아이템 수, 상호작용 수 등) |

### 데이터 생성 과정

**run_split_ratings.py**

1. **데이터 분할** (Random Split 8:2)

   - `ratings.csv` 로드
   - 최소 5개 이상 평점 남긴 사용자만 필터링
   - Stratified Split (유저별 비율 유지)

2. **Implicit Feedback 변환**

   - 현재 Threshold: `None` (모든 평점을 positive로 사용)
   - 설정 가능: `THRESHOLD = 3.5` 등

3. **ID 재매핑**

   - User ID: 0 ~ n_users-1
   - Item ID: 0 ~ n_items-1
   - Train/Test 모두 동일한 매핑 사용

4. **Sparse Matrix 생성**

   - User-Item Interaction Matrix (CSR 형식)
   - 학습 효율성 향상

5. **Graph 구조 생성**
   - User-Item 양방향 그래프 (Bipartite Graph)
   - Edge Index: [2, num_edges] 형태
   - PyTorch Geometric 호환

---

## 📂 5. lightgcn-checkpoints/ (LightGCN 체크포인트)

LightGCN 모델 학습 결과물입니다.

### 파일 목록

| 파일명                   | 설명                                     |
| ------------------------ | ---------------------------------------- |
| `best_model.pt`          | 검증 손실이 가장 낮은 최고 성능 모델     |
| `final_model.pt`         | 마지막 epoch의 모델                      |
| `checkpoint_epoch_5.pt`  | Epoch 5 체크포인트                       |
| `checkpoint_epoch_10.pt` | Epoch 10 체크포인트                      |
| `checkpoint_epoch_15.pt` | Epoch 15 체크포인트                      |
| `checkpoint_epoch_20.pt` | Epoch 20 체크포인트                      |
| `training_history.pkl`   | 학습 히스토리 (loss, bpr_loss, reg_loss) |

### 모델 학습 설정

**하이퍼파라미터** (run_train_lightgcn.py)

```python
EMBEDDING_DIM = 256        # 임베딩 차원
N_LAYERS = 3               # LightGCN 레이어 수
BATCH_SIZE = 4096          # BPR 샘플링 배치 크기
LEARNING_RATE = 0.001      # 학습률
REG_WEIGHT = 1e-4          # L2 정규화 가중치
N_EPOCHS = 20              # 총 학습 에포크
```

**손실 함수**

- BPR Loss (Bayesian Personalized Ranking)
- L2 Regularization

### 체크포인트 내용

각 .pt 파일에는 다음 정보가 저장되어 있습니다:

```python
{
    'epoch': int,
    'model_state_dict': OrderedDict,
    'optimizer_state_dict': OrderedDict,
    'loss': float,
    'history': dict,  # (최종 모델만)
    'n_users': int,
    'n_items': int,
    'embedding_dim': int,
    'n_layers': int
}
```

---

## 📊 모델별 요약

### Content-Based Filtering (CBF v2)

- **모델**: Sentence-BERT (multilingual-e5-large)
- **입력**: 영화 태그 + 줄거리 + 제목
- **출력**: 1024차원 임베딩 벡터
- **검색**: FAISS IndexFlatIP (내적 기반)
- **용도**: 콘텐츠 기반 영화 유사도 검색

### LightGCN (Collaborative Filtering)

- **모델**: LightGCN (Graph Neural Network)
- **입력**: User-Item 상호작용 그래프
- **출력**: User/Item 임베딩 (256차원)
- **학습**: BPR Loss + L2 Regularization
- **용도**: 협업 필터링 기반 개인화 추천

---

## 🚀 사용 방법

### CBF 모델 로드 예시

```python
import pandas as pd
import faiss
import pickle

# 영화 데이터 로드
df = pd.read_pickle('sbert-data/movies_with_embeddings.pkl')

# FAISS 인덱스 로드
index = faiss.read_index('sbert-index/movies.faiss')
with open('sbert-index/movie_ids.pkl', 'rb') as f:
    movie_ids = pickle.load(f)

# 유사 영화 검색 (movieId 기준)
query_movie_id = 1  # 예: Toy Story
query_embedding = df[df['movieId'] == query_movie_id]['embedding'].values[0]
D, I = index.search(query_embedding.reshape(1, -1), k=10)
similar_movie_ids = [movie_ids[i] for i in I[0]]
```

### LightGCN 모델 로드 예시

```python
import torch
import pickle

# 체크포인트 로드
checkpoint = torch.load('lightgcn-checkpoints/best_model.pt')

# 메타데이터 로드
with open('lightgcn-data/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

# ID 매핑 로드
with open('lightgcn-data/id_mappings.pkl', 'rb') as f:
    mappings = pickle.load(f)

# 모델 재구성 및 추론
# (run_evaluate_lightgcn.py 참고)
```

---

## 📝 참고사항

- **originam-data**: 모든 모델의 기본 데이터 소스
- **sbert-data**: CBF v2 전용 (whitened 버전은 미사용)
- **lightgcn-data**: LightGCN 전용 (Random Split 8:2)
- **Threshold**: LightGCN은 현재 모든 평점을 positive로 사용 (Threshold=None)

### ID 체계 통일

- **CBF와 LightGCN 모두 `movieId` (MovieLens ID) 사용**
- `final_movies_processed.pkl`에는 `movieId`와 `tmdb_id` 모두 포함
- FAISS 인덱스 (`movie_ids.pkl`)도 `movieId`로 매핑하여 일관성 유지
- 추천 시스템에서 두 모델 간 영화 ID 변환 없이 직접 통합 가능

---

**작성일**: 2025-12-15
**서버**: 카카오클라우드 (210.109.82.91)
**프로젝트**: AI 영화 추천 시스템
