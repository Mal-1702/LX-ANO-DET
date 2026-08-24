# LINUX AUTHENTICATION LOG ANOMALY DETECTION SYSTEM
## COMPREHENSIVE ARCHITECTURAL CONTEXT & CODEBASE SPECIFICATION
*Document Version: 1.0.0 | Production-Grade Machine Learning & Security Operations System*

---

# SECTION 1: SYSTEM OVERVIEW & CORE OBJECTIVE

### 1.1 Project Purpose
The **Linux Authentication Log Anomaly Detection System** is an enterprise-grade, end-to-end Machine Learning and Security Operations (SecOps) pipeline designed to ingest, audit, preprocess, extract leakage-safe features from, classify, and explain security anomalies in Linux authentication logs (`/var/log/auth.log` / `secure`).

### 1.2 Target Threat Classes (5-Class Multi-Class Taxonomy)
The system classifies every authentication log entry into one of five strictly defined classes:
1. `normal`: Legitimate administrative and user operations, scheduled cron executions, valid SSH/system logins.
2. `brute_force`: Repeated, high-frequency failed password/key attempts across single or distributed source IPs targeting system accounts.
3. `port_scan`: Reconnaissance connection attempts across anomalous, randomized, non-standard, or sequential network ports.
4. `geo_anomaly`: Logins or connection attempts originating from anomalous source IPs, atypical network ranges, or unusual geographic time patterns.
5. `privilege_escalation`: Suspicious privilege elevation events, unauthorized `sudo` executions, `su` transitions, or abnormal session overrides.

### 1.3 Key Architectural Principles
- **Zero Data Leakage**: Feature maps (frequency encoding, label encodings) are fitted strictly on the chronological training split and evaluated on future validation/test sets.
- **Deterministic Preprocessing**: Gracefully handles missing values, novel categories, non-numeric ports, malformed timestamps, and numerical overflow without throwing runtime exceptions.
- **Multi-Model Benchmark & Automated Selection**: Trains Decision Trees, Random Forests, Gradient Boosting, and XGBoost with class-weight balancing and 3-fold Stratified CV parameter search, choosing the champion model based on validation/test **Macro-F1 score**.
- **Self-Contained Deployment**: Serves predictions via both a **FastAPI REST API** (asynchronous, CORS-enabled, Pydantic-validated) and a **Streamlit Web Dashboard** (interactive KPI cards, color-coded tabular viewer, dynamic explainability plots).
- **Explainable AI (XAI)**: Generates per-record feature importance attributions for anomaly triage.

---

# SECTION 2: COMPLETE FILE TREE & DIRECTORY INDEX

```text
c:\Users\Malha\OneDrive\Desktop\Linux-anomaly\
│
├── .gitignore                                      # VCS exclusion rules (ignores raw CSVs, large joblib artifacts, caches)
├── README.md                                       # High-level project summary and quickstart documentation
├── requirements.txt                                # Python dependency manifest with strict version boundaries
├── run.py                                          # Core monolithic pipeline (Phases 1-10: training, evaluation, inference API)
│
├── datasets/ (Ignored from git tracking via *.csv)
│   ├── linux_auth_logs_labeled.csv                 # Primary supervised dataset (~63.5 MB, with 'anomaly_label')
│   ├── linux_auth_logs_full(balanced).csv          # Auxiliary balanced dataset (~41.8 MB) for robustness testing
│   ├── linux_auth_logs_full(new_unbalanced).csv    # Auxiliary unbalanced dataset (~59.6 MB) for real-world drift evaluation
│   └── linux_auth_logs_multiple_anomalies.csv      # Unlabeled inference demonstration dataset (~39.9 MB)
│
├── tests/
│   └── test_pipeline.py                            # Production test suite (14 inference validation tests + 4 FastAPI endpoint tests)
│
└── outputs/                                        # Generated models, metadata, visualization charts, and serving interfaces
    ├── fastapi_app.py                              # Production FastAPI REST microservice (single/batch inference, XAI, health)
    ├── streamlit_app.py                            # Interactive Streamlit SecOps dashboard for analysts
    ├── model_meta.json                             # Champion model metadata, ordered feature column list, class index mappings
    ├── best_model.joblib                           # Serialized champion scikit-learn/XGBoost model binary
    ├── feature_maps.joblib                         # Serialized categorical & frequency mapping dictionaries
    ├── label_encoder.joblib                        # Serialized scikit-learn LabelEncoder object for target classes
    ├── inference_predictions.csv                   # Output predictions from batch evaluation on demo dataset
    │
    ├── eda_overview.png                            # 9-panel diagnostic exploratory data analysis visualization
    ├── correlation_heatmap.png                     # Pearson correlation heatmap of numerical features
    │
    ├── cm_DecisionTree_val.png                     # Confusion Matrix: Decision Tree (Validation Split)
    ├── cm_DecisionTree_test.png                    # Confusion Matrix: Decision Tree (Test Split)
    ├── cm_RandomForest_val.png                     # Confusion Matrix: Random Forest (Validation Split)
    ├── cm_RandomForest_test.png                    # Confusion Matrix: Random Forest (Test Split)
    ├── cm_GradientBoosting_val.png                 # Confusion Matrix: Gradient Boosting (Validation Split)
    ├── cm_GradientBoosting_test.png                # Confusion Matrix: Gradient Boosting (Test Split)
    ├── cm_XGBoost_val.png                          # Confusion Matrix: XGBoost (Validation Split)
    ├── cm_XGBoost_test.png                         # Confusion Matrix: XGBoost (Test Split)
    ├── cm_DecisionTreeClassifier_balanced.png      # Confusion Matrix: Best Model on Auxiliary Balanced Dataset
    ├── cm_DecisionTreeClassifier_unbalanced.png    # Confusion Matrix: Best Model on Auxiliary Unbalanced Dataset
    ├── cm_RandomForestClassifier_balanced.png      # Confusion Matrix: Alternate Model on Auxiliary Balanced Dataset
    ├── cm_RandomForestClassifier_unbalanced.png    # Confusion Matrix: Alternate Model on Auxiliary Unbalanced Dataset
    │
    ├── feature_importance_DecisionTree.png         # Bar chart: Gini/Entropy feature importances for Decision Tree
    ├── feature_importance_RandomForest.png         # Bar chart: Gini impurity feature importances for Random Forest
    ├── feature_importance_GradientBoosting.png     # Bar chart: Tree split gain feature importances for Gradient Boosting
    └── feature_importance_XGBoost.png              # Bar chart: Gain/Weight feature importances for XGBoost
```

---

# SECTION 3: DEPENDENCY MANIFEST (`requirements.txt`)

```text
scikit-learn>=1.4.0   # Decision trees, ensemble models, cross-validation, metrics, label encoding
pandas>=2.0.0         # DataFrame manipulation, CSV ingestion, timestamp vectorized operations
numpy>=1.24.0         # Numerical arrays, matrix operations, nan/inf cleaning, clipping
matplotlib>=3.7.0     # Static figure generation, non-interactive 'Agg' backend rendering
seaborn>=0.13.0       # Statistical visualization, heatmaps, color palettes
joblib>=1.3.0         # Model persistence and serialized dictionary storage
fastapi>=0.110.0      # REST API ASGI framework
uvicorn>=0.28.0       # Production ASGI server implementation
pydantic>=2.0.0       # Strict request/response validation schemas
streamlit>=1.30.0     # Reactive SecOps analyst web dashboard
xgboost>=2.0.0        # Extreme Gradient Boosting classifier with histogram tree method
pytest>=8.0.0         # Automated unit and integration test runner
httpx>=0.27.0         # ASGI HTTP client for FastAPI TestClient integration
```

---

# SECTION 4: DATASET SCHEMAS & ATTRIBUTES

### 4.1 Raw Ingestion Schema (Input Columns)
Every raw input CSV record contains the following security log attributes:
- `timestamp` *(string / ISO-8601)*: Event occurrence time (e.g. `"2024-03-28T19:34:14.984218"`).
- `source_ip` *(string)*: Originating IPv4 address (e.g. `"195.241.151.7"`, `"186.144.249.195"`).
- `server` *(string)*: Hostname of target host/server (e.g. `"srv-tok-03"`, `"proxy-mow-01"`).
- `username` *(string)*: Account identifier attempting authentication (e.g. `"root"`, `"admin"`, `"juancampos"`).
- `service` *(string)*: Authentication daemon/service (e.g. `"ssh"`, `"sshd"`, `"sudo"`, `"su"`, `"login"`, `"cron"`).
- `attempts` *(integer / float / string)*: Consecutive authentication attempt count.
- `status` *(string)*: Authentication outcome (`"Success"`, `"Failed"`).
- `port` *(integer / float / string)*: Destination connection port (e.g. `22`, `443`, `8080`, `"N/A"`).
- `protocol` *(string)*: Transport/application protocol (e.g. `"SSH"`, `"SSH2"`, `"RDP"`, `"TELNET"`).
- `comment` *(string)*: Free-text syslog message payload (e.g. `"Failed password for invalid user admin from 195.241.151.7 port 22 ssh2"`).
- `anomaly_label` *(string, Target Column)*: Ground-truth target class (present in training data).

---

# SECTION 5: MACHINE LEARNING PIPELINE ARCHITECTURE (`run.py`)

`run.py` is the central orchestration engine. It is organized into 10 structured phases:

```
[Phase 1: Load & Audit]
        │
        ▼
[Phase 2: EDA & Heatmaps]
        │
        ▼
[Phase 4: Chronological Split (70/15/15)]   <-- Prevents temporal lookahead leakage
        │
        ▼
[Phase 3: Fit Feature Maps on Train & Transform (Train, Val, Test)]
        │
        ▼
[Phase 5: Subsample Tuning & Model Training (DT, RF, GB, XGBoost)]
        │
        ▼
[Phase 6: Unified Multi-Metric Evaluation (Val & Test)]
        │
        ▼
[Phase 7: Model Selection (Champion on Test Macro-F1)]
        │
        ▼
[Phase 8: Feature Importance & Aux Dataset Robustness Check]
        │
        ▼
[Phase 9: Artifact Serialization (best_model, maps, encoder, meta)]
        │
        ▼
[Phase 10: Production Standalone Inference Verification]
```

### 5.1 Constant & Configuration Definitions
- `SEED = 42`: Fixed random seed for NumPy, scikit-learn estimators, and StratifiedKFold.
- `PROJECT_ROOT`: Resolved absolute path to the repository directory.
- `OUTPUT_DIR`: Path to `<PROJECT_ROOT>/outputs` (created automatically if missing).
- `TARGET = "anomaly_label"`: Name of the classification target column.
- `PROTO_MAP = {"SSH": 0, "SSH2": 0, "RDP": 1, "TELNET": 2, "unknown": 3}`: Static protocol numerical mapping.
- `FEATURE_COLS`: The exact 16-element ordered list of engineered input features:
  ```python
  FEATURE_COLS = [
      "attempts",          # Numeric: Login attempts count >= 1.0
      "log_port",          # Numeric: Natural log log1p(clipped_port)
      "hour",              # Temporal: Hour of day (0-23)
      "dow",               # Temporal: Day of week (0=Monday, 6=Sunday)
      "month",             # Temporal: Month of year (1-12)
      "day",               # Temporal: Day of month (1-31)
      "is_weekend",        # Temporal: Binary flag (1 if dow >= 5 else 0)
      "is_night",          # Temporal: Binary flag (1 if hour < 6 or hour >= 22 else 0)
      "protocol_enc",      # Categorical: Protocol mapped via PROTO_MAP
      "server_enc",        # Categorical: Train-fitted LabelEncoder index (unseen -> -1)
      "service_enc",       # Categorical: Train-fitted LabelEncoder index (unseen -> -1)
      "status_enc",        # Categorical: Train-fitted LabelEncoder index (unseen -> -1)
      "source_ip_freq",    # Frequency: Normalized frequency of source_ip in train (unseen -> 0.0)
      "username_freq",     # Frequency: Normalized frequency of username in train (unseen -> 0.0)
      "comment_failed",    # Text Flag: Binary indicator for presence of 'failed|Failed' in comment
      "comment_accepted",  # Text Flag: Binary indicator for presence of 'accepted|Accepted' in comment
  ]
  ```

---

### 5.2 Detailed Phase-by-Phase Technical Breakdown

#### PHASE 1 — Data Ingestion & Quality Auditing (`load_and_audit`)
- **Function**: `load_and_audit(path, label: str = "dataset") -> pd.DataFrame`
- **Logic**:
  1. Ingests raw CSV via `pd.read_csv(path)`.
  2. Prints audit banner: Shape, Column names, Duplicate count (`df.duplicated().sum()`), Data types (`df.dtypes`), and Null count per column (`df.isnull().sum()`).
  3. Audits class counts and percentage proportions if target column exists.
  4. Parses timestamps to determine temporal span `[ts.min(), ts.max()]` and invalid timestamp null counts.
  5. Computes cardinality metrics: splits columns into numeric, low-cardinality string (`<=20` unique values), and high-cardinality string (`>20` unique values).
  6. Drops duplicates via `df.drop_duplicates().reset_index(drop=True)` to prevent memory waste and repeated rows.

#### PHASE 2 — Exploratory Data Analysis (`run_eda`)
- **Function**: `run_eda(df: pd.DataFrame)`
- **Logic**:
  1. Generates a 9-subplot diagnostic figure saved to `outputs/eda_overview.png` (`figsize=(18, 14)`, `dpi=120`):
     - `[0, 0]` Class distribution bar chart with exact count annotations.
     - `[0, 1]` Attempts distribution histogram per target class (density normalized).
     - `[0, 2]` Top 20 most frequent network connection ports.
     - `[1, 0]` Temporal anomaly frequency curve by hour of day (0-23).
     - `[1, 1]` Anomaly frequency distribution by day of week (0=Mon ... 6=Sun).
     - `[1, 2]` Stacked bar chart of anomaly counts grouped by auth service.
     - `[2, 0]` Grouped bar chart of anomaly distribution by authentication status (`Success` vs `Failed`).
     - `[2, 1]` Missingness rate of protocol attribute grouped by class.
     - `[2, 2]` Scatter plot: `attempts` vs `log1p(port)` for a sampled 5,000 records.
  2. Computes Pearson correlation matrix on numeric features (`attempts`, `port`, `_hour`, `_dow`, `_month`) and outputs `outputs/correlation_heatmap.png`.

#### PHASE 4 — Chronological Partitioning (`chronological_split`)
*(Executed before Phase 3 fitting to guarantee strict temporal ordering without lookahead contamination)*
- **Function**: `chronological_split(df: pd.DataFrame, train_frac=0.70, val_frac=0.15)`
- **Logic**:
  1. Sorts entire DataFrame strictly by `timestamp` column: `df.sort_values("timestamp").reset_index(drop=True)`.
  2. Partitions into:
     - **Train Split (70%)**: Index `0` to `int(N * 0.70)`.
     - **Validation Split (15%)**: Index `int(N * 0.70)` to `int(N * 0.85)`.
     - **Test Split (15%)**: Index `int(N * 0.85)` to `N`.
  3. Outputs exact sample sizes and class distributions for each split.

#### PHASE 3 — Leakage-Safe Feature Engineering (`fit_feature_maps`, `transform_features`, `get_X`)
- **Functions**:
  1. `fit_feature_maps(df_train: pd.DataFrame) -> dict`:
     - Fits standard `LabelEncoder` on `server`, `service`, `status` using `df_train` only.
     - Computes normalized probability distribution maps (`value_counts(normalize=True).to_dict()`) for `source_ip` and `username`.
     - Returns dictionary of fitted mappings: `{col_le: {...}, col_freq: {...}}`.
  2. `transform_features(df: pd.DataFrame, maps: dict) -> pd.DataFrame`:
     - **Attempts**: Coerces non-numeric to NaN, fills missing with `1.0`, applies lower bound `np.maximum(attempts, 1.0)`.
     - **Port**: Coerces non-numeric to `0.0`, clips to valid network range `[0.0, 65535.0]`, computes `log_port = np.log1p(port)`.
     - **Temporal**: Parses `timestamp` into `hour` (0-23), `dow` (0-6), `month` (1-12), `day` (1-31), `is_weekend` (1 if dow >= 5), `is_night` (1 if hour < 6 or hour >= 22). Missing timestamps default safely to 0/1.
     - **Protocol**: Maps via static `PROTO_MAP` (SSH/SSH2 -> 0, RDP -> 1, TELNET -> 2, unknown -> 3).
     - **Categorical Encodings**: Maps `server`, `service`, `status` against train dictionary; unobserved novel categories at inference safely map to `-1`.
     - **Frequency Encodings**: Maps `source_ip` and `username` against train frequency dictionary; novel/cold-start IPs or usernames map to `0.0`.
     - **Comment Indicators**: Regex checks for `"failed|Failed"` -> `comment_failed` (1/0) and `"accepted|Accepted"` -> `comment_accepted` (1/0).
  3. `get_X(df_transformed: pd.DataFrame) -> np.ndarray`:
     - Extracts columns strictly in `FEATURE_COLS` order.
     - Converts to `np.float32`.
     - Sanitizes array via `np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)`.

#### PHASE 5 — Class Imbalance Handling & Multi-Model Training
- **Class Balancing Logic**:
  - Computes exact balanced inverse-frequency weights:
    $$\text{weight}_c = \frac{N}{|C| \times N_c}$$
  - Implemented in `build_class_weight_dicts(y_train, classes)` -> returns `cw_int` (integer-keyed for scikit-learn models) and `cw_str` (class-name-keyed for sample weight arrays).
- **Subsampling Utility**:
  - `make_sub_sample(X_train, y_train, size=20_000)`: Extracts a stratified, class-proportional subset for rapid hyperparameter exploration without bias.
- **Hyperparameter Optimization Engine**:
  - `quick_param_search(estimator, param_grid, X_sub, y_sub, n_iter=5)`: Evaluates candidate parameter sets using 3-fold `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)` scored on **Macro-F1** (`scoring="f1_macro"`).
- **Trained Estimators**:
  1. `train_decision_tree`:
     - Grid: `max_depth: [10, 15, 20, 25]`, `min_samples_split: [2, 5, 10]`, `min_samples_leaf: [1, 2, 4]`, `criterion: ["gini", "entropy"]`.
     - Fits full `X_train` with `class_weight=cw_int`.
  2. `train_random_forest`:
     - Grid: `n_estimators: [100, 150, 200]`, `max_depth: [12, 16, 20]`, `min_samples_split: [2, 5]`, `min_samples_leaf: [1, 2]`, `max_features: ["sqrt", "log2"]`.
     - Fits full `X_train` with `class_weight=cw_int`, `n_jobs=-1`.
  3. `train_gradient_boosting`:
     - Grid: `n_estimators: [80, 120]`, `max_depth: [3, 5]`, `learning_rate: [0.05, 0.1]`, `subsample: [0.8, 1.0]`.
     - Trained on 100k stratified subsample with per-row `sample_weight=sw_gb`.
  4. `train_xgboost_model`:
     - Grid: `n_estimators: [100, 150]`, `max_depth: [4, 6]`, `learning_rate: [0.05, 0.1]`, `subsample: [0.8, 1.0]`, `colsample_bytree: [0.8, 1.0]`.
     - Configured with `eval_metric="mlogloss"`, `tree_method="hist"`, `n_jobs=-1`, fitted with `sample_weight=sw_train`.

#### PHASE 6 — Unified Evaluation Framework (`evaluate_model`, `evaluate_all`)
- **Evaluation Metrics Computed**:
  - Overall Accuracy (`accuracy_score`).
  - **Macro-Averaged F1 Score** (`f1_score(..., average="macro")`): Primary benchmark ensuring minority attack classes are heavily weighted.
  - **Weighted-Averaged F1 Score** (`f1_score(..., average="weighted")`).
  - Full Precision, Recall, F1-Score breakdown per class via `classification_report(digits=4)`.
  - **Multi-Class One-vs-Rest ROC-AUC** (`roc_auc_score(..., multi_class="ovr", average="macro")`) using predicted probability matrices.
  - Confusion Matrix rendered as seaborn heatmap saved to `outputs/cm_{model}_{split}.png`.

#### PHASE 7 — Champion Model Selection (`select_best_model`)
- **Logic**:
  - Collects test set evaluation results across all trained models.
  - Sorts models descending by `macro_f1`.
  - Automatically designates the rank-1 model as champion (`best_name`, `best_model`).
  - Displays summary leaderboard and computes performance gap over runner-up.

#### PHASE 8 — Explainability & Auxiliary Robustness Checks
- **Feature Importance**:
  - `show_feature_importance(model, model_name)`: Extracts `.feature_importances_`, ranks all 16 features, logs top 10, and generates bar chart `outputs/feature_importance_{model_name}.png`.
- **Generalization / Robustness Checks (`robustness_check`)**:
  - Evaluates champion model on out-of-distribution auxiliary datasets:
    1. `linux_auth_logs_full(balanced).csv`: Assesses performance under balanced prior class distributions.
    2. `linux_auth_logs_full(new_unbalanced).csv`: Assesses performance under heavy real-world class skew.

#### PHASE 9 & 10 — Model Persistence & Standalone Inference API
- **Persistence (`save_artifacts`)**:
  - `outputs/best_model.joblib`: Serialized champion model.
  - `outputs/feature_maps.joblib`: Serialized categorical/frequency lookup dictionary.
  - `outputs/label_encoder.joblib`: Serialized target LabelEncoder.
  - `outputs/model_meta.json`: Metadata JSON containing model name, feature column ordering, class lists, and class-to-index mappings.
- **Inference Functions**:
  1. `load_artifacts(output_dir: Path = None)`: Standalone loader that resolves paths independently of current working directory.
  2. `predict_single(row: dict, ...) -> dict`: Evaluates single raw log dictionary. Returns:
     ```python
     {
         "label": "brute_force",
         "probability": 0.9842,
         "is_anomaly": True,
         "all_probs": {"brute_force": 0.9842, "normal": 0.0102, ...}
     }
     ```
  3. `predict_batch(csv_or_df, output_path: str = None, ...) -> pd.DataFrame`: High-throughput vectorized inference over entire CSV files or DataFrames. Appends `predicted_label`, `is_anomaly`, and per-class probability columns (`prob_{class}`).
  4. `get_feature_contributions(row: dict, ...) -> dict`: Explainability calculation mapping feature values to model importances for security triage.

---

# SECTION 6: FASTAPI REST API SERVICE (`outputs/fastapi_app.py`)

### 6.1 Server Startup & Architecture
- **Title**: `Linux Auth Log Anomaly Detection API` (v1.0.0).
- **Run Command**: `uvicorn outputs.fastapi_app:app --host 0.0.0.0 --port 8000 --reload`
- **Path Resolution**: Automatically injects `PROJECT_ROOT` into `sys.path` to ensure importability regardless of invocation directory.
- **Middleware**: Configured with `CORSMiddleware(allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])`.
- **Startup Artifact Loading**: Calls `load_artifacts()` at module load; sets `MODEL_LOADED = True` or gracefully handles missing models.

### 6.2 Pydantic Validation Schemas
```python
class LogRecord(BaseModel):
    timestamp: Optional[str] = Field(None, example="2024-03-28T19:34:14")
    source_ip: Optional[str] = Field(None, example="195.241.151.7")
    server: Optional[str]    = Field(None, example="srv-tok-03")
    username: Optional[str]  = Field(None, example="admin")
    service: Optional[str]   = Field(None, example="sshd")
    attempts: Optional[float]= Field(1.0, ge=1.0, example=3)
    status: Optional[str]    = Field("Failed", example="Failed")
    port: Optional[float]    = Field(22.0, ge=0.0, le=65535.0, example=22)
    protocol: Optional[str]  = Field("SSH", example="SSH")
    comment: Optional[str]   = Field("", example="Failed password for invalid user admin")

class PredictionResponse(BaseModel):
    label: str
    is_anomaly: bool
    probability: Optional[float]
    all_probs: Dict[str, float]
```

### 6.3 Endpoints Specification
1. `GET /health`:
   - Returns service health status, model load state, champion model name, recognized classes, and feature column list.
2. `GET /classes`:
   - Returns `{"classes": ["brute_force", "geo_anomaly", "normal", "port_scan", "privilege_escalation"]}`.
3. `POST /predict/single`:
   - Ingests single JSON `LogRecord`.
   - Returns structured `PredictionResponse`.
4. `POST /predict/batch`:
   - Ingests multipart CSV file upload (`UploadFile`).
   - Writes to secure temporary file, runs `predict_batch()`, and deletes temporary file.
   - Returns summary metrics (`total_records`, `anomaly_records`, `anomaly_rate_percent`, `summary_by_class`) and top 100 sample prediction records.
5. `POST /explain`:
   - Ingests `LogRecord`.
   - Returns sorted dictionary of feature importances and extracted feature values for analyst inspection.

---

# SECTION 7: STREAMLIT ANALYST DASHBOARD (`outputs/streamlit_app.py`)

### 7.1 Interface Overview
- **Run Command**: `streamlit run outputs/streamlit_app.py`
- **Page Config**: Title `"Linux Auth Anomaly Detector"`, Wide Layout (`layout="wide"`), Favicon `"🛡️"`.
- **Model Caching**: Loads artifacts once using `@st.cache_resource def get_model()`.

### 7.2 Dashboard Layout & Features
1. **Sidebar**:
   - Displays Model Info badge (e.g. `DecisionTree` / `RandomForest`), target class taxonomy list, and CSV upload instructions.
2. **File Uploader**:
   - Ingests authentication log `.csv` files.
3. **KPI Metrics Cards (4 Columns)**:
   - `Total Records` (formatted with commas).
   - `Normal Events` count.
   - `Anomalies Detected` count with inverse delta percentage badge (`anomaly_rate%`).
   - `Model Selected` badge.
4. **Classification Breakdown**:
   - Interactive bar chart showing distribution across detected anomaly types.
   - Summary count & percentage table.
5. **Interactive Anomaly Inspection Table**:
   - Filter dropdown: `All Records`, `Anomalies Only`, or specific anomaly class filter.
   - Visual highlighting: Anomaly rows are dynamically highlighted with red/pink backgrounds (`#ffcccc`).
   - Download Button: Exports full enriched predictions CSV (`auth_anomaly_predictions.csv`).
6. **Explainable AI Drilldown**:
   - Interactive slider allowing security analysts to select any detected anomaly row.
   - Displays event context (`Target Class`, `User`, `Source IP`).
   - Renders horizontal bar chart of top 10 contributing feature importances for that specific event.

---

# SECTION 8: COMPREHENSIVE PRODUCTION TEST SUITE (`tests/test_pipeline.py`)

`pytest tests/ -v` executes 18 automated production tests:

### 8.1 14 Inference Edge-Case Validation Tests (`test_01` to `test_14`)
- `test_01_normal_valid_input`: Validates standard SSH login produces valid class, valid probability in `[0.0, 1.0]`, and complete class probability dictionary.
- `test_02_known_anomaly_brute_force`: Confirms high failed login attempts (e.g., 25 failed attempts) are properly processed.
- `test_03_missing_optional_fields`: Validates input with omitted `comment`, `protocol`, and `server` fields evaluates safely.
- `test_04_missing_required_fields`: Confirms completely empty payload `{}` defaults cleanly without crashing.
- `test_05_non_numeric_port`: Verifies string port values (`"N/A"`, `"http"`) are safely coerced to `0.0`.
- `test_06_invalid_port_range`: Validates negative ports (`-500`) and out-of-range ports (`999999`) are clipped to `[0.0, 65535.0]`.
- `test_07_missing_attempts`: Verifies omitted `attempts` key defaults to `1.0`.
- `test_08_invalid_attempts`: Confirms `0`, negative, and malformed attempts strings are bounded to `>= 1.0`.
- `test_09_unknown_source_ip_cold_start`: Validates completely novel IP addresses unobserved in training receive `source_ip_freq = 0.0`.
- `test_10_unknown_username_cold_start`: Validates novel usernames unobserved in training receive `username_freq = 0.0`.
- `test_11_extreme_numeric_values`: Validates huge numbers (`attempts: 1e9`, `port: 1e9`) do not create `NaN` or `Inf` in feature matrix `X`.
- `test_12_empty_batch`: Confirms empty DataFrame batch returns empty DataFrame with expected columns.
- `test_13_batch_mixed_valid_and_invalid`: Confirms batch with mixed valid and severely corrupted rows processes all records successfully.
- `test_14_explainability_consistency`: Validates feature contribution output contains all 16 features with valid `importance` and `value` fields.

### 8.2 4 FastAPI Integration Tests
- `test_api_health`: Confirms `GET /health` returns status code 200, `healthy`, and 5 recognized classes.
- `test_api_get_classes`: Confirms `GET /classes` returns exact set of 5 target classes.
- `test_api_predict_single`: Tests `POST /predict/single` with JSON payload.
- `test_api_predict_batch_csv`: Tests `POST /predict/batch` with uploaded multipart CSV content.

---

# SECTION 9: STEP-BY-STEP WORKFLOW & RUNBOOK

### 9.1 Environment Setup
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate    # Linux / macOS
.venv\Scripts\activate       # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
```

### 9.2 Train Full ML Pipeline & Generate All Artifacts
```bash
python run.py
```
*Outputs generated in `outputs/`: models (`.joblib`), metadata (`model_meta.json`), EDA plots, confusion matrices, and feature importance charts.*

### 9.3 Run Production Test Suite
```bash
pytest tests/ -v
```

### 9.4 Start FastAPI REST API Microservice
```bash
uvicorn outputs.fastapi_app:app --host 0.0.0.0 --port 8000 --reload
# Interactive Swagger Documentation accessible at: http://localhost:8000/docs
```

### 9.5 Start Streamlit Analyst Dashboard
```bash
streamlit run outputs/streamlit_app.py
# Dashboard accessible at: http://localhost:8501
```

---

# SECTION 10: SERIALIZED MODEL METADATA SCHEMA (`outputs/model_meta.json`)

```json
{
  "model_name": "DecisionTree",
  "feature_names": [
    "attempts",
    "log_port",
    "hour",
    "dow",
    "month",
    "day",
    "is_weekend",
    "is_night",
    "protocol_enc",
    "server_enc",
    "service_enc",
    "status_enc",
    "source_ip_freq",
    "username_freq",
    "comment_failed",
    "comment_accepted"
  ],
  "classes": [
    "brute_force",
    "geo_anomaly",
    "normal",
    "port_scan",
    "privilege_escalation"
  ],
  "class_to_index": {
    "brute_force": 0,
    "geo_anomaly": 1,
    "normal": 2,
    "port_scan": 3,
    "privilege_escalation": 4
  }
}
```

---
*End of Technical Specification Document.*
