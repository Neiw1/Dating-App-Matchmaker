# Cupid Matchmaker - System Architecture Diagram

## Overall System Flow

```mermaid
graph TB
    subgraph "Data Sources"
        A[Speed Dating Data.csv<br/>5MB Dataset]
        B[Kaggle Dataset<br/>annavictoria/speed-dating-experiment]
    end

    subgraph "Data Processing Layer"
        C[DataProcessor<br/>data_processor.py]
        D[pretrain_data.py<br/>Dataset Download]
    end

    subgraph "Model Architecture"
        E[UserAutoencoder<br/>75 → 8 dimensions]
        F[MatchingModel<br/>Collaborative Filtering]
    end

    subgraph "Training Pipeline"
        G[ModelTrainer<br/>training.py]
        H[main.py<br/>Orchestration]
    end

    subgraph "Web Server"
        I[Flask App<br/>server/main.py]
        J[run.py<br/>Entry Point]
    end

    subgraph "Original Implementation"
        K[_original.py<br/>Complete Working System]
    end

    %% Data Flow
    A --> C
    B --> D
    D --> A
    C --> H
    H --> G
    G --> E
    E --> F

    %% Web Server Flow
    J --> I
    I --> F

    %% Original System
    K -.-> E
    K -.-> F
    K -.-> C

    %% Styling
    classDef dataSource fill:#e1f5fe
    classDef processing fill:#f3e5f5
    classDef model fill:#e8f5e8
    classDef training fill:#fff3e0
    classDef server fill:#fce4ec
    classDef original fill:#f1f8e9

    class A,B dataSource
    class C,D processing
    class E,F model
    class G,H training
    class I,J server
    class K original
```

## Detailed Component Architecture

### 1. Data Processing Pipeline

```mermaid
graph LR
    subgraph "Raw Data"
        A1[CSV Data<br/>75+ Features]
        A2[User Profiles<br/>iid, pid, match]
    end

    subgraph "DataProcessor"
        B1[Hobbies Processing<br/>Sports, Art, Gaming, etc.]
        B2[Goals Mapping<br/>Casual, Long-term, etc.]
        B3[Values Processing<br/>Physical, Emotional, etc.]
        B4[Personality Traits<br/>Confident, Caring, etc.]
        B5[Faculty Mapping<br/>Chula Departments]
        B6[Gender Processing<br/>Preferences]
    end

    subgraph "Output"
        C1[Processed Features<br/>One-hot Encoded]
        C2[Normalized Data<br/>Ready for Training]
    end

    A1 --> B1
    A1 --> B2
    A1 --> B3
    A1 --> B4
    A1 --> B5
    A1 --> B6
    B1 --> C1
    B2 --> C1
    B3 --> C1
    B4 --> C1
    B5 --> C1
    B6 --> C1
    C1 --> C2
```

### 2. Neural Network Architecture

```mermaid
graph TB
    subgraph "UserAutoencoder"
        A1[Input Layer<br/>75 Dimensions]
        A2[Encoder<br/>75 → 32 → 16 → 8]
        A3[Latent Space<br/>8 Dimensions]
        A4[Decoder<br/>8 → 16 → 32 → 75]
        A5[Output Layer<br/>75 Dimensions]
    end

    subgraph "MatchingModel"
        B1[User Embeddings<br/>8 Dimensions]
        B2[Partner Embeddings<br/>8 Dimensions]
        B3[Cosine Similarity]
        B4[Compatibility Score<br/>0-1 Range]
    end

    A1 --> A2
    A2 --> A3
    A3 --> A4
    A4 --> A5
    A3 --> B1
    A3 --> B2
    B1 --> B3
    B2 --> B3
    B3 --> B4
```

### 3. Training Pipeline

```mermaid
graph LR
    subgraph "Stage 1: Autoencoder Training"
        A1[Processed Data<br/>75 Features]
        A2[UserAutoencoder]
        A3[MSE Loss]
        A4[Trained Encoder<br/>User Representations]
    end

    subgraph "Stage 2: Matching Training"
        B1[User Embeddings<br/>8 Dimensions]
        B2[Partner Embeddings<br/>8 Dimensions]
        B3[MatchingModel]
        B4[Match Predictions<br/>0-1 Scores]
        B5[Binary Cross-Entropy Loss]
    end

    A1 --> A2
    A2 --> A3
    A3 --> A4
    A4 --> B1
    A4 --> B2
    B1 --> B3
    B2 --> B3
    B3 --> B4
    B4 --> B5
```

### 4. File Structure & Dependencies

```mermaid
graph TD
    subgraph "Entry Points"
        A[run.py<br/>Main Entry Point]
        B[uv run run.py<br/>Command to Start]
    end

    subgraph "Server Layer"
        C[server/main.py<br/>Flask App]
        D[Flask Routes<br/>Web Endpoints]
    end

    subgraph "Model Layer"
        E[model/main.py<br/>Orchestration]
        F[model/architecture.py<br/>Neural Networks]
        G[model/data_processor.py<br/>Data Processing]
        H[model/training.py<br/>Training Logic]
        I[model/pretrain_data.py<br/>Data Download]
    end

    subgraph "Original Implementation"
        J[_original.py<br/>Complete System]
    end

    subgraph "Dependencies"
        K[torch<br/>PyTorch]
        L[flask<br/>Web Framework]
        M[numpy<br/>Numerical Computing]
        N[pandas<br/>Data Manipulation]
    end

    A --> C
    B --> A
    C --> E
    E --> F
    E --> G
    E --> H
    E --> I
    F --> K
    C --> L
    G --> N
    H --> K
    I --> K
    J -.-> F
    J -.-> G
    J -.-> H
```

### 5. Data Flow Overview

```mermaid
sequenceDiagram
    participant User
    participant Flask
    participant Model
    participant Data
    participant Autoencoder
    participant Matcher

    User->>Flask: Request Match
    Flask->>Model: Process Request
    Model->>Data: Load User Data
    Data->>Model: Raw User Profile
    Model->>Autoencoder: Encode User
    Autoencoder->>Model: User Embedding (8D)
    Model->>Matcher: Find Compatible Partners
    Matcher->>Model: Compatibility Scores
    Model->>Flask: Match Recommendations
    Flask->>User: Return Matches
```

## Key Features

### Data Processing Features

- **75+ Input Features**: Hobbies, goals, values, personality, faculty, gender
- **Chula-Specific**: Faculty mapping for Chulalongkorn University students
- **Categorical Encoding**: One-hot encoding for all categorical features
- **Normalization**: Age and numerical features normalized to 0-1 range

### Model Features

- **Two-Stage Architecture**: Autoencoder + Collaborative Filtering
- **Dimensionality Reduction**: 75 → 8 dimensions for efficient matching
- **Cosine Similarity**: For compatibility scoring
- **Real-time Matching**: Fast inference for user recommendations

### System Features

- **Modular Design**: Clean separation of concerns
- **Flask Web Server**: RESTful API endpoints
- **PyTorch Backend**: Modern deep learning framework
- **Development Ready**: Complete working implementation in `_original.py`

## Technology Stack

- **Backend**: Python 3.13+, Flask 3.1.0
- **ML Framework**: PyTorch 2.7.1
- **Data Processing**: Pandas, NumPy 2.3.0
- **Package Management**: uv (Astral)
- **Development**: Jupyter Notebooks for analysis

This architecture enables efficient, scalable matching for Chula students with a modern web interface and robust machine learning backend.
