# IEEE Research-Grade Prototype Architecture

This document contains the high-fidelity conceptual schematic for the Explainable AI (XAI) framework, optimized for academic publication and research rigor.

## Architecture Diagram

```mermaid
%%{init: {
    'theme': 'base', 
    'themeVariables': { 
        'primaryColor': '#ffffff', 
        'mainBkg': '#ffffff', 
        'fontSize': '14px', 
        'fontFamily': 'arial', 
        'lineColor': '#2c3e50', 
        'nodeBorder': '#2c3e50', 
        'clusterBkg': '#fdfdfd', 
        'clusterBorder': '#7f8c8d'
    }
}}%%
graph TB
    %% STAGE I: SYSTEM DESIGN
    subgraph Stage_I [<b>PHASE I: SYSTEM INITIALIZATION & DATA ORCHESTRATION</b>]
        direction LR
        subgraph Data_Silo [Ingestion Layer]
            RD[(<b>Raw Input</b><br/>Table/CSV)] 
        end
        
        subgraph Pre_Processing [Transformation & Engineering]
            direction TB
            FE[Feature Selection]
            NT[Normalization]
            CE[Categorical Encoding]
            FE --> NT --> CE
        end
        
        Data_Silo --> Pre_Processing
    end

    %% STAGE II: ARCHITECTURAL LOGIC
    subgraph Stage_II [<b>PHASE II: PREDICTIVE MODELING & BASELINING</b>]
        direction LR
        subgraph Neural_Ensembles [Non-Linear Models]
            RF([Random Forest])
            XGB([XGBoost])
        end
        
        subgraph Formal_Baselines [Linear Reference]
            LR([Logistic Regression])
        end
        
        Models_Gate{<b>Model Optimization</b>}
    end

    Stage_I --> Models_Gate
    Models_Gate --> Neural_Ensembles
    Models_Gate --> Formal_Baselines

    %% STAGE III: CORE INTERPRETABILITY
    subgraph Stage_III [<b>PHASE III: POST-HOC INTERPRETABILITY ENGINE</b>]
        direction LR
        subgraph XAI_Cores [Explanation Generators]
            SHAP[[SHAP Framework]]
            LIME[[LIME Framework]]
        end

        subgraph Analytic_Artifacts [Research Output]
            GS(Global Summary)
            LS(Local Force Plot)
            SA(Surrogate Analysis)
        end
        
        SHAP --> GS & LS
        LIME --> SA
    end

    Neural_Ensembles & Formal_Baselines --> XAI_Cores

    %% STAGE IV: ROBUSTNESS
    subgraph Stage_IV [<b>PHASE IV: STABILITY & TRUST VALIDATION</b>]
        direction LR
        subgraph Reliability_Check [Robustness Protocol]
            NP[Noise Perturbation]
            SI[Stability Indexing]
            NP --> SI
        end
    end

    Stage_III --> Reliability_Check

    %% FINAL SYNTHESIS
    subgraph Stage_V [<b>PHASE V: SCIENTIFIC KNOWLEDGE SYNTHESIS</b>]
        direction LR
        Artifacts[Research Artifacts]
        Publication[/<b>IEEE Standard Publication</b>/]
        Artifacts --> Publication
    end

    Reliability_Check --> Artifacts

    %% ADVANCED NODE STYLING
    style RD fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style Publication fill:#ffebee,stroke:#c62828,stroke-width:3px
    style Stage_I fill:#f8f9fa,stroke:#333,stroke-dasharray: 5 5
    style Stage_II fill:#f3e5f5,stroke:#7b1fa2
    style Stage_III fill:#e8f5e9,stroke:#2e7d32
    style Stage_IV fill:#fff3e0,stroke:#e65100
    style Stage_V fill:#eceff1,stroke:#455a64

    %% Advanced Class Styling
    classDef engine fill:#fff,stroke:#2c3e50,stroke-width:2px,rx:10,ry:10
    class RF,XGB,LR,SHAP,LIME engine
```
