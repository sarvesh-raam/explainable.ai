# 🏥 Explainable AI (XAI) in Healthcare
### *Balancing Accuracy and Interpretability: A Post-hoc Framework for Structured Healthcare Data*

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-Black?style=for-the-badge&logo=expert-systems&logoColor=white)
![Healthcare](https://img.shields.io/badge/Focus-Healthcare-red?style=for-the-badge)

---

## 🎯 Project Overview
This research project investigates the reliability, stability, and clinical utility of post-hoc explainability methods (**SHAP** and **LIME**) when applied to black-box machine learning models in healthcare. We aim to bridge the gap between high-performance "black-box" models and the transparency required for clinical decision-making.

### 🧪 Central Research Question
> *"Can post-hoc explainability methods produce faithful, stable, and clinically useful explanations for black-box healthcare models—while maintaining high predictive performance?"*

---

## 🔬 Research Objectives
*   **Empirical Comparison:** Evaluating the performance of Logistic Regression, Random Forest, and Gradient Boosting.
*   **XAI Evaluation:** Comparing SHAP (Global & Local) vs. LIME (Local) explanations.
*   **Stability Analysis:** Measuring the robustness of explanations against data perturbations (Gaussian noise).
*   **Consistency Check:** Quantifying the agreement between different explanation methods.

---

## 🏗️ The 5-Phase Research Plan

| Phase | Title | Status | Description |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **Dataset & Preprocessing** | 🟡 In Progress | Cleaning and engineering UCI Heart Disease & Adult Income datasets. |
| **Phase 2** | **Model Training** | ⚪ Pending | Building predictive engines using ensemble and linear methods. |
| **Phase 3** | **SHAP Analysis** | ⚪ Pending | Global and local feature driver identification. |
| **Phase 4** | **LIME Analysis** | ⚪ Pending | Local instance analysis and consistency comparisons. |
| **Phase 5** | **Stability Testing** | ⚪ Pending | Quantifying the trustworthiness of explanations for publication. |

---

## 🛠️ Tech Stack
*   **Core Logic:** `Python 3.x`
*   **Data Handling:** `pandas`, `NumPy`
*   **Machine Learning:** `scikit-learn`, `XGBoost`
*   **Explainability:** `SHAP`, `LIME`
*   **Visualization:** `Matplotlib`, `Seaborn`

---

## 📂 Directory Structure
```bash
Explainable-AI/
├── data/               # Raw and processed healthcare datasets
├── src/                # Python source code for preprocessing & modeling
├── notebooks/          # Experimental analysis and Jupyter Visualizations
├── results/            # Research plots, tables, and findings
└── requirements.txt    # Project dependencies
```

---

## 🤝 Collaborators

| <a href="https://github.com/sarvesh-raam"><img src="https://github.com/sarvesh-raam.png" width="120px;" style="border-radius: 50%;" alt=""/></a> | <a href="https://github.com/Vigneshhhhhhhhhh"><img src="https://github.com/Vigneshhhhhhhhhh.png" width="120px;" style="border-radius: 50%;" alt=""/></a> |
| :---: | :---: |
| **[sarvesh-raam](https://github.com/sarvesh-raam)** | **[Vigneshhhhhhhhhh](https://github.com/Vigneshhhhhhhhhh)** |

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
