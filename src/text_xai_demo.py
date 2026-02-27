import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import shap
import matplotlib.pyplot as plt
import os

def run_text_xai_demo():
    print("====================================================")
    print("      INTERACTIVE TEXTUAL XAI DEMO")
    print("====================================================")
    plots_dir = 'results/plots/text_demo'
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Simple 'Training Data'
    data = {
        'text': [
            "I have a slight headache", "My finger is itching", "I think I have a cold",
            "SEVERE CHEST PAIN AND DIZZINESS", "URGENT HEART ATTACK SYMPTOMS", "I CANNOT BREATHE",
            "Just a normal checkup", "My knee hurts after walking", "I feel fine today",
            "SUDDEN PARALYSIS ON LEFT SIDE", "CRUSHING PRESSURE IN CHEST", "My foot is sore",
            "My stomach feels a bit weird", "Extreme difficulty in breathing", "Vision is blurry and head is spinning"
        ],
        'label': [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1]
    }
    df = pd.DataFrame(data)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['text'])
    model = LogisticRegression()
    model.fit(X, df['label'])

    explainer = shap.LinearExplainer(model, X.toarray(), feature_names=vectorizer.get_feature_names_out())

    print("\nModel is ready! Type symptoms to see how the AI 'thinks'.")
    
    while True:
        user_input = input("\nEnter a symptom sentence (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        
        test_sentence = [user_input]
        test_vec = vectorizer.transform(test_sentence).toarray()
        
        # Check if words are known to the model
        if test_vec.sum() == 0:
            print("(!) None of these words were in the training data. Try words like 'chest', 'pain', 'mild', 'severe', 'headache'.")
            continue

        # Get prediction
        prob = model.predict_proba(test_vec)[0][1]
        status = "URGENT (Red Alert)" if prob > 0.5 else "NORMAL (Green)"
        print(f"AI Decision: {status} (Urgency Score: {prob:.2f})")

        # Explain
        shap_values = explainer.shap_values(test_vec).flatten()
        feature_names = vectorizer.get_feature_names_out()
        
        word_indices = test_vec[0].nonzero()[0]
        relevant_importance = shap_values[word_indices]
        relevant_names = [feature_names[i] for i in word_indices]

        # Plot
        plt.figure(figsize=(10, 5))
        colors = ['red' if x > 0 else 'blue' for x in relevant_importance]
        plt.barh(relevant_names, relevant_importance, color=colors)
        plt.axvline(0, color='black', linewidth=0.8)
        plt.title(f"XAI Explanation: '{user_input}'\nDecision: {status}")
        plt.xlabel("SHAP Value (Contribution to Urgency)")
        
        output_path = os.path.join(plots_dir, 'text_explanation_demo.png')
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

        print(f"Explanation Plot updated: {output_path}")

    print("\nDemo finished.")

if __name__ == "__main__":
    run_text_xai_demo()
