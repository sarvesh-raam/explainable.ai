import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import shap
import matplotlib.pyplot as plt
import os

def run_text_xai_demo():
    print("Initializing Textual XAI Demo...")
    plots_dir = 'results/plots/text_demo'
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Simple 'Training Data': Sentences categorized as Normal (0) or Urgent (1)
    data = {
        'text': [
            "I have a slight headache", "My finger is itching", "I think I have a cold",
            "SEVERE CHEST PAIN AND DIZZINESS", "URGENT HEART ATTACK SYMPTOMS", "I CANNOT BREATHE",
            "Just a normal checkup", "My knee hurts after walking", "I feel fine today",
            "SUDDEN PARALYSIS ON LEFT SIDE", "CRUSHING PRESSURE IN CHEST", "My foot is sore"
        ],
        'label': [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0]
    }
    df = pd.DataFrame(data)

    # 2. Convert text to numbers (Tfidf)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['text'])
    
    # 3. Train a simple 'Conversation Monitor' model
    model = LogisticRegression()
    model.fit(X, df['label'])

    # 4. The NEW sentence we want to explain
    test_sentence = ["I feel mild chest pain but I am mostly dizzy"]
    print(f"\nExplaining sentence: '{test_sentence[0]}'")

    # 5. Define a wrapper function that handles the conversion for SHAP
    # This is exactly how we use XAI on 'behind the scenes' of a conversation
    def predict_function(texts):
        return model.predict_proba(vectorizer.transform(texts))

    # Use LinearExplainer for TF-IDF + Logistic Regression as it's more stable
    # Transform the test sentence first
    test_vec = vectorizer.transform(test_sentence).toarray()
    
    explainer = shap.LinearExplainer(model, X.toarray(), feature_names=vectorizer.get_feature_names_out())
    shap_values = explainer.shap_values(test_vec)

    # 7. Generate Bar Plot for words
    plt.figure(figsize=(12, 6))
    
    # We take the values for the test sentence
    # For binary classification, shap_values is a simple array of importance for the positive class
    importance = shap_values.flatten()
    feature_names = vectorizer.get_feature_names_out()
    
    # Only plot words that were actually in the test sentence
    word_indices = test_vec[0].nonzero()[0]
    relevant_importance = importance[word_indices]
    relevant_names = [feature_names[i] for i in word_indices]

    # Simple bar chart for demonstration
    plt.barh(relevant_names, relevant_importance, color=['red' if x > 0 else 'blue' for x in relevant_importance])
    plt.axvline(0, color='black', linewidth=0.8)
    plt.title("XAI 'Behind the Scenes': Which words triggered the 'Urgent' Alert?")
    plt.xlabel("SHAP Value (Impact on Decision)")
    
    output_path = os.path.join(plots_dir, 'text_explanation_demo.png')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"\nSuccess! Text XAI plot saved to: {output_path}")
    print("This plot shows the exact 'SHAP values' of the words in the human-AI conversation.")

if __name__ == "__main__":
    run_text_xai_demo()
