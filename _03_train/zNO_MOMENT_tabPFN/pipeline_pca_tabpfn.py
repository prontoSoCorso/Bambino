import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNClassifier
from sklearn.metrics import classification_report, f1_score, balanced_accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class HybridPipeline:
    def __init__(self, n_pca_components=50):
        self.n_pca_components = n_pca_components
        self.pca = PCA(n_components=n_pca_components)
        self.scaler_meta = StandardScaler()
        # TabPFN is robust, but scaling input embeddings helps PCA
        self.scaler_emb = StandardScaler()
        
        # TabPFN Classifier
        self.clf = TabPFNClassifier(device='cuda', n_estimators=8)

    def process_metadata(self, dataset, yes_audio=False, fit_scaler=False):
        """Extracts and encodes static metadata."""
        # Extract features manually to DataFrame
        meta_rows = []
        for inst in dataset.instances:
            row = {
                'age': inst.age,
                'sex': inst.sex,
                'trial_id': inst.trial_id
            }
            if yes_audio:
                # Encode audio group to integer
                row['audio'] = dataset.audio_to_idx.get(inst.audio, -1)
            meta_rows.append(row)
        
        df = pd.DataFrame(meta_rows)
        
        # Normalize numeric metadata (Age)
        # Sex and Audio are categorical integers; TabPFN handles them well, 
        # but we treat Age as continuous.
        if fit_scaler:
            df[['age']] = self.scaler_meta.fit_transform(df[['age']])
        else:
            df[['age']] = self.scaler_meta.transform(df[['age']])
            
        return df.values

    def fit_transform_pca(self, embeddings_train):
        """Fits PCA on training embeddings and transforms them."""
        # Scale first
        emb_scaled = self.scaler_emb.fit_transform(embeddings_train)
        
        print(f"Fitting PCA to reduce dim from {emb_scaled.shape[1]} to {self.n_pca_components}...")
        pca_emb = self.pca.fit_transform(emb_scaled)
        
        explained_var = np.sum(self.pca.explained_variance_ratio_)
        print(f"PCA Explained Variance: {explained_var:.4f}")
        
        return pca_emb

    def transform_pca(self, embeddings_test):
        """Transforms test embeddings using fitted PCA."""
        emb_scaled = self.scaler_emb.transform(embeddings_test)
        return self.pca.transform(emb_scaled)

    def train_tabpfn(self, X_train, y_train):
        print(f"Training TabPFN on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
        # TabPFN expects y to be integers? It handles it, but good to ensure
        self.clf.fit(X_train, y_train)

    def evaluate(self, X, y, split_name="Test"):
        preds = self.clf.predict(X)
        probs = self.clf.predict_proba(X)
        
        print(f"--- {split_name} Results ---")
        print(classification_report(y, preds))
        print(f"Balanced Acc: {balanced_accuracy_score(y, preds):.4f}")
        print(f"F1 Macro: {f1_score(y, preds, average='macro'):.4f}")
        
        # Plot Confusion Matrix
        cm = confusion_matrix(y, preds)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"{split_name} Confusion Matrix")
        plt.show()
        
        return preds, probs