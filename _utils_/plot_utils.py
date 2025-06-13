from typing import Dict, List, Optional
import seaborn as sns
import matplotlib.pyplot as plt

# Umap import
import pandas as pd
import mplcursors
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output
from umap.umap_ import UMAP
from sklearn.preprocessing import StandardScaler
import os
import numpy as np
import torch
import dash
from dash import dcc, html, Input, Output


# Funzioni per salvare la matrice di confusione come immagine
def save_confusion_matrix(conf_matrix, filename, model_name, num_kernels=0):
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False, xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    if num_kernels>0:
        plt.title(f"Confusion Matrix - {model_name} - {num_kernels} Kernels")
    else: 
        plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(filename)
    plt.close()


# Plot ROC Curve
def plot_roc_curve(fpr, tpr, roc_auc, filename):
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(filename)
    plt.close()

def plot_training_history(
        train_history: Dict[str, List[float]] | List[Dict[str, float]],
        save_path: Optional[str] = None,
        figsize: tuple = (12, 6),
        dpi: int = 150
        ) -> None:
    """
    Plot training and validation metrics over epochs in una singola figura.
    Accetta sia:
      - un dict di liste (come prima)
      - una lista di dict (ogni dict = metriche di un’epoca)
    
    Args:
        Expected keys:
            - train_loss: List of training loss values
            - train_acc: List of training accuracy values
            - val_loss: List of validation loss values
            - val_balanced_acc: List of validation balanced accuracy values
            - val_mcc: List of validation Matthews Correlation Coefficient values
            - val_brier: List of validation Brier score values
        save_path: Optional path to save the figure. If None, figure is displayed.
        figsize: Size of the figure (width, height) in inches.
        dpi: Dots per inch for figure resolution.
    """

    # 1) Se mi arriva una lista di dict, la pivotto in dict di liste
    if isinstance(train_history, list):
        import pandas as pd
        df = pd.DataFrame(train_history)
        # Manteniamo solo le colonne che ci servono
        required_keys = {
            "train_loss", "train_acc", "val_loss",
            "val_balanced_acc", "val_mcc", "val_brier"
        }
        # Controllo che esistano nel DataFrame
        missing = required_keys - set(df.columns)
        if missing:
            raise ValueError(f"train_history manca delle colonne: {missing}")
        # Ricostruisco il dict di liste
        train_history = {k: df[k].tolist() for k in required_keys}

    # 2) A questo punto train_history deve essere un dict
    if not isinstance(train_history, dict):
        raise ValueError(f"train_history deve essere dict o list of dict, non {type(train_history)}")
    
    # 3) Validazione delle chiavi (solo per sicurezza)
    required_keys = {
        "train_loss", "train_acc", "val_loss",
        "val_balanced_acc", "val_mcc", "val_brier"
    }
    if not required_keys.issubset(train_history.keys()):
        missing = required_keys - set(train_history.keys())
        raise ValueError(f"Missing required keys in train_history: {missing}")
    
    epochs = list(range(1, len(train_history["train_loss"]) + 1))
    
    # Create figure and subplots
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=figsize, dpi=dpi)
    # Adjust layout parameters to prevent overlap
    plt.subplots_adjust(
        left=0.1,
        right=0.95,
        bottom=0.1,
        top=0.9,
        wspace=0.3,  # Horizontal space between subplots
        hspace=0.4   # Vertical space between subplots
    )
    
    # Define plot configurations for each metric
    plot_configs = [
        {
            "data": train_history["train_loss"],
            "title": "Training Loss",
            "ylabel": "Loss",
            "color": "blue",
            "ax_idx": (0, 0)
        },
        {
            "data": train_history["train_acc"],
            "title": "Training Accuracy",
            "ylabel": "Accuracy",
            "color": "blue",
            "ax_idx": (0, 1)
        },
        {
            "data": train_history["val_loss"],
            "title": "Validation Loss",
            "ylabel": "Loss",
            "color": "orange",
            "ax_idx": (0, 2)
        },
        {
            "data": train_history["val_balanced_acc"],
            "title": "Validation Balanced Accuracy",
            "ylabel": "Accuracy",
            "color": "green",
            "ax_idx": (1, 0)
        },
        {
            "data": train_history["val_mcc"],
            "title": "Validation MCC",
            "ylabel": "MCC Score",
            "color": "red",
            "ax_idx": (1, 1)
        },
        {
            "data": train_history["val_brier"],
            "title": "Validation Brier Score",
            "ylabel": "Brier Score",
            "color": "purple",
            "ax_idx": (1, 2)
        }
    ]
    
    # Create plots using configuration
    for config in plot_configs:
        ax = axes[config["ax_idx"][0], config["ax_idx"][1]]
        ax.plot(epochs, config["data"], 
               marker='o', 
               markersize=3,
               linestyle='-',
               linewidth=1.5,
               color=config["color"])
        
        ax.set_title(config["title"], pad=12, fontsize=12)
        ax.set_xlabel("Epoch", labelpad=8)
        ax.set_ylabel(config["ylabel"], labelpad=8)
        ax.grid(True, alpha=0.3)
        
        # Add padding and set limits to prevent crowding
        ax.margins(x=0.1, y=0.1)
        
        # Rotate x-axis labels if many epochs
        if len(epochs) > 10:
            ax.tick_params(axis='x', rotation=45)
        
    # Adjust super title
    fig.suptitle("Training & Validation Metrics", 
                fontsize=16, 
                y=1.,  # Adjusted downward
                weight='bold')
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi, pad_inches=0.5)
        plt.close(fig)
    else:
        plt.show()



def compute_UMAP(dataset, output_dir, filename, modalities={'g': "gaze_info",'h': "head_info",'f': "face_info"}):
    """
    Esegue UMAP su un singolo dataset PyTorch custom, estraendo per ogni istanza:
       - media temporale di gaze_info, head_info, face_info
       - etichetta (trial_type)
       - ID combinando pt_id e trial_id
    Produce e salva uno scatter interattivo con hover sugli ID.

    Args:
        dataset: oggetto custom con attributo `.instances`, lista di instanze che hanno
                 `.gaze_info`, `.head_info`, `.face_info`, `.trial_type`, `.pt_id`, `.trial_id`
        output_path_base: cartella in cui salvare il grafico (viene creato se non esiste)
    """
    # 1) Estrai feature, label, id
    features = []
    labels = []
    ids = []
    for inst in dataset.instances:
        # per ogni modality prendi la media lungo il tempo (asse 0)
        vecs = []
        for _, value in modalities.items():
            data = getattr(inst, value)
            # se è torch.Tensor → numpy
            if isinstance(data, torch.Tensor):
                data = data.cpu().numpy()
            # media lungo il tempo
            vecs.append(np.mean(data, axis=0))
        # concatena per sample
        features.append(np.concatenate(vecs, axis=0))
        # label e ID
        labels.append(inst.trial_type)
        ids.append(f"{inst.pt_id}_{inst.trial_id}")

    features = np.vstack(features)   # shape: [n_samples, total_dim]
    labels   = np.array(labels)
    ids      = np.array(ids)

    # 2) Standardizza
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 3) Applica UMAP
    umap_model = UMAP(n_components=2, n_jobs=-1, random_state=42)
    embedding = umap_model.fit_transform(features_scaled)

    # 4) DataFrame per plotting
    umap_df = pd.DataFrame({
        "Dim1": embedding[:, 0],
        "Dim2": embedding[:, 1],
        "Label": labels,
        "ID":    ids
    })

    # 5) Plot
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 8))
    colors = {0: "red", 1: "blue"}
    scatters = []
    for lbl, color in colors.items():
        sub = umap_df[umap_df["Label"] == lbl]
        sc = plt.scatter(sub["Dim1"], sub["Dim2"],
                         c=color, label=f"Classe {lbl}", alpha=0.7)
        # salva insieme allo scatter anche la lista degli ID, per l'hover
        scatters.append((sc, sub["ID"].tolist()))

    # hover interattivo
    for sc, id_list in scatters:
        cursor = mplcursors.cursor(sc)
        cursor.connect("add", lambda sel, ids=id_list: sel.annotation.set_text(ids[sel.index]))

    plt.title("Visualizzazione UMAP")
    plt.xlabel("Dimensione 1")
    plt.ylabel("Dimensione 2")
    plt.legend()
    plt.grid(True)

    # 6) Salva + mostra
    out_file = os.path.join(output_dir, filename)
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"UMAP plot salvato in: {out_file}")
    plt.show()



def compute_UMAP_plotly(dataset,
                        port: int = 8050):
    """
    Esegue UMAP su un singolo dataset custom e visualizza con Plotly/Dash.
    Se `output_path_base` è valorizzato, salva un PNG statico e mostra la figura.
    Altrimenti parte un'app Dash con dropdown per filtrare pt_id e trial_id.
    
    Args:
        dataset: dataset custom con .instances, ognuna con
                 .gaze_info, .head_info, .face_info, .trial_type, .pt_id, .trial_id
        output_path_base: cartella di output; se vuota si attiva la modalità Dash
        port: porta per il server Dash
    """
    # 1) Estrai features, labels, metadata
    features, labels, pt_ids, trial_ids = [], [], [], []
    for inst in dataset.instances:
        vecs = []
        for attr in ("gaze_info", "head_info", "face_info"):
            data = getattr(inst, attr)
            if isinstance(data, torch.Tensor):
                data = data.cpu().numpy()
            vecs.append(np.mean(data, axis=0))
        features.append(np.concatenate(vecs, axis=0))
        labels.append(inst.trial_type)
        pt_ids.append(inst.pt_id)
        trial_ids.append(inst.trial_id)
    
    features = np.vstack(features)
    labels   = np.array(labels)
    pt_ids   = np.array(pt_ids, dtype=str)
    trial_ids= np.array(trial_ids, dtype=str)

    # 2) Standardizza e UMAP
    scaler = StandardScaler()
    Xs = scaler.fit_transform(features)
    umap_model = UMAP(n_components=2, random_state=42, n_jobs=-1)
    emb = umap_model.fit_transform(Xs)

    # 3) Costruisci DataFrame
    df = pd.DataFrame({
        "Dim1":      emb[:, 0],
        "Dim2":      emb[:, 1],
        "Label":     labels,
        "pt_id":     pt_ids,
        "trial_id":  trial_ids
    })
    # mappa etichette in stringhe per legenda
    df["LabelStr"] = df["Label"].map({0: "Class 0", 1: "Class 1"})

    # 4) Crea traces per Plotly
    colors = {"Class 0": "red", "Class 1": "blue"}
    fig = go.Figure()
    customdata_cols = ["pt_id", "trial_id"]
    hovertemplate = (
        "Dim1: %{x:.3f}<br>"
        "Dim2: %{y:.3f}<br>" +
        "<br>".join([f"{col}: %{{customdata[{i}]}}" 
                     for i, col in enumerate(customdata_cols)]) +
        "<extra></extra>"
    )
    for lbl, col in colors.items():
        sub = df[df["LabelStr"] == lbl]
        fig.add_trace(go.Scatter(
            x=sub["Dim1"], y=sub["Dim2"],
            mode="markers",
            marker=dict(color=col),
            name=lbl,
            customdata=sub[customdata_cols].values,
            hovertemplate=hovertemplate
        ))
    fig.update_layout(
        title="UMAP Projection",
        width=1000, height=700,
        hovermode="closest"
    )

    # 5) Interattivo con Dash
    categorical = ["pt_id", "trial_id"]
    # costruisci dropdown
    dropdowns = []
    for col in categorical:
        opts = [{"label": "All", "value": "All"}] + [
            {"label": v, "value": v} for v in sorted(df[col].unique())
        ]
        dropdowns.append(html.Div([
            html.Label(col),
            dcc.Dropdown(
                id=f"dd-{col}", options=opts, value="All", clearable=False
            )
        ], style={"width": "20%", "display": "inline-block"}))

    app = dash.Dash(__name__)
    app.layout = html.Div([
        html.H2("UMAP Interactive"),
        html.Div(dropdowns),
        dcc.Graph(id="umap-graph", figure=fig)
    ])

    @app.callback(
        Output("umap-graph", "figure"),
        [Input(f"dd-{col}", "value") for col in categorical]
    )
    def update_figure(*selected):
        dff = df.copy()
        for col, val in zip(categorical, selected):
            if val != "All":
                dff = dff[dff[col] == val]
        new_fig = go.Figure()
        for lbl, colr in colors.items():
            sub = dff[dff["LabelStr"] == lbl]
            new_fig.add_trace(go.Scatter(
                x=sub["Dim1"], y=sub["Dim2"],
                mode="markers",
                marker=dict(color=colr),
                name=lbl,
                customdata=sub[customdata_cols].values,
                hovertemplate=hovertemplate
            ))
        new_fig.update_layout(
            title="UMAP Projection (filtered)",
            width=1000, height=700,
            hovermode="closest"
        )
        return new_fig

    app.run(debug=True, port=port)