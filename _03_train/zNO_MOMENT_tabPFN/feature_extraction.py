import torch
import numpy as np
from tqdm import tqdm
from momentfm import MOMENTPipeline

class MomentEmbedder:
    def __init__(self, model_name="AutonLab/MOMENT-1-large", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading MOMENT model: {model_name} on {self.device}...")
        
        # Initialize MOMENT in embedding mode
        self.model = MOMENTPipeline.from_pretrained(
            model_name, 
            model_kwargs={'task_name': 'embedding'}
        )
        self.model.init()
        self.model.to(self.device)
        self.model.eval()

    def _prepare_tensor(self, dataset):
        """
        Converts OpenFace instances into a (N, C, T) tensor.
        C = Channels (Gaze+Head+Face), T = Time steps
        """
        batch_list = []
        for inst in dataset.instances:
            # Stack modalities: Gaze (8) + Head (13) + Face (17) = 38 Channels
            # Shape becomes (T, 38)
            combined = np.hstack([inst.gaze_info, inst.head_info, inst.face_info])
            
            # Transpose to (C, T) as expected by Time Series Transformers
            combined = combined.T 
            batch_list.append(combined)
        
        # Stack into (Batch, Channels, Time)
        # Note: All instances MUST have same time length (e.g. 250). 
        # If not, pad
        max_length = max(combined.shape[1] for combined in batch_list)
        padded_batch_list = []
        for combined in batch_list:
            if combined.shape[1] < max_length:
                pad_width = max_length - combined.shape[1]
                padded = np.pad(combined, ((0, 0), (0, pad_width)), mode='constant')
                padded_batch_list.append(padded)
            else:
                padded_batch_list.append(combined)
        
        data_tensor = torch.tensor(np.stack(padded_batch_list), dtype=torch.float32)
        
        # MOMENT expects normalization (often done internally, but good practice to check nan)
        data_tensor = torch.nan_to_num(data_tensor)
        return data_tensor

    def get_embeddings(self, dataset, batch_size=16):
        """
        Runs the dataset through MOMENT to extract embeddings.
        Returns: Numpy array of shape (N_samples, Embedding_Dim)
        """
        data_tensor = self._prepare_tensor(dataset)
        embeddings = []
        
        print(f"Extracting embeddings for {len(dataset)} samples...")
        with torch.no_grad():
            for i in tqdm(range(0, len(data_tensor), batch_size)):
                batch = data_tensor[i : i + batch_size].to(self.device)
                
                # MOMENT input mask (all ones = observe everything)
                input_mask = torch.ones(batch.shape[0], batch.shape[2]).to(self.device)
                
                # Forward pass
                # Output object has an 'embeddings' attribute
                output = self.model(x_enc=batch, input_mask=input_mask)
                
                # Embedding shape is usually (Batch, Channels, Hidden) or pooled
                # For classification, we often mean-pool over channels or take the representation
                # output.embeddings is typically (Batch, Time, Dim) or (Batch, Dim) depending on head
                # We will perform Global Average Pooling over time/channels if needed
                emb = output.embeddings.detach().cpu().numpy()
                
                # If shape is (Batch, N_Tokens, Dim), we pool to get (Batch, Dim)
                if emb.ndim == 3:
                    emb = emb.mean(axis=1) 
                    
                embeddings.append(emb)
                
        return np.vstack(embeddings)