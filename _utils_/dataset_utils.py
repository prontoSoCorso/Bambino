import torch

# ────────────────────────────────────────────────────────────────────────────────
# Dataset analysis: NaN check, number of instances, modalities
# ────────────────────────────────────────────────────────────────────────────────
def analyze_data_quality(datasets):
    """Analyze data quality and provide insights"""
    print("\n🔍 Data Quality Analysis")
    print("=" * 50)
    
    for name, dataset in datasets.items():
        print(f"\n{name.upper()} SET:")
        print(f"  📊 Size: {len(dataset)} samples")
        
        # Sample a few instances to check data
        if len(dataset) > 0:
            sample_X, sample_y, _ = dataset[0]
            print(f"  🔢 Input shapes:")
            for modality, data in sample_X.items():
                print(f"    {modality}: {data.shape}")
                
                # Check for NaNs
                nan_count = torch.isnan(data).sum().item()
                if nan_count > 0:
                    print(f"    ⚠️  WARNING: {nan_count} NaN values found in {modality}")
                
                # Basic statistics
                print(f"    📈 {modality} - Mean: {data.mean():.4f}, Std: {data.std():.4f}")
            
            print(f"  🎯 Label: {sample_y.item()}")


# ────────────────────────────────────────────────────────────────────────────────
# Normalization and Saving norm_params of train
# ────────────────────────────────────────────────────────────────────────────────
def compute_normalization_params(train_dataset):
    """
    Given a dataset where each __getitem__ returns (X_dict, y, _),
    this function iterates over all samples in `train_dataset`, extracts
    the modality tensors (possibilities: "g", "h", "f"), and computes, for each modality,
    a per feature (column wise) mean and std (over all samples *and* all timesteps).
    Returns a dict:
        {
          "g": {"mean": Tensor of shape [g_dim], "std": Tensor of shape [g_dim]},
          "h": {"mean": Tensor of shape [h_dim], "std": Tensor of shape [h_dim]},
          "f": {"mean": Tensor of shape [f_dim], "std": Tensor of shape [f_dim]},
        }
    """
    # Accumulate sums and squared sums for each modality
    modalities = getattr(train_dataset, 'modalities', ['g', 'h', 'f'])
    sum_dict = {m: None for m in modalities}
    sq_sum_dict = {m: None for m in modalities}
    n_total = 0  # total number of (samples * timesteps)

    # First, loop over every sample in train_dataset
    for X_dict, _, _ in train_dataset:
        # Assume X_dict["m"] is a FloatTensor of shape [T, m_dim].
        for modality in modalities:
            data = X_dict[modality]              # shape: [T, dim]
            T, _ = data.shape

            # Flatten time dimension into batch: shape [T, dim] → shape [T, dim]
            # We'll accumulate sums over rows and later divide by (num_samples * T).
            if sum_dict[modality] is None:
                sum_dict[modality] = data.sum(dim=0)             # shape [dim]
                sq_sum_dict[modality] = (data ** 2).sum(dim=0)    # shape [dim]
            else:
                sum_dict[modality] += data.sum(dim=0)
                sq_sum_dict[modality] += (data ** 2).sum(dim=0)

        n_total += T  # add number of timesteps (T must be constant across samples)

    # Now compute mean and std for each modality
    norm_params = {}
    for modality in modalities:
        # Mean over all samples * timesteps:
        mean = sum_dict[modality] / float(n_total)       # shape [dim]

        # Var = E[x^2] - (E[x])^2
        var = (sq_sum_dict[modality] / float(n_total)) - (mean ** 2)
        # Numerical stability: clamp var to a minimum (e.g., 1e-6)
        var = torch.clamp(var, min=1e-6)
        std = torch.sqrt(var)

        norm_params[modality] = {"mean": mean, "std": std}

    return norm_params


# ────────────────────────────────────────────────────────────────────────────────
# Apply normalization with already known params (from the train, for val and test)
# ────────────────────────────────────────────────────────────────────────────────
def apply_normalization(dataset, norm_params):
    """
    Given any dataset (train/val/test) whose __getitem__ returns (X_dict, y, _),
    modify the dataset *in place* so that each X_dict[modality] is replaced by
        (X_dict[modality] - mean) / std
    using the provided norm_params (computed on TRAIN).
    We assume that the dataset stores its raw data in some internal list/structure
    that __getitem__ reads from. For most custom PyTorch Dataset classes, one approach
    is to override __getitem__, but here we’ll modify each instance directly
    (assuming dataset.instances[i].___ attributes or similar). If your dataset
    does not expose a straightforward way to write back, you can wrap it instead.
    """
    # Check for instance attribute
    if not hasattr(dataset, 'instances'):
        raise RuntimeError(
            "Dataset does not expose `instances`. Adapt apply_normalization."
        )

    # Each instance is often a custom object with attributes `gaze_info`, `head_info`, `face_info`.
    # But since __getitem__ yields X_dict = {"g": tensor, "h": tensor, "f": tensor},
    # we can patch the same attributes.
    attr_map = {'g': 'gaze_info', 'h': 'head_info', 'f': 'face_info'}
    modalities = getattr(dataset, 'modalities', list(norm_params.keys()))

    for inst in dataset.instances:
        for m in modalities:
            attr = attr_map.get(m)
            if not attr:
                continue
            # The complete attributes are:
            #   - inst.gaze_info (shape [T, 8])
            #   - inst.head_info (shape [T, 13])
            #   - inst.face_info (shape [T, 17])
            raw = getattr(inst, attr)
            tensor_data = (
                torch.tensor(raw, dtype=torch.float32)
                if not isinstance(raw, torch.Tensor)
                else raw
            )
            mean = norm_params[m]['mean']
            std = norm_params[m]['std']
            normed = (tensor_data - mean.unsqueeze(0)) / std.unsqueeze(0)

            # write back in original type
            if not isinstance(raw, torch.Tensor):
                setattr(inst, attr, normed.numpy())
            else:
                setattr(inst, attr, normed)