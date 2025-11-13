
##################################
# ROCKET Hyperparameters
rocket_kernels_list = [2500, 5000, 10000, 15000, 20000, 25000, 30000]
modality_combinations = [
    ['g','h','f']
]

"""

    ['g'],
    ['h'],
    ['f'],
    ['g','h'],
    ['g','f'],
    ['h','f'],
"""

classifier = "RF"               # RF / LR / XGB
rf_n_estimators=300
rf_max_depth=50                 # Limit depth
rf_min_split=5                  # Require n samples to split
rf_max_features='sqrt'          # Use sqrt(n_features) per split
rf_n_jobs=-1
rf_class_weight='balanced'      # Handle class imbalance

lr_max_iter=5000
solver='saga'
penalty='elasticnet'
lr_l1_ratio=0.5
# Uses y values to automatically adjust weights inversely proportional 
# to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))
lr_class_weight='balanced' 
