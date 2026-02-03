"""
Test MLTrainer with object dtype data to reproduce and debug the error.
"""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, 'python-services')

from aurix.ml.trainer import MLTrainer, TrainingConfig

# Create test data with object dtype (simulating real scenario)
n_samples = 100
n_features = 10

# Create DataFrame with object dtype (mixed None values)
data = {}
for i in range(n_features):
    col = np.random.randn(n_samples).tolist()
    # Add some None values to make it object dtype
    col[0] = None
    data[f'feature_{i}'] = col

X = pd.DataFrame(data)
print(f"X dtypes: {X.dtypes.unique()}")
print(f"X shape: {X.shape}")

# Create labels
y = np.random.randint(0, 2, n_samples)

# Try MLTrainer
config = TrainingConfig(model_type="lightgbm", psi_threshold=0.25)
trainer = MLTrainer(config=config, model_dir="test_models")

try:
    model = trainer.train(X=X, y=y, feature_names=X.columns.tolist(), direction="LONG")
    print("SUCCESS!")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
