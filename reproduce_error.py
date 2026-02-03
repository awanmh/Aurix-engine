
import numpy as np
import pandas as pd
import logging
from aurix.ml.trainer import MLTrainer, TrainingConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_ml_trainer():
    logger.info("Starting MLTrainer test...")
    
    # Create dummy data
    n_samples = 1000
    n_features = 20
    
    # X as DataFrame with floats
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features), 
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # y as string labels (simulating backtest behavior)
    y_ints = np.random.randint(0, 2, n_samples)
    y = np.where(y_ints == 1, 'WIN', 'LOSS')
    
    # Create trainer
    config = TrainingConfig(model_type="lightgbm", psi_threshold=0.25)
    trainer = MLTrainer(config=config, model_dir="test_models")
    
    # Train
    logger.info("Calling train()...")
    try:
        model = trainer.train(
            X=X,
            y=y,
            feature_names=X.columns.tolist(),
            direction="LONG"
        )
        logger.info("Training successful!")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)

if __name__ == "__main__":
    test_ml_trainer()
