import joblib
import os
import logging

# Set up logging so you can see errors in the console
logger = logging.getLogger(__name__)

def load_model(model_name: str = "model.joblib"):
    """
    Loads the trained model pipeline. 
    If the file is missing, it returns None so the API can stay online for debugging.
    """
    # Use absolute path logic if you are running from different folders
    base_path = os.path.dirname(os.path.dirname(__file__)) # Moves up from services/ to project root
    model_path = os.path.join(base_path, model_name)

    if not os.path.exists(model_path):
        # We log the error instead of 'raising' it to prevent a boot crash
        logger.error(f"MODEL NOT FOUND at: {model_path}")
        return None

    try:
        model = joblib.load(model_path)
        logger.info("✅ Model loaded successfully into memory.")
        return model
    except Exception as e:
        logger.error(f"❌ Error unpickling the model: {e}")
        return None