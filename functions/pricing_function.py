# Models import
from Models.blackscholes import BlackScholes
from Models.heston import HestonModel
from Models.gammavariance import VarianceGamma
import numpy as np

MODELS = {
    "Black-Scholes": BlackScholes,
    "Heston": HestonModel,
    "Gamma Variance": VarianceGamma, 
}

def price_option(model_name: str, params: dict):
    """
    Fonction générique de pricing.
    model_name : nom du modèle sélectionné
    params : dictionnaire des paramètres de pricing
    """
    if model_name not in MODELS:
        raise ValueError(f"Modèle '{model_name}' non reconnu")

    ModelClass = MODELS[model_name]
    model = ModelClass(**params)  
    
    return model.price()
