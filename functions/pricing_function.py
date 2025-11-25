# Models import
from Models.blackscholes import BlackScholes
from Models.heston import HestonLewis
from Models.gammavariance import VarianceGamma

# Dictionnaire central des modèles : classes, pas instances
MODELS = {
    "Black-Scholes": BlackScholes,
    "Heston": HestonLewis,
    "Gamma Variance": VarianceGamma
}

def price_option(model_name: str, params: dict):
    """
    Fonction générique de pricing.
    model_name : nom du modèle sélectionné
    params : dictionnaire des paramètres de pricing
    """
    if model_name not in MODELS:
        raise ValueError(f"Modèle '{model_name}' non reconnu")

    # Instanciation dynamique avec les params
    ModelClass = MODELS[model_name]
    model = ModelClass(**params)  # **params contient S, K, r, sigma, T, etc.
    
    return model.price()
