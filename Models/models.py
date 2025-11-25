from abc import ABC, abstractmethod

class Model(ABC):
    """
    Classe de base pour tous les modèles de pricing.
    Chaque modèle doit implémenter au minimum :
    - price()
    - implied_volatility()
    """

    def __init__(self, S, K, r, T, option_type="call", position="buy", option_class="vanille"):
        self.S = S              # Prix du sous-jacent
        self.K = K              # Strike
        self.r = r              # Taux sans risque
        self.T = T              # Maturité
        self.option_type = option_type.lower()
        self.position = position.lower()
        self.option_class = option_class.lower()

    @abstractmethod
    def price(self, **kwargs):
        """Retourne le prix du modèle"""
        pass

    
