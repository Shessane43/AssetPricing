from Models.blackscholes import BlackScholes
from Models.heston import HestonModel
from Models.gammavariance import VarianceGamma
from Models.treemodel import TrinomialTree
from Models.bachelier import Bachelier
from Models.mertonjump import MertonJumpDiffusion


# ============================================================
# Available pricing models registry
# ============================================================
# Maps user-facing model names to their corresponding classes.
# This dictionary is used to dynamically select the pricing model.
MODELS = {
    "Black-Scholes": BlackScholes,
    "Heston": HestonModel,
    "Gamma Variance": VarianceGamma,
    "Trinomial Tree": TrinomialTree,
    "Bachelier": Bachelier,
    "Merton Jump Diffusion": MertonJumpDiffusion
}


def price_option(model_name: str, params: dict):
    """
    Price an option using the selected pricing model.

    This function acts as a unified interface for all supported models.
    It cleans and adapts the input parameters to match the expected
    constructor of each model.

    Parameters
    ----------
    model_name : str
        Name of the pricing model (must be a key of MODELS).
    params : dict
        Dictionary of model and option parameters.

    Returns
    -------
    float
        Option price.

    Raises
    ------
    ValueError
        If the model is not supported or required parameters are missing.
    """
    if model_name not in MODELS:
        raise ValueError(f"Modèle '{model_name}' non reconnu")

    ModelClass = MODELS[model_name]
    params_clean = params.copy()

    # Normalize option descriptors
    params_clean["option_type"] = params_clean["option_type"].lower()
    params_clean["option_class"] = params_clean["option_class"].lower()

    # ============================================================
    # Black-Scholes model
    # ============================================================
    if model_name == "Black-Scholes":

        # Convert buy/sell or long/short convention
        buy_sell = params_clean.pop("buy_sell").lower()
        params_clean["buy_sell"] = "buy" if buy_sell in ["long", "buy"] else "sell"

        # Remove parameters not used in Black-Scholes
        for p in ["v0", "kappa", "theta", "sigma_v", "rho", "nu"]:
            params_clean.pop(p, None)

        return ModelClass(**params_clean).price()

    # ============================================================
    # Heston stochastic volatility model
    # ============================================================
    elif model_name == "Heston":

        # Convert buy/sell or long/short convention
        buy_sell = params_clean.pop("buy_sell").lower()
        params_clean["position"] = "buy" if buy_sell in ["long", "buy"] else "sell"

        # Remove constant volatility (not used in Heston)
        params_clean.pop("sigma", None)

        # Check required Heston parameters
        for p in ["v0", "kappa", "theta", "sigma_v", "rho"]:
            if p not in params_clean:
                raise ValueError(f"Paramètre '{p}' manquant pour Heston")

        return ModelClass(**params_clean).price()

    # ============================================================
    # Variance Gamma model
    # ============================================================
    elif model_name == "Gamma Variance":

        # Convert buy/sell or long/short convention
        buy_sell = params_clean.pop("buy_sell").lower()
        params_clean["position"] = "buy" if buy_sell in ["long", "buy"] else "sell"

        # Remove unused parameters
        params_clean.pop("q", None)
        params_clean.pop("option_class", None)

        # Check required Variance Gamma parameters
        for p in ["theta", "nu", "sigma"]:
            if p not in params_clean:
                raise ValueError(f"Paramètre '{p}' manquant pour Gamma Variance")

        return ModelClass(**params_clean).price()

    # ============================================================
    # Trinomial tree model (discrete-time)
    # ============================================================
    elif model_name == "Trinomial Tree":

        # Convert buy/sell or long/short convention
        buy_sell = params_clean.pop("buy_sell").lower()
        position = "buy" if buy_sell in ["long", "buy"] else "sell"

        # Remove unused parameters
        params_clean.pop("option_class", None)

        return ModelClass(
            S=params_clean["S"],
            K=params_clean["K"],
            r=params_clean["r"],
            q=params_clean.get("q", 0.0),
            T=params_clean["T"],
            sigma=params_clean["sigma"],
            option_type=params_clean["option_type"],
            exercise=params_clean.get("exercise", "european"),
            n_steps=params_clean.get("n_steps", 100),
        ).price()

    # ============================================================
    # Bachelier (normal) model
    # ============================================================
    elif model_name == "Bachelier":

        # Convert buy/sell or long/short convention
        buy_sell = params_clean.pop("buy_sell").lower()
        sign = 1.0 if buy_sell in ["long", "buy"] else -1.0

        # Remove unused parameters
        params_clean.pop("q", None)
        params_clean.pop("option_class", None)

        # Check required parameters
        for p in ["sigma"]:
            if p not in params_clean:
                raise ValueError(f"Paramètre '{p}' manquant pour Bachelier")

        price = ModelClass(
            S=params_clean["S"],
            K=params_clean["K"],
            r=params_clean["r"],
            T=params_clean["T"],
            sigma=params_clean["sigma"],
            option_type=params_clean["option_type"],
        ).price()

        return sign * price

    # ============================================================
    # Merton jump-diffusion model
    # ============================================================
    elif model_name == "Merton Jump Diffusion":

        # Convert buy/sell or long/short convention
        buy_sell = params_clean.pop("buy_sell").lower()
        sign = 1.0 if buy_sell in ["long", "buy"] else -1.0

        # Remove unused parameters
        params_clean.pop("q", None)
        params_clean.pop("option_class", None)

        # Check required jump parameters
        for p in ["sigma", "lambd", "mu_j", "sigma_j"]:
            if p not in params_clean:
                raise ValueError(f"Paramètre '{p}' manquant pour Merton Jump")

        price = ModelClass(
            S=params_clean["S"],
            K=params_clean["K"],
            r=params_clean["r"],
            T=params_clean["T"],
            sigma=params_clean["sigma"],
            lambd=params_clean["lambd"],
            mu_j=params_clean["mu_j"],
            sigma_j=params_clean["sigma_j"],
            option_type=params_clean["option_type"],
        ).price()

        return sign * price

    # ============================================================
    # Fallback (should not be reached)
    # ============================================================
    else:
        raise ValueError("Model non supported")
