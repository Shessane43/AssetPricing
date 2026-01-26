from Models.blackscholes import BlackScholes
from Models.heston import HestonModel
from Models.gammavariance import VarianceGamma

MODELS = {
    "Black-Scholes": BlackScholes,
    "Heston": HestonModel,
    "Gamma Variance": VarianceGamma
}

def price_option(model_name: str, params: dict):
    if model_name not in MODELS:
        raise ValueError(f"Modèle '{model_name}' non reconnu")

    ModelClass = MODELS[model_name]
    params_clean = params.copy()

    params_clean["option_type"] = params_clean["option_type"].lower()
    params_clean["option_class"] = params_clean["option_class"].lower()

    if model_name == "Black-Scholes":

        buy_sell = params_clean.pop("buy_sell").lower()
        params_clean["buy_sell"] = "buy" if buy_sell in ["long", "buy"] else "sell"

        for p in ["v0", "kappa", "theta", "sigma_v", "rho", "nu"]:
            params_clean.pop(p, None)

        return ModelClass(**params_clean).price()

    elif model_name == "Heston":

        buy_sell = params_clean.pop("buy_sell").lower()
        params_clean["position"] = "buy" if buy_sell in ["long", "buy"] else "sell"

        params_clean.pop("sigma", None)

        for p in ["v0", "kappa", "theta", "sigma_v", "rho"]:
            if p not in params_clean:
                raise ValueError(f"Paramètre '{p}' manquant pour Heston")

        return ModelClass(**params_clean).price()


    elif model_name == "Gamma Variance":

        buy_sell = params_clean.pop("buy_sell").lower()
        params_clean["position"] = "buy" if buy_sell in ["long", "buy"] else "sell"
        params_clean.pop("q", None)
        params_clean.pop("option_class", None)

        for p in ["theta", "nu", "sigma"]:
            if p not in params_clean:
                raise ValueError(f"Paramètre '{p}' manquant pour Gamma Variance")

        return ModelClass(**params_clean).price()


    else:
        raise ValueError("Modèle non supporté")
