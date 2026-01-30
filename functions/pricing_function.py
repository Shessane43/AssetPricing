from Models.blackscholes import BlackScholes
from Models.heston import HestonModel
from Models.gammavariance import VarianceGamma
from Models.treemodel import TrinomialTree
from Models.bachelier import Bachelier
from Models.mertonjump import MertonJumpDiffusion

MODELS = {
    "Black-Scholes": BlackScholes,
    "Heston": HestonModel,
    "Gamma Variance": VarianceGamma,
    "Trinomial Tree": TrinomialTree,
    "Bachelier": Bachelier,
    "Merton Jump Diffusion": MertonJumpDiffusion
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
    elif model_name == "Trinomial Tree":

        buy_sell = params_clean.pop("buy_sell").lower()
        position = "buy" if buy_sell in ["long", "buy"] else "sell"
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
    elif model_name == "Bachelier":
        buy_sell = params_clean.pop("buy_sell").lower()
        sign = 1.0 if buy_sell in ["long", "buy"] else -1.0

        params_clean.pop("q", None)
        params_clean.pop("option_class", None)
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
    
    elif model_name == "Merton Jump Diffusion":

        buy_sell = params_clean.pop("buy_sell").lower()
        sign = 1.0 if buy_sell in ["long", "buy"] else -1.0

        params_clean.pop("q", None)
        params_clean.pop("option_class", None)

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

    else:
        raise ValueError("Modèle non supporté")
