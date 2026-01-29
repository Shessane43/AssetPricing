# Asset Pricing Application 📊

Une application complète et professionnelle pour la **tarification d'options**, l'analyse des **grecques**, la simulation de **volatilité** et la gestion de **portefeuille**. Construite avec **Streamlit** et **Python**, elle intègre plusieurs modèles de pricing avancés.

---

## 🎯 Vue d'ensemble

L'**Asset Pricing Application** est une plateforme interactive pour les analystes quantitatifs, traders et risk managers qui souhaitent :
- Pricer des options vanilles et exotiques
- Analyser les grecques (Delta, Gamma, Vega, Theta, Rho)
- Visualiser les surfaces de volatilité implicite
- Simuler des scénarios de volatilité
- Évaluer des obligations et swaps
- Créer et analyser des portefeuilles

---

## 📋 Table des matières

- [Caractéristiques principales](#-caractéristiques-principales)
- [Architecture](#-architecture)
- [Modèles implémentés](#-modèles-implémentés)
- [Installation et setup](#-installation-et-setup)
- [Utilisation](#-utilisation)
- [Structure des fichiers](#-structure-des-fichiers)
- [Guide détaillé des modules](#-guide-détaillé-des-modules)

---

## ✨ Caractéristiques principales

### 1. **Tarification d'Options**
- Modèle Black-Scholes (analytique)
- Modèle Heston (volatilité stochastique)
- Variance Gamma (sauts et asymétrie)
- Arbre trinomial (discrétisation)
- Support des **options vanilles** (Call/Put européens)
- Support des **options exotiques** (Asian, Lookback)
- Positions Long/Short

### 2. **Analyse des Grecques**
- **Delta** : sensibilité au prix du sous-jacent
- **Gamma** : convexité du delta
- **Vega** : sensibilité à la volatilité
- **Theta** : décroissance temporelle
- **Rho** : sensibilité aux taux d'intérêt
- Calculs pour chaque modèle
- Visualisations interactives

### 3. **Volatilité**
- Surfaces de volatilité implicite
- Simulation de volatilité (Monte Carlo)
- Volatilité historique vs implicite
- Smiles de volatilité

### 4. **Données de Marché**
- Intégration **Yahoo Finance** pour données réelles
- Support de 10 tickers majeurs (AAPL, MSFT, GOOGL, AMZN, TSLA, etc.)
- Cotations en temps réel (ou presque)

### 5. **Produits Structurés**
- Pricing de produits structurés complexes
- Décomposition des flux de trésorerie

### 6. **Revenu Fixe**
- Pricing d'obligations
- Valuation de swaps
- Futures sur obligations
- Contrats Forward Rate Agreement (FRA)
- Caps et Floors

### 7. **Portfolio**
- Gestion d'un portefeuille personnalisé
- Analyse de la diversification
- Calcul du P&L
- Métriques de risque

---

## 🏗 Architecture

```
AssetPricing/
├── app.py                      # Application principale Streamlit
├── utils.py                    # Utilitaires globaux
│
├── Models/                     # Moteurs de pricing
│   ├── models.py              # Classe de base abstraite
│   ├── blackscholes.py        # Modèle Black-Scholes
│   ├── heston.py              # Modèle Heston (volatilité stochastique)
│   ├── gammavariance.py       # Modèle Variance Gamma
│   ├── mertonjump.py          # Modèle Merton Jump-Diffusion
│   ├── bachelier.py           # Modèle Bachelier (taux)
│   └── treemodel.py           # Modèle Arbre Trinomial
│
├── functions/                 # Logique métier et calculs
│   ├── parameters_function.py # Gestion des paramètres et payoffs
│   ├── pricing_function.py    # Orchestration du pricing
│   ├── greeks_function.py     # Calcul des grecques
│   ├── greeks_bs_function.py  # Grecques spécifiques Black-Scholes
│   ├── greeks_heston_function.py
│   ├── greeks_gamma_variance_function.py
│   ├── vol_function.py        # Volatilité implicite et historique
│   ├── vol_simulation_function.py  # Simulation de volatilité
│   ├── bond_function.py       # Pricing d'obligations
│   ├── swap_function.py       # Pricing de swaps
│   ├── fra_future_function.py # FRA et Futures
│   ├── capfloor_function.py   # Caps et Floors
│   ├── structured_function.py # Produits structurés
│   ├── portfolio_function.py  # Gestion de portefeuille
│   ├── treepricing.py        # Pricing par arbre trinomial
│   ├── data_function.py       # Récupération de données
│   ├── model_explanations.py  # Explications pédagogiques
│   └── hull_white_function.py # Modèle Hull-White (taux)
│
├── views/                     # Interface utilisateur (pages Streamlit)
│   ├── accueil.py            # Page d'accueil
│   ├── parametre.py          # Configuration des paramètres
│   ├── pricing.py            # Module de tarification
│   ├── greeks.py             # Analyse des grecques
│   ├── vol.py                # Surfaces de volatilité
│   ├── vol_simulation.py     # Simulation de volatilité
│   ├── data.py               # Données de marché
│   ├── bond_swap_futures.py  # Revenu fixe
│   ├── structured.py         # Produits structurés
│   └── portfolio.py          # Gestion de portefeuille
│
├── requirements.txt           # Dépendances Python
└── README.md                  # Ce fichier
```

---

## 🔧 Modèles implémentés

### 1. **Black-Scholes (1973)**
**Caractéristiques :**
- Volatilité constante
- Pas de sauts
- Solution analytique rapide
- **Meilleur pour :** Options vanilles européennes

**Formule du prix :**
$$C = S_0 e^{-qT} N(d_1) - K e^{-rT} N(d_2)$$

où :
$$d_1 = \frac{\ln(S_0/K) + (r - q + \sigma^2/2)T}{\sigma\sqrt{T}}$$
$$d_2 = d_1 - \sigma\sqrt{T}$$

**Paramètres :** S, K, r, T, σ, q

---

### 2. **Heston (1993)**
**Caractéristiques :**
- Volatilité stochastique
- Correlation entre prix et volatilité
- Capture le smile de volatilité
- Solution semi-analytique (Fourier, Laguerre)
- **Meilleur pour :** Options avec volatilité variable

**Paramètres additionnels :**
- `v0` : volatilité initiale
- `kappa` : vitesse de reversion
- `theta` : volatilité long-terme
- `sigma_v` : volatilité de la volatilité
- `rho` : corrélation prix-volatilité

**EDP du modèle :**
$$dS = rS dt + \sqrt{v} S dW_1$$
$$dv = \kappa(\theta - v) dt + \sigma_v \sqrt{v} dW_2$$

---

### 3. **Variance Gamma**
**Caractéristiques :**
- Processus à sauts purs (Lévy)
- Captures l'asymétrie et l'aplatissement (kurtosis)
- Reproduit les smiles de volatilité empiriques
- **Meilleur pour :** Modéliser les crashes et l'asymétrie

**Paramètres :**
- `sigma` : volatilité continue
- `nu` : paramètre de variance des sauts
- `theta` : drift des sauts

---

### 4. **Merton Jump-Diffusion**
**Caractéristiques :**
- Mouvement brownien + processus de Poisson
- Modélise les discontinuités (sauts)
- Réaliste pour les marchés avec chocs
- **Meilleur pour :** Marchés stressés, avec évènements rares

**Paramètres :**
- Paramètres Black-Scholes +
- `lambda` : intensité des sauts
- `mu_j` : moyenne des sauts
- `sigma_j` : volatilité des sauts

---

### 5. **Bachelier (Modèle Normal)**
**Caractéristiques :**
- Suppose des taux d'intérêt normalement distribués
- Approprié pour les taux bas/négatifs
- Utilisé par convention pour les obligations
- **Meilleur pour :** Produits de taux

**Formule :**
$$C = (F - K) N(d) + \sigma\sqrt{T} n(d)$$

---

### 6. **Arbre Trinomial**
**Caractéristiques :**
- Méthode numérique de discrétisation
- Flexibilité pour options américaines
- Arbres recombinants efficaces
- **Meilleur pour :** Options avec exercice anticipé

**Avantages :**
- Options américaines
- Termes structures complexes
- Dividendes discrets

---

## 💾 Installation et setup

### Prérequis
- Python 3.8+
- pip ou conda

### Installation

1. **Cloner le repository :**
```bash
git clone <repository_url>
cd AssetPricing
```

2. **Créer un environnement virtuel :**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

3. **Installer les dépendances :**
```bash
pip install -r requirements.txt
```

### Dépendances principales
```
streamlit>=1.0.0          # Interface web
numpy>=1.20.0             # Calculs numériques
scipy>=1.5.0              # Optimisation, statistiques
matplotlib>=3.3.0         # Graphiques
yfinance>=0.1.70          # Données de marché
pandas>=1.1.0             # Manipulation de données
plotly>=5.0.0             # Graphiques interactifs
```

---

## 🚀 Utilisation

### Lancer l'application

```bash
streamlit run app.py
```

L'application s'ouvre automatiquement dans votre navigateur (généralement `http://localhost:8501`)

---

## 📖 Guide détaillé des modules

### 🏠 **Home (Accueil)**
Page d'introduction avec :
- Présentation générale
- Navigation vers les différents modules
- Explication des concepts

---

### 📊 **Derivatives (Dérivés)**

#### **1. Parameters & Payoff**
Configurez les paramètres de votre option :
- **Ticker** : Sélectionnez un sous-jacent (AAPL, MSFT, etc.)
- **Position** : Long ou Short
- **Type d'option** : Vanille (Call/Put) ou Exotique (Asian/Lookback)
- **Paramètres numériques** :
  - `K` : Strike (prix d'exercice)
  - `T` : Maturité (en années)
  - `r` : Taux sans risque
  - `σ` : Volatilité implicite
  - `q` : Dividende continu

**Affichage :** Graphique du payoff à maturité

---

#### **2. Pricing**
Tarificez votre option avec différents modèles :
- Sélectionnez le modèle (Black-Scholes, Heston, etc.)
- Configurez les paramètres spécifiques au modèle
- Obtenez le prix en temps réel
- Comparaison entre modèles

**Outputs :**
- Prix
- Sensibilité au prix
- Détails du calcul

---

#### **3. Greeks**
Analysez les risques de votre position :
- **Delta (Δ)** : Exposition au prix
- **Gamma (Γ)** : Convexité
- **Vega (ν)** : Exposition à la volatilité
- **Theta (Θ)** : Décroissance temporelle
- **Rho (ρ)** : Exposition aux taux

**Visualisations :**
- Courbes des grecques
- Heatmaps de sensibilité
- Surface 3D (prix vs sous-jacent vs temps)

---

### 📈 **Market (Marché)**

#### **1. Data**
Récupérez et visualisez les données de marché :
- Historiques de prix (1j, 1m, 3m, 1y, 5y)
- Volumes de trading
- Retours calculés
- Volatilité historique

---

#### **2. Implied Volatility Surface**
Construisez la surface de volatilité implicite :
- Varie par strike et maturité
- Capture le smile/smirk de volatilité
- Calibre les modèles sur les données réelles

**Paramètres :**
- Grille de strikes : [K-20%, K, K+20%]
- Grille de maturités : [3M, 6M, 1Y, 2Y, 5Y]

---

#### **3. Volatility Simulation**
Simulez des trajectoires de volatilité :
- Monte Carlo pour les modèles stochastiques
- Heston, Variance Gamma, etc.
- Analyse des chemins (histogrammes, quantiles)

---

### 💰 **Fixed Income (Revenu Fixe)**

#### **Bond & Swap**
- **Bonds (Obligations)** :
  - Pricing par formule de valeur présente
  - Duration et convexité
  - Courbe de taux
  - Spreads de crédit

- **Swaps** :
  - Swaps vanilles (IRS : Interest Rate Swap)
  - Pricing par différence de valeur présente
  - Courbe de taux nulle

- **Futures** :
  - Futures sur obligations
  - Contrats FRA (Forward Rate Agreement)

- **Caps & Floors** :
  - Pricing par Black formule
  - Volatilité implicite

---

### 🏢 **Structured Products**
Pricing de produits complexes :
- Reverse convertibles
- Callables
- Autocallables
- Stochastiques (dépend du modèle)

---

### 💼 **Portfolio (Portefeuille)**
Créez et analysez votre portefeuille :
- Ajout/suppression de positions
- Calcul du P&L total
- Greeks du portefeuille (agrégés)
- Analyse de la diversification
- Historique des opérations

---

## 📁 Structure des fichiers

### Models/
Chaque modèle hérite de la classe `Model` et implémente :
```python
class Model(ABC):
    @abstractmethod
    def price(self, **kwargs):
        """Retourne le prix"""
        pass
```

**Méthodes communes :**
- `price()` : Prix de l'option
- `delta()`, `gamma()`, `vega()`, etc. : Grecques
- `implied_volatility()` : Volatilité implicite

---

### Functions/
Logique métier groupée par thème :

**parameters_function.py**
```python
class MarketDataFetcher      # Récupère données Yahoo Finance
class OptionParameters       # Encapsule les paramètres d'option
class PayoffCalculator       # Calcule les payoffs
class PayoffPlotter         # Visualise les payoffs
```

**pricing_function.py**
```python
MODELS = {
    "Black-Scholes": BlackScholes,
    "Heston": HestonModel,
    "Gamma Variance": VarianceGamma,
    "Trinomial Tree": TrinomialTree
}

def price_option(model_name, params):
    """Interface unifiée de pricing"""
```

**greeks_function.py**
```python
class GreeksCalculator:
    @staticmethod
    def calculate_greeks(model, params):
        """Calcule tous les grecques"""
```

---

### Views/
Chaque vue est une page Streamlit indépendante avec :
```python
def app():
    """Point d'entrée de la page"""
    # Récupère paramètres depuis session_state
    # Affiche interface
    # Met à jour session_state
```

**Gestion de session :**
- `st.session_state` persiste les paramètres entre pages
- Évite les re-calculs inutiles
- Maintient l'historique des opérations

---

## 🔍 Flux de données

```
app.py
  ├─> Définit navigation
  ├─> Gère session_state
  └─> Route vers views/

views/*.py
  ├─> Affiche interface Streamlit
  ├─> Récupère input utilisateur
  ├─> Appelle functions/
  └─> Met à jour session_state

functions/*.py
  ├─> Logique métier
  ├─> Transformations de données
  └─> Appelle Models/

Models/*.py
  ├─> Calculs mathématiques
  ├─> Pricing & grecques
  └─> Retourne résultats
```

---

## 🧮 Formules clés

### Delta
$$\Delta = \frac{\partial C}{\partial S}$$

### Gamma
$$\Gamma = \frac{\partial^2 C}{\partial S^2}$$

### Vega
$$\nu = \frac{\partial C}{\partial \sigma}$$

### Theta
$$\Theta = \frac{\partial C}{\partial t}$$

### Rho
$$\rho = \frac{\partial C}{\partial r}$$

---

## 🔗 Dépendances externes

**Yahoo Finance**
- Récupération de prix réels
- API gratuite via `yfinance`
- Limitations : délai de 15 min

**Scipy**
- Optimisation (calibration)
- Distribution normale (N, n)
- Intégration numérique

**Numpy/Pandas**
- Calculs vectorisés rapides
- Manipulation de séries temporelles

---

## 💡 Bonnes pratiques

### Pour les utilisateurs
1. **Calibrez vos modèles** sur les prix de marché réels
2. **Comparez les résultats** entre modèles
3. **Analysez la sensibilité** des grecques
4. **Documentez vos hypothèses** (volatilité, corrélation, etc.)

### Pour les développeurs
1. Toujours hériter de `Model` pour ajouter un nouveau modèle
2. Implémenter au minimum `price()`
3. Ajouter les grecques si possible (analytique)
4. Tester avec des cas connus (p.ex. Black-Scholes)
5. Utiliser `@st.cache_data` pour les calculs lourds

---

## 🐛 Debugging

**Erreurs courantes :**

| Erreur | Cause | Solution |
|--------|-------|----------|
| `ModuleNotFoundError` | Dépendances manquantes | Relancer `pip install -r requirements.txt` |
| `YFinance error` | Internet down ou API down | Vérifier connexion, réessayer plus tard |
| `ValueError: parameters mismatch` | Paramètres incorrects | Vérifier les paramètres requis du modèle |
| `Volatilité négative` | Input utilisateur erroné | σ doit être > 0 |

---

## 📚 Références académiques

- **Black, F., Scholes, M.** (1973) - *The pricing of options and corporate liabilities*
- **Heston, S. L.** (1993) - *A closed-form solution for options with stochastic volatility*
- **Merton, R. C.** (1976) - *Option pricing when underlying stock returns are discontinuous*
- **Madan, D. B., Carr, P. P., Chang, E. C.** (1998) - *The variance gamma process and option pricing*

---

## 📝 Licence

Voir le fichier [LICENSE](LICENSE)

---

## 👨‍💻 Auteurs

Asset Pricing Application - Équipe Quantitative

**Version** : 1.0.0  
**Dernière mise à jour** : Janvier 2026

---

## 📧 Support

Pour des questions, bugs ou suggestions, veuillez ouvrir une issue sur le repository.

---

**Disclaimer** : Cette application est à titre éducatif. Les utilisateurs sont responsables de valider tous les calculs avant utilisation dans un contexte commercial.