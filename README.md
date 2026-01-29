# Asset Pricing Application

Our project allows pricing, visualization, and asset management support through various tools. This application runs using Streamlit and Python.

Among the available features, you can:

- Price vanilla and exotic options  
- Analyze the Greeks (Delta, Gamma, Vega, Theta, Rho)  
- Visualize implied volatility surfaces  
- Simulate volatility scenarios  
- Evaluate bonds and swaps  
- Create and analyze portfolios  

---

## Table of Contents

- [Main Features](#-main-features)  
- [Architecture](#-architecture)  
- [Implemented Models](#-implemented-models)  
- [Installation and Setup](#-installation-and-setup)  

---

## Main Menus and Capabilities

Here we list each page of our application along with all the functionalities they offer.

### 1. **Option Pricing**
- Black-Scholes Model (analytical) 
- Bachelier
- Heston Model (stochastic volatility)  
- Variance Gamma (jumps and asymmetry)  
- Trinomial Tree (discretization)  
- Merton Jump
- Support for vanilla options (European Call/Put)  
- Support for exotic options (Asian, Lookback)  
- Long/Short positions  

### 2. **Greeks Analysis**
- **Delta**: sensitivity to the underlying price  
- **Gamma**: convexity of delta  
- **Vega**: sensitivity to volatility  
- **Theta**: time decay  
- **Rho**: sensitivity to interest rates  
- Calculations for each model  
- Interactive visualizations  

### 3. **Volatility**
- Implied volatility surfaces  
- Volatility simulation (Monte Carlo)  
- Historical vs implied volatility  
- 3D volatility smiles  

### 4. **Market Data**
- Support for 10 major tickers (AAPL, MSFT, GOOGL, AMZN, TSLA, etc.)  
- Real-time quotes  

### 5. **Structured Products**
- Pricing of complex structured products  
- Cash flow decomposition  

### 6. **Fixed Income**
- Bond pricing  
- Swap valuation  
- Bond futures  
- Forward Rate Agreements (FRA)  
- Caps and Floors  
- Bond and Swap valuation with Hull-White and Monte Carlo simulation

### 7. **Portfolio**
- Manage a custom portfolio  
- Diversification analysis  
- P&L calculation  
- Risk metrics  

---

## Architecture

The `views` folder handles user interaction and then calls the necessary functions and models to execute the user’s requests.

Required packages are listed in `requirements.txt`.

---

## Implemented Models

Here we list the different models available in our application.

### 1. **Black-Scholes (1973)**
**Features:**  
- Constant volatility  
- No jumps  
- Fast analytical solution  
- **Best for:** European vanilla options  

**Pricing formula:**  
$$C = S_0 e^{-qT} N(d_1) - K e^{-rT} N(d_2)$$

where:  
$$d_1 = \frac{\ln(S_0/K) + (r - q + \sigma^2/2)T}{\sigma\sqrt{T}}$$  
$$d_2 = d_1 - \sigma\sqrt{T}$$

**Parameters:** S, K, r, T, σ, q  

---

### 2. **Heston (1993)**
**Features:**  
- Stochastic volatility  
- Correlation between price and volatility  
- Captures the volatility smile  
- Semi-analytical solution (Fourier, Laguerre)  
- **Best for:** Options with variable volatility  

**Additional parameters:**  
- `v0`: initial volatility  
- `kappa`: speed of mean reversion  
- `theta`: long-term volatility  
- `sigma_v`: volatility of volatility  
- `rho`: price-volatility correlation  

**Model SDEs:**  
$$dS = rS dt + \sqrt{v} S dW_1$$  
$$dv = \kappa(\theta - v) dt + \sigma_v \sqrt{v} dW_2$$  

---

### 3. **Variance Gamma**
**Features:**  
- Pure jump process (Lévy)  
- Captures skewness and kurtosis  
- Reproduces empirical volatility smiles  
- **Best for:** Modeling crashes and asymmetry  

**Parameters:**  
- `sigma`: continuous volatility  
- `nu`: jump variance parameter  
- `theta`: jump drift  

---

### 4. **Merton Jump-Diffusion**
**Features:**  
- Brownian motion + Poisson process  
- Models discontinuities (jumps)  
- Realistic for markets with shocks  
- **Best for:** Stressed markets with rare events  

**Parameters:**  
- Black-Scholes parameters +  
- `lambda`: jump intensity  
- `mu_j`: jump mean  
- `sigma_j`: jump volatility  

---

### 5. **Bachelier (Normal Model)**
**Features:**  
- Assumes normally distributed interest rates  
- Suitable for low/negative rates  
- Commonly used for bonds  
- **Best for:** Rate products  

**Formula:**  
$$C = (F - K) N(d) + \sigma\sqrt{T} n(d)$$  

---

### 6. **Trinomial Tree**
**Features:**  
- Numerical discretization method  
- Flexible for American options  
- Efficient recombining trees  
- **Best for:** Options with early exercise  

**Advantages:**  
- American options  
- Complex term structures  
- Discrete dividends  

---

## 💾 Installation and Setup

### Prerequisites
- Python 3.8+  
- Install all packages listed in `requirements.txt`  

To run the application, execute in the terminal:

```bash
streamlit run app.py
```