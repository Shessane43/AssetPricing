MODEL_EXPLANATIONS = {

    "Black-Scholes": r"""
### Black–Scholes Model

The Black–Scholes model assumes that the underlying asset price follows a **geometric Brownian motion**
with constant volatility and interest rate.

**Asset price dynamics:**

$$
\frac{dS_t}{S_t} = (r - q)\,dt + \sigma\, dW_t
$$

where:
- $S_t$ : asset price  
- $r$ : risk-free interest rate  
- $q$ : dividend yield  
- $\sigma$ : constant volatility  
- $W_t$ : standard Brownian motion  

**Key assumptions:**
- Constant volatility and interest rate  
- Log-normal asset prices  
- No jumps  
- Frictionless markets  

**Strengths:**
- Closed-form pricing formulas  
- Fast computation  
- Easy interpretation of Greeks  

**Limitations:**
- No volatility smile  
- Unrealistic tail behavior  
- Poor fit for equity options at short maturities  
""",

    "Heston": r"""
### Heston Stochastic Volatility Model

The Heston model extends Black–Scholes by introducing **stochastic variance**
with mean reversion and correlation (leverage effect).

**Asset price dynamics:**

$$
\frac{dS_t}{S_t} = (r - q)\,dt + \sqrt{v_t}\, dW_t^S
$$

**Variance dynamics:**

$$
dv_t = \kappa(\theta - v_t)\,dt + \sigma_v \sqrt{v_t}\, dW_t^v
$$

with correlation:
$$
dW_t^S\, dW_t^v = \rho\, dt
$$

where:
- $v_t$ : instantaneous variance  
- $\kappa$ : speed of mean reversion  
- $\theta$ : long-term variance level  
- $\sigma_v$ : volatility of variance  
- $\rho$ : correlation (leverage effect)  

**Key features:**
- Endogenous volatility smile  
- Leverage effect  
- Mean-reverting volatility  

**Strengths:**
- Better fit to equity options  
- Realistic implied volatility surface  

**Limitations:**
- No jumps  
- Calibration can be unstable  
- More computationally expensive than BS  
""",

    "Gamma Variance": r"""
### Variance Gamma (VG) Model

The Variance Gamma model replaces Brownian motion with a **pure jump Lévy process**.
There is **no continuous diffusion component**.

**Log-price dynamics:**

$$
X_t = \theta\, G_t + \sigma\, W_{G_t}
$$

where:
- $X_t = \log(S_t)$  
- $W_t$ : Brownian motion  
- $G_t$ : Gamma subordinator (random time change)

**Equivalent asset price representation:**

$$
S_t = S_0 \exp(X_t)
$$

**Key parameters:**
- $\theta$ : skewness (asymmetry of returns)  
- $\nu$ : jump activity / kurtosis  
- $\sigma$ : scale (volatility of jumps)  

**Properties:**
- Heavy tails  
- Skewed return distributions  
- Infinite jump activity  
- No continuous diffusion  

**Strengths:**
- Excellent fit for short maturities  
- Captures skewness and kurtosis  
- Generates volatility smiles naturally  

**Limitations:**
- No stochastic volatility dynamics  
- Parameters are not directly interpretable as implied volatility  
- Calibration required for market consistency  
"""
}
