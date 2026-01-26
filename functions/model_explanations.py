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

""",
    "Trinomial Tree": r"""
### Trinomial Tree Model

The trinomial tree model is a **discrete-time lattice method** used to price options
by approximating the continuous-time dynamics of the underlying asset.

At each time step $\Delta t$, the asset price can move to **three possible states**:
- Up
- Middle
- Down

**Price evolution:**

$$
S_{t+\Delta t} =
\begin{cases}
S_t \cdot u      & \text{with probability } p_u \\
S_t              & \text{with probability } p_m \\
S_t / u          & \text{with probability } p_d
\end{cases}
$$

where the up factor is defined as:

$$
u = e^{\sigma \sqrt{3 \Delta t}}
$$

---

### Risk-neutral probabilities

The probabilities are chosen so that the tree matches the **first two moments**
(mean and variance) of the Black–Scholes model under the risk-neutral measure.

$$
\begin{aligned}
p_u &= \frac{1}{6}
      + \frac{(r - q - \tfrac{1}{2}\sigma^2)\sqrt{\Delta t}}
             {2\sigma\sqrt{3}} \\
p_m &= \frac{2}{3} \\
p_d &= 1 - p_u - p_m
\end{aligned}
$$

These probabilities ensure:
- No-arbitrage
- Correct drift under the risk-free measure
- Convergence to Black–Scholes as $\Delta t \to 0$

---

### Option pricing principle

The option price is obtained by **backward induction**:

1. Compute the payoff at maturity
2. Discount the expected value at each node
3. Apply early exercise if the option is American

$$
V_t = e^{-r \Delta t}
\mathbb{E}^{\mathbb{Q}}\!\left[ V_{t+\Delta t} \mid \mathcal{F}_t \right]
$$


"""

}
