### A Pluto.jl notebook ###
# v0.19.5

using Markdown
using InteractiveUtils

# ╔═╡ 46de748a-9a7c-4066-b594-895314149429
md"""
# Dissertation Revision Notes
## Week 1
### Frequentist vs Bayesian
Sampling distribution is a key concept behind all frequentist inference. Sampling distribution is the distribution of the estimator. Frequentist assumes data is random, Bayesian assumes data is fixed. If data is random, then MLE is random as well. MLE is a function of the data. All random variables has a distribution. 

The question is: what is the distribution of the estimator?

If it is a distribution, then we can calculate the mean and variance.

Frequentist assumes that the data can be continuously sampled, if your data is time series then you cannot go back and sample it again.

Bayesian inference assumes the other way around, data is fixed, and unknown parameter λ is a random variable.
"""

# ╔═╡ 9ebada59-0389-49e2-a8c5-2f118fd37470
md"""
### Poisson MLE

Poisson probability mass function (p.m.f) = $P(y=k) = \frac{λ^ke^{-λ}}{k!} ≡ exp(-λ) \frac{1}{k!}λ^k$

Likelihood function. n observations are independent so the likelihood function is a product of each p.m.f.

$l(λ) = \prod_{i=1}^{n} exp(-λ)\frac{1}{x_i!}λ^{x_i}$

Log likelihood:

$L(λ) = ln(\prod_{i=1}^{n} exp(-λ)\frac{1}{x_i!}λ^{x_i})$

Log of products becomes sum of logs:

$L(λ) = \sum_{i=1}^{n} ln(exp(-λ)\frac{1}{x_i!}λ^{x_i})$

"""

# ╔═╡ d4897a9d-3971-4396-90a9-1bf64b68523f
md"""
### Normal Linear Regression Probabilistic Model
Normal Linear Regression Model (NLRM) is a linear regression model where the error term is Gaussian distributed. 

The linear regression model is:

$y_i = x_i β_0 + ϵ_i$

This can be written in matrix form:

$y = Xβ_0 + ϵ$

The likelihood, is the probability of the data given the model. $P(D|θ)$

In the probabilistic sense for linear regression, each $y$ is Gaussian distributed. So the line is the mean, and the variance is the error from the line.

$y_i = N(x_i^Tθ, σ^2) = x_i^Tθ + N(0, σ^2)$

The RHS is shifting the Gaussian by the mean.

$p(y|X,θ,σ) = \prod_{i=1}^{n}p(y_i|x_i,θ,σ)$
$= \prod_{i=1}^{n}(2πσ^2)^{-\frac{1}{2}}e^{-\frac{1}{2σ^2}(y_i-x_i^Tθ)^2}$
$= (2πσ^2)^{-\frac{n}{2}}e^{-\frac{1}{2σ^2}\sum_{i=1}^{n}(y_i-x_i^Tθ)^2}$

Convert to matrix notation:

$= (2πσ^2)^{-\frac{n}{2}}e^{-\frac{1}{2σ^2}(y-Xθ)^T(y-Xθ)}$

N.B. Minimising a cost function is equivalent to maximising a probability or likelihood.
"""

# ╔═╡ e1e693bf-0115-4d93-8cdb-0ab9f6ed76ae
md"""
### MLE of Normal Linear Regression Model
[Source](https://www.statlect.com/fundamentals-of-statistics/linear-regression-maximum-likelihood)

MLE:
1. Get the likelihood
2. Turn it into log-likelihood. Why? Easier to work in log space, it gets rid of the exponent.
3. Take derivative
4. Maximise probability of generating training data $y$ given parameters $(θ, σ)$

$p(y|X,θ,σ) = (2πσ^2)^{-\frac{n}{2}}e^{-\frac{1}{2σ^2}(y-Xθ)^T(y-Xθ)}$
$l(θ) = log(p(y|X,θ,σ)) = -\frac{n}{2}ln(2πσ^2)-\frac{1}{2σ^2}(y-Xθ)^T(y-Xθ)$

Derivative of log likelihood w.r.t θ:

$\frac{∂l(θ)}{∂θ} = -\frac{1}{2σ^2}(-2X^Ty + X^TXθ)$

Set to 0

$0 = -\frac{1}{2σ^2}(-2X^Ty + X^TXθ)$

$θ̂_{MLE} = (X^TX)^{-1}X^Ty$

The MLE of σ is:

$σ^2 = \frac{1}{n}(y - Xθ)^T(y - Xθ) = \frac{1}{n}\sum_{i=1}^{n}(y_i-x_iθ)^2$
"""

# ╔═╡ bba4bb93-16d5-481d-8fea-933280b12fb3
md"""
### MLE of Bernoulli Distribution
[Source](https://youtu.be/2sORKoeX8g8)

Likelihood = $\prod_{i=1}^{n}p^{x_i}(1-p)^{1-x_i}$

Log likelihood = $log(\prod_{i=1}^{n}p^{x_i}(1-p)^{1-x_i})$ 

= $\sum_{i=1}^{n}log(p^{x_i}(1-p)^{1-x_i})$

= $\sum_{i=1}^{n}{x_i}log(p) + (1-x_i) log(1-p))$ 

= $log(p)\sum_{i=1}^{n}{x_i} + log(1-p) \sum_{i=1}^{n}(1-x_i))$ 

Apparently the sums above can be removed using $x̄$, I am not sure what $x̄$ is?

$l = Nx̄⋅log(p) + N(1-x̄)⋅log(1-p)))$ 

$\frac{∂l}{∂p} = \frac{Nx̄}{p} - \frac{N(1-x̄)}{1-p}$

The second term is a negative because of $1-p$, we have to consider the differential of the inner function $1-p$ which is $-1$, hence the negative term.

$\frac{Nx̄}{p} - \frac{N(1-x̄)}{1-p} = 0$

$\frac{Nx̄}{p} = \frac{N(1-x̄)}{1-p}$

Cancel the N on both sides.

$\frac{x̄}{p} = \frac{1-x̄}{1-p}$

Cross multiply through.

$x̄(1-p) = p(1-x̄)$

$x̄ - px̄ = p - px̄$

Cancel $px̄$ on both sides

$p = x̄$

The parameter is then sample mean.
"""

# ╔═╡ d91fdd98-0542-4026-b4c7-f0dbc50bb3e6
md"""
### Logistic Regression
[Basics](https://youtu.be/yhogDBEa0uQ)

[MLE](https://youtu.be/TM1lijyQnaI)

Why do we use Logistic Regression at all? Why not Linear Regression?
- Because Linear Regression predicts the value itself, Logistic Regression predicts the probability which has a range restricted between [0,1]
- The line in LinReg suggests any value from $-∞$ to $∞$ can occur

Logistic regression assumes that the response variables (dependent variables) follow a binomial distribution. Sometimes logistic regression is referred to as "binomial logistic regression".

The binomial distribution p.m.f is $P(k) = {n \choose k} p^k (1-p)^{n-k}$
"""

# ╔═╡ 0f153b93-84f0-4e4e-909c-fa888d4121c8
md"""
### The Logistic Sigmoid Function

$σ(x) = \frac{1}{1+e^{-z}}$ Where $z$ is the linear combination of inputs and weights.

Likelihood function of a Bernoulli distribution:

$l(θ) = P(Y|X;θ) = \prod_{i-1}^{n}σ(x_i)^{y_i}(1-σ(x_i))^{1-y_i}$

The above probability reads as: "The probability of Y given input X and parameter θ".

$L(θ) = \sum_{i=1}^{n} y_i log(σ(θ^Tx̄_i)) + (1-y_i) log (1-σ(θ^Tx̄_i))$
"""

# ╔═╡ 3b1ec801-e635-4a50-b524-706b3d54e0ea
md"""
### MLE of Logistic Regression (version 2)
[Source](https://www.statlect.com/fundamentals-of-statistics/logistic-model-maximum-likelihood)

The log likelihood, is the natural log of the likelihood function.

$l(β;y,X) = \ln(L(β;y,X))$

$= \ln(\prod_{i=1}^{n}[σ(x_iβ)]^{y_i}[1-σ(x_iβ)]^{1-y_i})$

Log of products, is sum of logs.

$= \sum_{i=1}^{n}[y_i\ln(σ(x_iβ)) + (1-y_i) \ln(1-σ(x_iβ))]$

Subtitute the sigmoid function in.

$= \sum_{i=1}^{n}[y_i\ln(\frac{1}{1 + exp(-x_iβ)}) + (1-y_i) \ln(1-\frac{1}{1 + exp(-x_iβ)})]$

Substitute 1 on the last term with $\frac{1 + exp(-x_iβ)}{1 + exp(-x_iβ)}$

$= \sum_{i=1}^{n}[y_i\ln(\frac{1}{1 + exp(-x_iβ)}) + (1-y_i) \ln(\frac{{1 + exp(-x_iβ)} - 1}{1 + exp(-x_iβ)})]$

Eliminate 1-1 = 0

$= \sum_{i=1}^{n}[y_i\ln(\frac{1}{1 + exp(-x_iβ)}) + (1-y_i) \ln(\frac{{exp(-x_iβ)}}{1 + exp(-x_iβ)})]$

Perform some factoring and rearrange terms

$= \sum_{i=1}^{n}\left[\ln(\frac{exp(-x_iβ)}{1 + exp(-x_iβ)}) + y_i\left(\ln(\frac{1}{1 + exp(-x_iβ)}) - \ln(\frac{exp(-x_iβ)}{1 + exp(-x_iβ)})\right)\right]$

Introduce $\frac{exp(x_iβ)}{exp(x_iβ}$ to the first term. And combine summation of logs to product of larger log. Since the last term is a negative, the top and bottom terms are flipped to make it a product.

$= \sum_{i=1}^{n}\left[\ln\left(\frac{exp(-x_iβ)}{1 + exp(-x_iβ)} \frac{exp(x_iβ)}{exp(x_iβ}\right) + y_i\left(\ln(\frac{1}{1 + exp(-x_iβ)}\frac{1 + exp(-x_iβ)}{exp(-x_iβ)})\right)\right]$

Eliminate $1 + exp(-x_iβ)$ from last term. Power rules of exponents will apply to the first term.

$= \sum_{i=1}^{n}\left[\ln(\frac{1}{1 + exp(x_iβ)}) + y_i(\ln(\frac{1}{exp(-x_iβ)}))\right]$

Apply log rules:

$= \sum_{i=1}^{n}\left[\ln(1) - \ln(1+exp(x_iβ)) + y_i(\ln(1) - \ln(exp(-x_iβ)))\right]$

Simplify: $\ln(1) = 0$

$= \sum_{i=1}^{n}\left[- \ln(1+exp(x_iβ)) + y_ix_iβ)\right]$
"""

# ╔═╡ dbb4ef23-6c48-4397-9a8c-a4cebf1556e9
md"""
### Newton's Method
$X_{n+1} = X_n - \frac{f(X_n)}{f'(X_n)}$
"""

# ╔═╡ a0597c88-13e8-4712-bf93-7638b2bdab87
md"""
# Notes
## Linear Algebra
- A diagonal matrix is one in which all values other than in the diagonal are zeros.
- Sum of squares: $\sum_{i=1}^{n}x_i^2 = x^Tx$.
- A design matrix is a matrix containing data about multiple characteristics of several individuals or objects. Each row corresponds to an individual and each column to a characteristic.
"""

# ╔═╡ 663c4d49-4e8c-458b-9e7d-030f7afda72f
md"""
## Univariate Gaussian Distribution
Lecture: https://youtu.be/voN8omBe2r4
- Width is controlled by σ
- The bigger the σ, the wider it is
- σ is the standard deviation
- The variance is $σ^2$
- μ is the expected value or the mean
- Area under the curve equals to 1, as it is a valid probability distribution

If we have a Gaussian distribution, then we can draw samples.

$x ∼ N(μ, σ^2)$

The tilde ∼ symbol means that we can simulate/sample/generate $x$ from a Gaussian distribution.

A gaussian is an exponential function, so at the tails the probabilities vanish at an exponential rate.
"""

# ╔═╡ 41e57d12-ccc0-4865-8376-e5a391f707f0
md"""
### Multivariate Gaussian Distribution

$p(y) = |2πΣ|^{-\frac{1}{2}}e^{-\frac{1}{2}(y-μ)^TΣ^{-1}(y-μ)}$

Why does it use determinant?

Why is the following equivalent?

$|2πΣ| ≡ (2πσ^2)$
"""

# ╔═╡ 6b3aa51e-ac39-4089-ad9a-0373f798b4c0
md"""
### How does a computer generate a Gaussian distributed random number?
1. Sample a random number from a uniform distribution
2. Project onto cumulative of Gaussian that looks like a sigmoid function
3. Then return the $x$ value
### Sampling from a multivariate Gaussian
1. Sample from $y ∼ N(0,1)$ Gaussian distribution with mean of 0 and variance of 1.
2. To get $x ∼ N(μ, σ^2)$ we need to $x ∼ μ + σ N(0,1)$. Why is it just a multiplication of σ?
"""

# ╔═╡ 2a194cd2-feab-4aab-ab59-5f17d751e533
md"""
### Mathematical Identities
#### Logs

Product: 

$log(ab) = log(a) + log(b)$

Power:

$log(a^b) = b⋅log(a)$

Quotient:

$log(\frac{x}{y}) = log(x) - log(y)$

Log of e:

$log(e) = 1$

Log of one:

$log(1) = 0$

Log of reciprocal:

$log(\frac{1}{x}) = -log(x)$

#### Derivatives

$y = ln(x)$
$\frac{dy}{dx} = \frac{1}{x}$

#### Exponents

Product:

$a^m * a^n = a^{m+n}$

Quotient:

$\frac{a^m}{a^n} = a^{m-n}$

Power:

$(a^m)^n = a^{(m*n)}$

Negative exponent:

$x^{-n} = \frac{1}{x^n}$

Zero exponent:

$x^0 = 1$

#### Differentiation

$\frac{dσ(x)}{dx} = σ(1-σ)$

$\frac{∂\textbf{a}^Tx}{∂x} = \textbf{a}^T$
"""

# ╔═╡ df5e0adf-fe1f-4b35-89f4-57347f2cd15b
md"""
## Things to learn
- Credible interval
- Hypothesis testing, null hypothesis
- What does the rank of a matrix mean? What is full-rank? Why is it useful?
- What is standard error?
- Hessian
- Central limit theory
"""

# ╔═╡ c77e932c-ed70-4016-a42d-25b772605a78
md"""
## Julia Libraries To Learn
- Zygote.jl
- Flux.jl
- LinearAlgebra.jl
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.1"
manifest_format = "2.0"

[deps]
"""

# ╔═╡ Cell order:
# ╟─46de748a-9a7c-4066-b594-895314149429
# ╟─9ebada59-0389-49e2-a8c5-2f118fd37470
# ╟─d4897a9d-3971-4396-90a9-1bf64b68523f
# ╟─e1e693bf-0115-4d93-8cdb-0ab9f6ed76ae
# ╟─bba4bb93-16d5-481d-8fea-933280b12fb3
# ╠═d91fdd98-0542-4026-b4c7-f0dbc50bb3e6
# ╟─0f153b93-84f0-4e4e-909c-fa888d4121c8
# ╟─3b1ec801-e635-4a50-b524-706b3d54e0ea
# ╟─dbb4ef23-6c48-4397-9a8c-a4cebf1556e9
# ╟─a0597c88-13e8-4712-bf93-7638b2bdab87
# ╟─663c4d49-4e8c-458b-9e7d-030f7afda72f
# ╟─41e57d12-ccc0-4865-8376-e5a391f707f0
# ╟─6b3aa51e-ac39-4089-ad9a-0373f798b4c0
# ╠═2a194cd2-feab-4aab-ab59-5f17d751e533
# ╟─df5e0adf-fe1f-4b35-89f4-57347f2cd15b
# ╟─c77e932c-ed70-4016-a42d-25b772605a78
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
