# <span style="font-size: 20px;">What is Logistic Regression?</span>

<span style="font-size: 14px;">Logistic regression is the fundamental algorithm for binary classification. Despite its name, it is a classification algorithm, not a regression algorithm. It models the probability that an input belongs to the positive class by passing a linear combination of features through the sigmoid function.</span>

<span style="font-size: 14px;">Given a dataset of</span> $n$ <span style="font-size: 14px;">samples, each with</span> $d$ <span style="font-size: 14px;">features:</span>

- <span style="font-size: 14px;">Feature matrix:</span> $X \in \mathbb{R}^{n \times d}$
- <span style="font-size: 14px;">Binary labels:</span> $y \in \{0, 1\}^n$

<span style="font-size: 14px;">The model predicts a probability:</span>

$$
\hat{y} = \sigma(Xw + b) = \frac{1}{1 + e^{-(Xw + b)}}
$$

<span style="font-size: 14px;">where</span> $w \in \mathbb{R}^d$ <span style="font-size: 14px;">is the weight vector,</span> $b \in \mathbb{R}$ <span style="font-size: 14px;">is the bias, and</span> $\sigma$ <span style="font-size: 14px;">is the sigmoid function.</span>

---

## <span style="font-size: 16px;">The Sigmoid Function</span>

<span style="font-size: 14px;">The sigmoid (logistic) function maps any real number to the interval</span> $(0, 1)$<span style="font-size: 14px;">:</span>

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

<span style="font-size: 14px;">Key properties:</span>

- <span style="font-size: 14px;">Output is always between 0 and 1, interpretable as a probability</span>
- $\sigma(0) = 0.5$ <span style="font-size: 14px;">: the decision boundary when</span> $z = 0$
- <span style="font-size: 14px;">As</span> $z \to +\infty$<span style="font-size: 14px;">,</span> $\sigma(z) \to 1$<span style="font-size: 14px;">; as</span> $z \to -\infty$<span style="font-size: 14px;">,</span> $\sigma(z) \to 0$
- <span style="font-size: 14px;">Its derivative has the elegant form:</span> $\sigma'(z) = \sigma(z)(1 - \sigma(z))$

<span style="font-size: 14px;">The sigmoid squashes the linear output</span> $z = Xw + b$ <span style="font-size: 14px;">into a valid probability. If</span> $\hat{y}_i > 0.5$<span style="font-size: 14px;">, we classify the sample as positive (class 1); otherwise negative (class 0). The threshold 0.5 corresponds to the decision boundary where</span> $Xw + b = 0$<span style="font-size: 14px;">.</span>

---

## <span style="font-size: 16px;">Binary Cross-Entropy Loss</span>

<span style="font-size: 14px;">We cannot use MSE for classification because it creates a non-convex loss surface when combined with the sigmoid. Instead, we use the Binary Cross-Entropy (BCE) loss, also called log loss:</span>

$$
L(w, b) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
$$

---

## <span style="font-size: 16px;">Why This Loss?</span>

<span style="font-size: 14px;">Consider what happens for a single sample:</span>

- <span style="font-size: 14px;">If</span> $y_i = 1$<span style="font-size: 14px;">: loss is</span> $-\log(\hat{y}_i)$<span style="font-size: 14px;">. As</span> $\hat{y}_i \to 1$ <span style="font-size: 14px;">(correct), loss</span> $\to 0$<span style="font-size: 14px;">. As</span> $\hat{y}_i \to 0$ <span style="font-size: 14px;">(wrong), loss</span> $\to \infty$
- <span style="font-size: 14px;">If</span> $y_i = 0$<span style="font-size: 14px;">: loss is</span> $-\log(1 - \hat{y}_i)$<span style="font-size: 14px;">. As</span> $\hat{y}_i \to 0$ <span style="font-size: 14px;">(correct), loss</span> $\to 0$<span style="font-size: 14px;">. As</span> $\hat{y}_i \to 1$ <span style="font-size: 14px;">(wrong), loss</span> $\to \infty$

<span style="font-size: 14px;">The logarithmic penalty means confident wrong predictions are punished extremely harshly. This is both mathematically elegant (it is the negative log-likelihood under a Bernoulli model) and practically effective.</span>

---

## <span style="font-size: 16px;">Convexity</span>

<span style="font-size: 14px;">BCE loss is convex in</span> $w$ <span style="font-size: 14px;">and</span> $b$ <span style="font-size: 14px;">when used with the sigmoid function. This means gradient descent is guaranteed to find the global minimum, just like linear regression with MSE.</span>

---

## <span style="font-size: 16px;">Computing the Gradients</span>

<span style="font-size: 14px;">Let</span> $z = Xw + b$ <span style="font-size: 14px;">and</span> $\hat{y} = \sigma(z)$<span style="font-size: 14px;">. The gradients have a remarkably clean form.</span>

---

## <span style="font-size: 16px;">Gradient with respect to weights</span>

<span style="font-size: 14px;">Using the chain rule:</span>

$$
\frac{\partial L}{\partial w} = \frac{1}{n} X^T (\hat{y} - y)
$$

<span style="font-size: 14px;">This is identical in form to the linear regression gradient! The only difference is that</span> $\hat{y}$ <span style="font-size: 14px;">here passes through the sigmoid, while in linear regression it does not.</span>

---

## <span style="font-size: 16px;">Gradient with respect to bias</span>

$$
\frac{\partial L}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i) = \frac{1}{n} \mathbf{1}^T (\hat{y} - y)
$$

---

## <span style="font-size: 16px;">Deriving the Weight Gradient (Details)</span>

<span style="font-size: 14px;">The derivation uses three facts:</span>

- <span style="font-size: 14px;">Chain rule:</span> $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w}$
- $\frac{\partial L}{\partial \hat{y}_i} = -\frac{y_i}{\hat{y}_i} + \frac{1 - y_i}{1 - \hat{y}_i}$ <span style="font-size: 14px;">(derivative of BCE)</span>
- $\frac{\partial \hat{y}_i}{\partial z_i} = \hat{y}_i(1 - \hat{y}_i)$ <span style="font-size: 14px;">(sigmoid derivative)</span>

<span style="font-size: 14px;">Multiplying these together, the</span> $\hat{y}_i(1 - \hat{y}_i)$ <span style="font-size: 14px;">terms cancel perfectly:</span>

$$
\frac{\partial L}{\partial z_i} = \hat{y}_i - y_i
$$

<span style="font-size: 14px;">This clean cancellation is not a coincidence: it is a fundamental property of the exponential family of distributions, to which the Bernoulli distribution (and hence logistic regression) belongs.</span>

---

## <span style="font-size: 16px;">The Update Rules</span>

<span style="font-size: 14px;">At each iteration:</span>

$$
w \leftarrow w - \alpha \cdot \frac{1}{n} X^T (\hat{y} - y)
$$

$$
b \leftarrow b - \alpha \cdot \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)
$$

<span style="font-size: 14px;">where</span> $\alpha$ <span style="font-size: 14px;">is the learning rate.</span>

<span style="font-size: 14px;">Notice there is no factor of 2 (unlike MSE). The BCE loss was carefully designed so the gradients come out clean.</span>

---

## <span style="font-size: 16px;">Numerical Stability</span>

<span style="font-size: 14px;">In practice, we must handle numerical issues:</span>

- <span style="font-size: 14px;">Computing</span> $e^{-z}$ <span style="font-size: 14px;">can overflow for large negative</span> $z$<span style="font-size: 14px;">. NumPy's</span> `np.exp` <span style="font-size: 14px;">handles this gracefully by returning</span> `inf`<span style="font-size: 14px;">, giving</span> $\sigma(z) = 0$
- <span style="font-size: 14px;">Computing</span> $\log(\hat{y})$ <span style="font-size: 14px;">when</span> $\hat{y} = 0$ <span style="font-size: 14px;">gives</span> $-\infty$<span style="font-size: 14px;">. We clip predictions to</span> $[\epsilon, 1-\epsilon]$ <span style="font-size: 14px;">where</span> $\epsilon$ <span style="font-size: 14px;">is a small value like</span> $10^{-15}$

<span style="font-size: 14px;">These are standard practices used in production ML libraries like scikit-learn and PyTorch.</span>

---

## <span style="font-size: 16px;">Logistic Regression vs Linear Regression</span>

- <span style="font-size: 14px;">Linear regression predicts continuous values; logistic regression predicts probabilities</span>
- <span style="font-size: 14px;">Linear regression uses MSE loss; logistic regression uses BCE loss</span>
- <span style="font-size: 14px;">Both have convex loss surfaces and converge to global optima</span>
- <span style="font-size: 14px;">The gradient formulas are structurally identical:</span> $\frac{1}{n} X^T (\hat{y} - y)$
- <span style="font-size: 14px;">The difference is how</span> $\hat{y}$ <span style="font-size: 14px;">is computed: linear (</span>$Xw+b$<span style="font-size: 14px;">) vs sigmoid(</span>$Xw+b$<span style="font-size: 14px;">)</span>

---

## <span style="font-size: 16px;">Decision Boundary</span>

<span style="font-size: 14px;">The decision boundary is the set of points where</span> $\hat{y} = 0.5$<span style="font-size: 14px;">, which means</span> $Xw + b = 0$<span style="font-size: 14px;">. This is a hyperplane in feature space. Logistic regression can only create linear decision boundaries, which is why it is called a linear classifier despite using the nonlinear sigmoid.</span>

<span style="font-size: 14px;">For non-linearly separable data, you would need feature engineering (e.g., polynomial features) or a more powerful model (e.g., neural networks, SVMs with kernels).</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

- <span style="font-size: 14px;">**Q: Why cross-entropy and not MSE?**</span>
  <span style="font-size: 14px;">A: MSE creates a non-convex loss surface with the sigmoid; BCE is convex and corresponds to maximum likelihood under a Bernoulli model</span>

- <span style="font-size: 14px;">**Q: What happens with linearly separable data?**</span>
  <span style="font-size: 14px;">A: Weights diverge to $\pm\infty$ without regularization, because the sigmoid can always be pushed closer to 0 or 1</span>

- <span style="font-size: 14px;">**Q: How does the decision boundary relate to weights?**</span>
  <span style="font-size: 14px;">A: $w$ is the normal vector to the hyperplane $w^T x + b = 0$; its magnitude controls the sharpness of the probability transition</span>

- <span style="font-size: 14px;">**Q: Relation to neural networks?**</span>
  <span style="font-size: 14px;">A: A single neuron with sigmoid activation and cross-entropy loss IS logistic regression</span>

- <span style="font-size: 14px;">**Q: When does logistic regression fail?**</span>
  <span style="font-size: 14px;">A: When the decision boundary is non-linear; use kernel methods, trees, or neural networks instead</span>

- <span style="font-size: 14px;">**Q: How to interpret weights?**</span>
  <span style="font-size: 14px;">A: Each $w_j$ represents the change in log-odds per unit increase in feature $x_j$</span>

- <span style="font-size: 14px;">**Q: Multi-class extension?**</span>
  <span style="font-size: 14px;">A: Softmax regression replaces sigmoid with softmax and BCE with categorical cross-entropy</span>

---