# <span style="font-size: 20px;">Linear Regression from Scratch</span>

<span style="font-size: 14px;">Linear regression is the starting point of supervised ML. It models the target as a linear function of features plus noise, and finding the optimal parameters reduces to a well-studied optimization problem. Implementing both the closed-form and iterative solutions tests whether you understand the math behind the model, not just how to call a library.</span>

---

## <span style="font-size: 16px;">The Linear Model</span>

<span style="font-size: 14px;">For</span> $n$ <span style="font-size: 14px;">samples with</span> $d$ <span style="font-size: 14px;">features:</span>

$$
\hat{y} = Xw + b \cdot \mathbf{1}
$$

* $X \in \mathbb{R}^{n \times d}$ <span style="font-size: 14px;">is the feature matrix</span>
* $w \in \mathbb{R}^d$ <span style="font-size: 14px;">is the weight vector</span>
* $b \in \mathbb{R}$ <span style="font-size: 14px;">is the bias (intercept)</span>

<span style="font-size: 14px;">The objective is to minimize the mean squared error:</span>

$$
L(w, b) = \frac{1}{n} \| Xw + b - y \|^2
$$

---

## <span style="font-size: 16px;">Normal Equation</span>

<span style="font-size: 14px;">Augment</span> $X$ <span style="font-size: 14px;">with a column of ones to absorb the bias:</span>

$$
\tilde{X} = [\mathbf{1} \,|\, X] \in \mathbb{R}^{n \times (d+1)}, \quad \theta = [b, w_1, \ldots, w_d]^T
$$

<span style="font-size: 14px;">Setting the gradient to zero gives:</span>

$$
\tilde{X}^T \tilde{X} \, \theta = \tilde{X}^T y \implies \theta^* = (\tilde{X}^T \tilde{X})^{-1} \tilde{X}^T y
$$

* <span style="font-size: 14px;">Gives the exact solution in one step, no hyperparameters</span>
* <span style="font-size: 14px;">Complexity:</span> $O(nd^2 + d^3)$ <span style="font-size: 14px;">- impractical when</span> $d$ <span style="font-size: 14px;">is very large</span>
* <span style="font-size: 14px;">Requires</span> $\tilde{X}^T \tilde{X}$ <span style="font-size: 14px;">to be invertible (fails with collinear features or</span> $d > n$<span style="font-size: 14px;">)</span>

---

## <span style="font-size: 16px;">Gradient Descent</span>

<span style="font-size: 14px;">The gradients of MSE with respect to the parameters are:</span>

$$
\begin{aligned}
\frac{\partial L}{\partial w} &= \frac{2}{n} X^T (\hat{y} - y) \\[4pt]
\frac{\partial L}{\partial b} &= \frac{2}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)
\end{aligned}
$$

<span style="font-size: 14px;">Update rules at each iteration:</span>

$$
\begin{aligned}
w &\leftarrow w - \alpha \cdot \frac{\partial L}{\partial w} \\[4pt]
b &\leftarrow b - \alpha \cdot \frac{\partial L}{\partial b}
\end{aligned}
$$

* <span style="font-size: 14px;">Each iteration costs</span> $O(nd)$<span style="font-size: 14px;">, total</span> $O(Tnd)$ <span style="font-size: 14px;">for</span> $T$ <span style="font-size: 14px;">iterations</span>
* <span style="font-size: 14px;">Scales to any</span> $d$<span style="font-size: 14px;">, generalizes to non-linear models</span>
* <span style="font-size: 14px;">Requires choosing</span> $\alpha$ <span style="font-size: 14px;">and</span> $T$<span style="font-size: 14px;">; too large</span> $\alpha$ <span style="font-size: 14px;">diverges, too small converges slowly</span>
* <span style="font-size: 14px;">Convergence speed depends on the condition number of</span> $X^T X$ <span style="font-size: 14px;">; feature scaling helps</span>

---

## <span style="font-size: 16px;">Normal Equation vs Gradient Descent</span>

* <span style="font-size: 14px;">**Normal equation:** exact, no hyperparameters, but</span> $O(d^3)$ <span style="font-size: 14px;">and requires invertibility</span>
* <span style="font-size: 14px;">**Gradient descent:** approximate, needs tuning, but</span> $O(nd)$ <span style="font-size: 14px;">per step and works for any differentiable model</span>
* <span style="font-size: 14px;">Use normal equation when</span> $d < 10{,}000$<span style="font-size: 14px;">; gradient descent when</span> $d$ <span style="font-size: 14px;">is large or the model is non-linear</span>

---

## <span style="font-size: 16px;">Geometric Interpretation</span>

<span style="font-size: 14px;">The prediction</span> $\hat{y} = \tilde{X}\theta$ <span style="font-size: 14px;">is the projection of</span> $y$ <span style="font-size: 14px;">onto the column space of</span> $\tilde{X}$<span style="font-size: 14px;">. The residual</span> $y - \hat{y}$ <span style="font-size: 14px;">is orthogonal to every column of</span> $\tilde{X}$<span style="font-size: 14px;">, which is exactly what the normal equation states:</span> $\tilde{X}^T(y - \tilde{X}\theta) = 0$<span style="font-size: 14px;">.</span>

---

## <span style="font-size: 16px;">Statistical Foundation</span>

* <span style="font-size: 14px;">Under Gaussian noise</span> $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$<span style="font-size: 14px;">, minimizing MSE is equivalent to maximum likelihood estimation</span>
* <span style="font-size: 14px;">The Gauss-Markov theorem guarantees OLS is the Best Linear Unbiased Estimator (BLUE)</span>
* <span style="font-size: 14px;">Coefficient of determination:</span> $R^2 = 1 - SS_{res}/SS_{tot}$ <span style="font-size: 14px;">measures proportion of variance explained</span>

---

## <span style="font-size: 16px;">Common Interview Follow-ups</span>

* <span style="font-size: 14px;">**Q: When to use each method?**</span>
  <span style="font-size: 14px;">A: Normal equation for small $d$, GD for large $d$ or non-linear extensions</span>

* <span style="font-size: 14px;">**Q: Collinear features?**</span>
  <span style="font-size: 14px;">A: Normal equation fails; GD still works but solution is not unique</span>

* <span style="font-size: 14px;">**Q: Adding regularization?**</span>
  <span style="font-size: 14px;">A: Ridge adds $\lambda I$ making $(\tilde{X}^T\tilde{X} + \lambda I)^{-1}\tilde{X}^T y$</span>

* <span style="font-size: 14px;">**Q: Large $n$?**</span>
  <span style="font-size: 14px;">A: Use stochastic or mini-batch gradient descent</span>

* <span style="font-size: 14px;">**Q: Why MSE over MAE?**</span>
  <span style="font-size: 14px;">A: MSE gives closed-form gradient and corresponds to Gaussian MLE; MAE is more robust but non-differentiable at zero</span>

---