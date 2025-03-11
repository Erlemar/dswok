---
aliases: 
tags:
  - interview
---
Q: You have three independent binary classifiers with 80% accuracy. What is the accuracy of the majority-vote classifier?
A: The majority vote will be correct if at least two out of the three classifiers are correct. This can be calculated as the sum of probabilities:
* All three classifiers are correct: $0.8×0.8×0.8=0.512$
* Two classifiers are correct, and one is incorrect: $3×0.8×0.8×0.2=0.384$
Total: $0.512+0.384=0.896$

Q: You have a discrete random variable X
$P(X = 0) = 0.5$
$P(X = 1) = 0.4$
$P(X = 6) = 0.1$
What is the mean and variance of X?

A:
Mean: $(0×0.5)+(1×0.4)+(6×0.1) = 1$
Variance: $(0^2×0.5)+(1^2×0.4)+(6^2×0.1) - 1 ^ 2 = 3$

Q: Let A and B be events on the same sample space, with P (A) = 0.6 and P (B) = 0.7. Can these two events be disjoint?
A: No. If they were disjoint, $P(A ∪ B) = P(A) + P(B) = 0.6 + 0.7 = 1.3$, which is impossible as probabilities cannot exceed 1.

Q: Given that Alice has 2 kids, at least one of which is a girl, what is the probability that both kids are girls?
A: 1/3. The possible outcomes are (G,G), (G,B), and (B,G). Given that at least one is a girl, we eliminate (B,B). Of the remaining three equally likely outcomes, only one is (G,G).

Q: A group of 60 students is randomly split into 3 classes of equal size. All partitions are equally likely. Jack and Jill are two students belonging to that group. What is the probability that Jack and Jill will end up in the same class?
A: 1/3. Jack can be in any class. Once Jack's class is determined, Jill has a 1/3 chance of being in the same class as Jack.

Q: What is the difference between arithmetic mean and harmonic mean? 
A: The harmonic mean is always lower than or equal to the arithmetic mean. It's useful for averaging rates or speeds.
- Arithmetic mean: The sum of values divided by the count of values: $\frac {(x₁ + x₂ + ... + xₙ)} {n}$
- Harmonic mean: The reciprocal of the arithmetic mean of reciprocals: $\frac {n} {(\frac {1}{x₁} + \frac {1}{x₂} + ... + \frac {1}{xₙ})}$
Q: What is the difference between mean, median and mode?
A: The mean is the average of all values. It is sensitive to outliers and extreme values. The median is the middle value of an ordered list of values. It is less sensitive to outliers than the mean. The mode is the value that appears most frequently in a dataset. There can be 0, 1 or 2 modes in the data. It can be used for categorical data.

Q: There are two boxes with balls. The first box has 3 green balls and 7 red balls. The second box has 6 green balls and 4 red balls. The probability of choosing the first box is 40%. We pulled one green ball at random. What is the probability that it was from the first box?
A: $P(B_1) = 0.40$, $P(B_2) = 0.60$, $P(G \mid B_1) = \frac{3}{3+7} = 0.30$, $P(G \mid B_2) = \frac{6}{6+4} = 0.60$. $P(B_1 \mid G)=\frac{ P(B_1) \, P(G \mid B_1) }{ P(B_1) \, P(G \mid B_1) + P(B_2) \, P(G \mid B_2) }=\frac{0.40 \times 0.30}{ 0.40 \times 0.30 + 0.60 \times 0.60 }=0.25$

Q: Disease Testing. A rare disease that affects 2% of the population. The test for this disease is 90% accurate:
- If a person has the disease, the test will correctly return positive 90% of the time.
- If a person does not have the disease, the test will correctly return negative 90% of the time.
If a person tests positive, what is the probability that they actually have the disease?
A: $P(\text{Disease}) = 0.02$. $P(\text{No Disease}) = 1 - 0.02 = 0.98$. $P(\text{Test Positive} \mid \text{Disease}) = 0.90$ (true positive rate). $P(\text{Test Positive} \mid \text{No Disease}) = 1 - 0.90 = 0.10$ (false positive rate).

$P(\text{Disease} \mid \text{Test Positive})=\frac{P(\text{Disease}) \cdot P(\text{Test Positive} \mid \text{Disease})}{P(\text{Disease}) \cdot P(\text{Test Positive} \mid \text{Disease}) + P(\text{No Disease}) \cdot P(\text{Test Positive} \mid \text{No Disease})}$
$\frac{0.02 \times 0.90}{0.02 \times 0.90 \;+\; 0.98 \times 0.10}=0.155$

Q: Covariance vs. correlation.
A: Covariance measures the linear relationship between two variables. $\text{Cov}(X, Y) = \frac{1}{n} \sum_{i=1}^{n} (X_i - \bar{X})(Y_i - \bar{Y})$. If positive, they increase together; if 0, there is no linear relationship; if negative, when one variable increases, the other decreases. Covariance is scale-dependent
Correlation standardizes covariance by dividing by the product of standard deviations, it is unitless. $\text{Corr}(X, Y) = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$ Correlation is between -1 and 1.

Q: What’s the difference between t-test and z-test?
A: Use a z-test when n is large or population variance is known, t-test, when n is small or population variance is unknown. At large n, they become nearly equivalent.