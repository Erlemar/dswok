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
