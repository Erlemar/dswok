---
tags:
  - interview
  - concept
---
ML System Design interview is a stage of the job [[Interview preparation|interview]] process focused on assessing a candidate's ability to design and implement machine learning systems at scale. It involves open-ended questions about designing ML solutions for real-world problems. A typical question is "Design a [[Recommendation system|recommendation]]/fraud detection/image classification/etc system at a bank/telecom/etc". The expected answer includes:
* a brief overview of the system development process at a high level
* asking clarifying questions
* describing every stage of the system development in detail, suggesting various approaches and explaining their trade-offs

In other words, you are being tested for:
- Understanding the end-to-end process of ML model lifecycle
- Being able to clearly communicate the steps necessary to deliver the solution
- Being able to discuss business requirements and relevant technical solutions
- Having theoretical knowledge of relevant technologies and tools
## General outline of the answer

### Preliminaries
Start with a high-level overview of the development process (the paragraphs below).

Understand and, if necessary, clarify the business problem. Often, the question is formulated straightforward enough, but it is better to ask questions, than to assume something and be wrong. Some systems are real-time by default (for example, showing something to the user at the moment of opening an app - estimation of something, recommendation, etc), while others are non real-time by default - for example, producing some scores for the users on a daily/weekly basis. However, most systems can be both real-time and non-real-time, so it is essential to clarify what system the interviewer expects. So, ask how often the system should be updated, whether it will be applied to a certain segment of the data or all the data (all users or excluding cold start, all orders or reorders/nonreorders), what the expected size of the data (e.g., thousands/millions/billions), what is SLA/latency, etc.

Discuss business and ML metrics (you could discuss the ML metrics at the modeling stage instead). Often, multiple metrics are important - for example, revenue and conversion/engagement, costs and user satisfaction, etc.

Discuss project success criteria - reaching certain values of the metrics or exceeding baseline/existing solution.

Formulate an ML problem - classification/regression/etc.

Discuss possible optimizations, like pre-calculating or caching predictions, using CPU/GPU/TPU for inference.

### High-level design and diagrams
At this step, you are expected to describe a high-level design of the system and, if possible, provide a flowchart.

Mention a possible baseline solution, such as one based on simple rules and statistics. For [[Recommendation system|recommendations]], use the most popular items or a previous item the user interacted with. For texts/images, use a pre-trained model without tuning/instructions, etc.

Prepare a diagram, it should include data collection for training and during the inference, training flow (data collection, model training, deployment), the general logic of the pipeline in production.
### Data preparation and analysis
Discuss data collection. If the task is related to tabular data, the data is usually obtained from an existing database using SQL queries. There can be many sources of data at different levels - raw logs of events or aggregations, metadata, interaction data, etc.
For non-structured data it can be more difficult - it may be necessary to parse it from somewhere. But, usually, in the interview it can be assumed (after checking with the interviewer) that the data is available or can be obtained from the openly published datasets.

Discuss the target variable - how it is defined and collected. If the labels already exist and are trustworthy, this makes things easier. If the labels aren't available or are noisy, discuss how to improve or collect them.
Getting positive examples is often straightforward - they are clearly defined, and their number is manageable. But there could be too many negative samples or samples with unclear labels - you'll need to discuss your approach to sampling them.

* Manual labeling by yourself/rules/LLM/business experts/dedicated labeling team. Discuss the trade-offs:
	* You could do it yourself if you have enough deep domain knowledge. The cost will be your time - which could have been spent on other tasks. But, often, it is much more time-efficient to spend a couple of days labeling the data, than waiting for other people to do it
	* Sometimes, it is possible to apply rules or use tools like [Snorkel](https://www.snorkel.org/use-cases/01-spam-tutorial) to generate labels
	* Using LLMs like ChatGPT with good prompts is a viable modern way to generate labels, but if the domain is specific, the results will be noisy
	* Business experts often have a better understanding of the data, but in practice, their labels will still require validation
	* The ideal situation is when a dedicated labeling team (internal or external) performs this task, but such a situation requires mature processes and tools
* Label verification. Collecting labels is the first step, but then it is necessary to verify them, as they can be noisy. A classical approach is to do cross-labellng - have several people label the same data, compare the results and update the labels based on the discovered errors.

Discuss data analysis - what would you check in the data (distributions, missing values, feature interactions), and what possible insights could be obtained. For example, you could decide to apply your solution only to a subset of the data based on specific criteria (new/old users, users with specific criteria, images with good enough quality, texts with a certain minimum length, etc). You could encounter data issues and need to decide how to deal with them.

Data could also be logged during the user session: every action taken, every item interacted with. Additionally, for each action we collect the metadata (timestamp, place in the feed, user and item features) and calculate the features that are necessary for making predictions.

Describe how would a sample of training data look like. For example, for the recommendation systems, the data may look like:
```
user1, item1, action, user1_features, item1_features
user1, item2, action, user1_features, item2_features
user2, item1, action, user2_features, item1_features
```

If, relevant, describe [[negative sampling]] approaches (for recommendation systems, for contrastive learning).

#### Feature engineering
Describe as many different ways to create the features, as possible. This may include: embedding extraction, user/item/creator features (numerical and categorical), metadata, context, graph-based features, interaction features, etc.
The interviewers expect imagination and being able to come up with many different types of the features.

### Models
In practice, the previous and current sections are interconnected - the data available defines the model, which in turn requires specific data prepared in a specific way.
Start by discussing the modeling approach. If you didn't mention starting with a baseline before, now is a good time to mention it. Suggest multiple approaches and discuss trade-offs. For example:
* Tabular data: linear models (like [[Logistic regression]]) vs. tree-based models ([[Random Forest]] or [[Gradient boosting]]). 
* Images: classical approaches vs using pre-trained models vs. training models from scratch
* Texts: using top models as a baseline vs. prompt engineering vs. instruction-tuning vs. fine-tuning
* Recommendation systems: two-stage (retrieval + ranking)
It is a good idea to start with a simple approach and describe modern SOTA alternatives.

For the selected model, explain how it will use the data - feature engineering and feature selection for tabular data, vectorization for texts, etc. If relevant, explain the losses that will be used for training. If relevant, discuss data augmentation. Explain the model itself - how it will be trained and how it will make predictions. 

Discuss model evaluation approaches, such as [[Validation|validation]] split (random, stratified, group-based, time-based, etc.), metrics used, retro evaluation, etc.

### Deployment
This will depend on the infrastructure of the specific company, but there are many common things - first, the technical side of the deployment itself.
* Data preparation - how will you fetch the data when making predictions? This could involve using a feature store (describe how the feature values will be updated and fetched), keeping data in-memory (when the size is small), using specialized databases, etc. Real-time systems would require more complex solutions (don't forget to discuss latency requirements), offline predictions are usually easier
* Model deployment. This could be running a model on a server and serving predictions using API, this could be about converting the model to ONNX or other formats. If relevant, discuss model optimization - quantization, pruning, distillation. If the model will run on the devices, discuss it.
* Model scaling - will the model work fine on 2x of expected traffic, 10x, 100x? What would be necessary to ensure it?
* Handling of failures - what should we do if the input is invalid, if we can't get the data, or if the prediction can't be produced?
Next is rolling the model into production.
- A/B tests to check the model performance. Discuss how you would organize it
- Smoke tests - making predictions but not acting on them, then calculating metrics
- Canary deployment - gradual rollout
- Bandits

Discuss how the model will be used:
- Will it trigger after a certain event and show a certain output? How will this output be represented?
- Will the predictions be used in other systems (predicting churned customers -> contacting them, predicting suspicious behavior -> validating/blocking)?
- Discuss how will you handle different loads - what would you do when the capacity of the solution isn't enough (scaling it up, serving cached predictions)?
- Discuss privacy - how would you ensure that the user data is kept private.
### Post-deployment
The deployment isn't the last step of the process:
- Monitoring: many things could be monitored - technical performance (number of errors, number of model calls, resources used), model performance (the distribution of the predictions, metrics), data checks (distribution of the data, including data drift, missing values, etc), business metric monitoring, checking other models (there could be an effect of model cannibalism - when the whole performance of the app increases, but the new model "steals" hits from other models)
- Re-training - when to do re-training, how the process would be organized, what to do with the [[feedback loop]] (if it is relevant)
- Possible model improvements

## Advice
- In each section, start with a high-level description and ask the interviewer if they want you to go deeper - sometimes they want it, sometimes they don't
- Describe the process step-by-step; don't jump back and forth between the steps
- Practice several talks about different systems beforehand

## Links
- [A huge collection of case studies](https://www.evidentlyai.com/ml-system-design)
- [A great book on system design](https://www.manning.com/books/machine-learning-system-design)
- [A great source of info on System Design (not ML)](https://github.com/donnemartin/system-design-primer)
- [Rules of Machine Learning by Google](https://martin.zinkevich.org/rules_of_ml/rules_of_ml.pdf)
