---
aliases:
  - AgenticRAG
tags:
  - nlp
  - rag
  - llm
---
[Paper link](https://arxiv.org/abs/2408.14484)
![Main image](https://andlukyane.com/images/paper_reviews/agentic_rag/2024-09-04_11-05-32.jpg)  
  
The proposed approach introduces a novel framework for time series analysis using a multi-agent [[RAG]] system. This framework addresses challenges such as complex spatio-temporal dependencies and distribution shifts in time series data. It employs a hierarchical architecture where a master agent delegates tasks to specialized sub-agents, each fine-tuned for specific time series tasks. These sub-agents use smaller pre-trained language models and retrieve relevant prompts from a shared repository to enhance predictions. The modular, multi-agent RAG approach offers flexibility and achieves state-of-the-art performance across various time series tasks.  
  
### Problem formulation  
  
The dataset consists of N univariate time series, each collected over T timestamps. It is represented as a data matrix. There are four tasks:  
  
* Forecasting: A sliding window is used to construct subsequences from previous steps to predict future values for the next steps.   
* Missing Data Imputation: A binary mask matrix identifies missing data, and observed samples are used to estimate missing values by leveraging spatio-temporal dependencies within a sliding window.   
* Anomaly Detection: Anomalies are detected by comparing current behavior with the normal pattern from a training period. A sliding window approach is used to predict future values, and anomaly scores are computed to flag deviations from the moving averaged maximum anomaly value.   
* Classification: Unsupervised K-means clustering is applied to identify clusters in the data. A sliding window approach is used to predict future cluster labels based on past time steps.  
  
### The approach  
#### Dynamic Prompting Mechanism  
A dynamic prompting mechanism enhances time series modeling by addressing non-stationarity and distributional shifts. It improves traditional methods that use fixed window lengths, which may miss short-range or long-range dependencies. The approach retrieves relevant prompts from a shared pool of key-value pairs encoding historical patterns like seasonality, cyclicality, irregularities, etc. Input time series are projected into a vector space, and cosine similarity is used to match them with the most relevant prompts. These prompts are combined with the input data to improve predictions, allowing the model to adapt and leverage past knowledge for better performance across varying datasets.  
#### Fine-Tuning/Preference Optimization SLM  
Pretrained small language models, like Google’s Gemma and Meta’s Llama-3, are limited by an 8K token context window, which hinders their ability to process long input sequences. To address this, a two-tiered attention mechanism (grouped and neighbor attention) is introduced, allowing SLMs to capture long-range dependencies without fine-tuning, improving performance on extended text sequences. While fine-tuning SLMs for specific tasks can enhance performance, instruction-tuning with an extended 32K token context window, using parameter-efficient fine-tuning techniques, improves their ability to handle time series tasks. Additionally, Direct DPO is used to steer SLM predictions toward more reliable task-specific outcomes by randomly masking 50% of the data and performing binary classification to predict correct task-specific outcomes.  
  
### Experiments  
  
![Dataset summary](https://andlukyane.com/images/paper_reviews/agentic_rag/2024-09-04_10-58-33.jpg)  
  
![Results](https://andlukyane.com/images/paper_reviews/agentic_rag/2024-09-04_11-00-03.jpg)  
  
The Agentic-RAG framework variants were evaluated against baseline methods on seven benchmark datasets for forecasting, as well as on anomaly detection tasks. The results demonstrate that the proposed framework significantly outperforms baseline methods across these datasets.   
  
An ablation study was conducted to evaluate the contribution of individual components within the Agentic-RAG framework. The study analyzed the impact of removing key components: dynamic prompting mechanism, sub-agent specialization, instruction-tuning, and DPO. Results showed that the full framework consistently outperformed ablated versions in time series forecasting, anomaly detection, and classification tasks across multiple datasets.