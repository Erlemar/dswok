---
tags:
  - interview
aliases:
  - interview
  - Interview
---
This note is for interview questions that can't be easily moved to other notes.
[[Deep Learning questions]]
[[General ML questions]]
[[Statistics questions]]
[[Questions to ask the interviewers]]

Different types of interviews:
- [[Behavioral interviews]]
- [[ML System design]]
- [[Leetcode code templates]]




1. Pearson correlation 
    
2. Covariance vs. correlation 
    
3. Covariance and PCA – how are they related? 
    
4. The complexity of the SVD decomposition? 
    
5. You are given two medicines: paracetamol and aspirin. You need to conclude which one is better at fighting a fever. How would you approach this problem (Randomized Controlled Trials, stat tests)
    
6. What’s the difference between t-test and z-test? 
    
7. What’s p-value? 
    
8. You are given two time series: BTC prices and news sentiments. How would you measure if there’s a price prediction signal in sentiments?
    
9. Suppose BTC prices and news sentiments are correlated. How to conclude which one has a causal effect on the other? (didn’t properly answer)
    
10. Suppose you build lags for both time series and measure partial correlations. How can this help in identifying the causal relation between BTC prices and news sentiments? (Came to an answer with a hint: you can build lagged series comparing yesterday’s sentiments with today’s prices and vice versa. If there’s a correlation between yesterday’s sentiment and today’s prices but no correlation between yesterday’s price and today’s sentiment, then sentiments drive prices, not vice versa)
    

11. What NER metrics do you know?
    
12. Why accuracy can be bad for NER? (class imbalance)
    
13. The difference between micro-and macro-averaged F1 scores
    
14. How to handle class imbalance with specific losses (Focal loss, answered but didn’t properly describe)
    
15.  What statistical tests for measuring inter-annotator agreement do you know? 
    
16. LSTMs vs. transformers – describe trade-offs
    
17. What’s the complexity of the attention mechanism? 
    
18. How do you apply transformers for a task with long documents? (I mentioned longformer, chunking, and block-wise attention)
    
19. Information retrieval: what’s pairwise loss? (described at a high level, mentioned Lambda MART) 
    
20. What’s the difference between DCG and nDCG? (described a bit clumsily)
    
21. You’re given a large annotated collection of question-document pairs labeled with relevance: excellent, medium, and poor. You are also given a large click dataset. What’s the relation between click data and human-annotated data? Is having click data only sufficient? How would you train an IR model with the data at hand?


22. How does boosting work?
    
23. Tell me about bias-variance error decomposition. How does it apply to boosting? Bagging? (I only struggled with bias reduction in case of boosting, pretty dumb: an ensemble of week learners decreases the bias of week learners)
    
24. Rf and gradient boosting: practical trade-offs?
    
25. What’s the advantage of BERT over RNNs?
    
26. Why do vanilla RNNs experience vanishing gradients? (backprop in time)
    
27. Can we use ReLU to fix vanishing gradients in a vanilla RNN? (struggled, never heard of ReLU being used in RNNs)
    
28. How does LSTM fix the same problem?
    
29. How does the attention mechanism work? (queries, keys, values, etc.)
    

What’s the BERT’s biggest innovation? (mentioned the MLM task)





1. You are given a large collection of user click data for a large set of search queries and documents (product descriptions). How would you come up with a formula to rank documents given a new search query? (described pairwise loss, SVM, and boosting, mentioned Lambda MART)
    
2. We might have some frequent queries like “iphone X” and a long tail of infrequent queries. How would that influence your model training process? (group instances – (query, doc) tuples by queries, and only compare 2 docs for the same query)
    
3. How do you adapt your search system to a new language? (multilingual models or translating descriptions to a new language, assess quality with a small set created by human translators)
    
4. How would you deal with Chinese queries? (no idea, mentioned that I’ll reuse existing Chinese tokenizers)
    
5. Suppose you a training an NMT model with data from 15 different sources. Now you need to extend the solution to a new source - news headlines. How would you approach this? (domain adaptation. I mentioned GPL – generative pseudo-labels for domain adaptation. However, wasn’t able to properly describe the model. So instead came up with adversarial validation to select those records in 15 sources that most resemble the new source. Turns out, the interviewer did the same in his real project)
    




Behavioral (6x)

  

1. You a finishing a paper but some experiments are not yet done. A deadline for an A-grade conference is approaching. You feel safer submitting to a B-grade conference with a deadline in 2 months. What would you do and why? 
    
2. Describe your largest ever deliverable. 
    
3. What’s your most innovative idea?
    
4. Tell me about a time when you made some user-facing simplification. What drove the need for such a simplification? 
    
5. Tell me about a time when you had to dig deep into the root-cause analysis. How did you know you needed to dig deeper?
    
6. What would you’ve done differently in any of your projects?  
    
7. Tell me about a time when you did something significant that was beyond your job responsibility.
    

  

8. What could you have done better in the previous example? 
    
9. Tell me about a time when you were not satisfied with some process or status quo in your company and decided to change this. 
    
10. Tell me about a time when you had to go along with a group decision that you disagreed with. What did you do?
    
11. Tell me about a time when you took a big risk. How did you decide that you are taking this risk anyway?
    
12. Tell me about a time when you took a calculated risk
    
13. Tell me about a time when you had several options and had to make a decision. How did you pick one of the options?
    
14. Tell me a time when you piece when you received a piece of critical feedback from your colleague. How did you handle it? What did you take out from this experience?
    

You work at an eCommerce startup. For most of the company's existence, customers have left comment-only reviews. Six months ago, you introduced the option to leave a star rating (1-5 stars) in addition to comments. Your boss wants to be able to sort products by average rating, while also taking advantage of the years of reviews that lack ratings (approx. 75% of all reviews). She has asked you to produce estimated star ratings for all reviews that lack an official rating using an ML system. 

That was a bit of an atypical challenge to propose an architecture of an LLM evaluation system.
Very open-ended, need to clarify and make assumptions a lot. 
Possible things to discuss: 
Are foundation models evaluated? Or only downstream task performance?
Which tasks to focus on?
How to run evaluation? How do SMEs annotate data? 
How would the annotator interface look like?  


1st technical interview
Intro 

1. Do you have experience dealing with super-large language models? Do you like do model parallelism at all? 
2. Did you work with 70b models or only with 7b and 13b? 
3. Do you have production experience with model alignment? 
4. Okay, so you're saying you haven't started with DPO and RLHF stuff yet, right?

NLP

5. Can you explain to me how self-attention works?
6. Now the same in mathematical terms 
7. Can transformer inference be parallelized? 
8. What’s the complexity of the self-attention operation? 
9. After the self-attention, what happens in the transformer?
10. How many feed-forward layers are there in the transformer block?


1st technical interview (cont.)
NLP 

11. What’s the dimension of the feed-forward layer?
12. So internally, it’s super wide. Do you know any reason why people design like that? 
13. Do you know this paper where people can edit the transformer memory? Have you heard this? 
14. Basically the knowledge is stored in the weights of the transformers, right? So like, for example, the Eiffel Tower is in Paris, right? So this knowledge can be edited. So they find out where the memory located. You know this paper?
15. Have you read about like Hopfield network? (No) Yeah, this is called associative memory. So it's a Hopfield network. It's kind of like an ML, feed-forward network, MLP. Basically, that's where the memory happens. You can store this key value.
16. Have you read the RETRO paper?
17. So have you done anything with RETRO before?
18. Do you know how this RETRO external information is feeding into the language model?
19. Can you explain to me what's the difference between T5 and GPT? 
20. How does the encoded information fit into the decoder in T5? 
21. So, can you revisit the question about RETRO feeding the retrieved documents into the decoder?

1st technical interview (cont.)
Coding

22. Let me first start with some easy questions. Can you explain to me what's the difference between variables on stack versus variables on heap? 
23. It’s about memory allocation. So what's the main difference, how it's stored in memory? 
24. So, have you done like programming? Anything apart from Python? 
25. In Java memory management, do you know the few generations of the variables in the memory?
26. How does garbage collection work in Java?
27. How does a variable on a stack work?
28. How is it related to the scope of variables, e.g. global and local ones? Where are those allocated in memory? 
29. Why does recursion use a stack? 

Algorithms

30. Describe a solution to the “8 queens” problem. Describe the pseudocode (no need to write code)
31. What’s the complexity of the algorithm? 
32. What’s the classic CS 101 algorithm for this problem? 

System design
Google wants to collect Street View images from taxis. You are responsible for designing the system that uploads images from the taxis and stores those images in the cloud for future processing.
You can specify the hardware to be deployed in each taxi, and can control when the camera should capture images. The default camera takes 360 degree images.
The images on the server will be processed by another layer to manage privacy, quality and consistency. That layer is responsible for storing the final images for Street View users.


You are tasked to create an application that summarizes content from a large set of user manuals for your haywack installation teams, heating, ventilation, air conditioning installation teams in the field. They will only have a cell phone as their device to access this content. Teams in the field need quick and accurate answers. Please describe how you would approach designing and architecting a solution for this. It's a long question. I can repeat by the way if you want.
How would you describe the complexities of training and serving large machine learning models, the challenges of doing this in a distributed environment,and techniques to mitigate these challenges to a group of data scientists and machine learning engineers?
So, if the team attempts to fine-tune a large language model but gets out of memory errors, what are some things that can be done to address this?
Do you have any examples for use cases where a large language model might not be the best choice and why?


Role-related knowledge (cont).
How would you explain large language models, the advantage and disadvantages, and when to recommend them to a non-technical CEO?
In terms of costs, what would be the key components? What would be the reason of high costs?
And what do you think, what would you expect would be important about AI ML to the non-technical business leader? What would be their most important thing in their mind when you talk about AI ML?
And what do you think the CEO needs to know to make the decision? You are in a like a sales situation and you're explaining what they can do with large language models or large models in general. So what does he need to know to decide?
I wonder your thoughts on using something out of the box versus building something from scratch. How would you approach the problems initially in different strategies or large-scale models, using large-scale models? 


https://interviewing.io/guides/system-design-interview
https://github.com/donnemartin/system-design-primer/tree/master
https://nn.labml.ai/

https://www.pramp.com/#/
https://interviewing.io/
https://github.com/alirezadir/Machine-Learning-Interviews/blob/main/src/MLSD/ml-system-design.md
https://huyenchip.com/ml-interviews-book/
https://igotanoffer.com/blogs/tech/amazon-machine-learning-engineer-interview
https://igotanoffer.com/blogs/tech/amazon-bar-raiser-interview
https://www.youtube.com/watch?v=PJKYqLP6MRE&ab_channel=JacksonGabbard
https://habr.com/ru/companies/ods/articles/673376/


