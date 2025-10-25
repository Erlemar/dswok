---
tags:
- nlp
---
[Paper link](https://arxiv.org/abs/2304.03442)

[Code link](https://github.com/joonspk-research/generative_agents)

[Demo link](https://reverie.herokuapp.com/arXiv_Demo/#)

![Main image](https://andlukyane.com/images/paper_reviews/ishb/2023-04-22_15-23-12.jpg)

This paper introduces generative agents, computational software agents that simulate believable human behavior. The architecture extends a large language model to store and synthesize agents' experiences, memories, and plans using natural language. These agents are instantiated in an interactive sandbox environment inspired by The Sims, where users can interact with them using natural language. In evaluations, generative agents demonstrate believable individual and emergent social behaviors. The paper shows that the components of the agent architecture—observation, planning, and reflection—are essential for believability. This work lays the foundation for simulations of human behavior by merging large language models with interactive agents.

### Generative agent behavior and interaction

![The Smallville sandbox world](https://andlukyane.com/images/paper_reviews/ishb/2023-04-22_10-53-42.jpg)

#### Agent Avatar and Communication

A community of 25 unique agents, represented by sprite avatars, inhabit Smallville. Each agent's identity, including occupation and relationships, is described in a one-paragraph natural language seed memory.

![John Lin](https://andlukyane.com/images/paper_reviews/ishb/2023-04-22_10-55-49.jpg)

In the interactive sandbox environment, agents engage with the world and with each other through a combination of actions and natural language communication. At each time step, agents output a natural language statement that describes their current action, such as "Isabella Rodriguez is writing in her journal." These statements are then translated into concrete movements that affect the sandbox world and are displayed on the interface as a set of emojis, providing an abstract representation of the action in the overhead view. A language model is employed to translate the action into emojis, which appear in a speech bubble above each agent's avatar.

Agents communicate with each other using full natural language, and their awareness of other agents is determined by their proximity within the sandbox environment. The generative agent architecture influences whether they walk past each other or engage in conversation based on their location.

Users running the simulation have control over its direction and can intervene in two ways: by engaging in conversation with an agent or by issuing a directive as the agent's "inner voice." Users communicate with agents using natural language and can specify a persona for the agent to perceive them as. When users want to directly command an agent, they assume the role of the agent's "inner voice," which increases the likelihood of the agent treating the user's statement as a directive. For example, if the user tells John, as his inner voice, "You are going to run against Sam in the upcoming election," John will decide to run in the election and share his candidacy with his family.

#### Environmental Interaction

Smallville is a simulated village featuring common spaces and objects, such as a cafe, bar, park, school, dorm, houses, and stores, as well as functional subareas like kitchens with stoves. Agents move around Smallville, interacting with the environment and each other. Users can also join Smallville as an agent, either embodying an existing character or as a new visitor.

Both users and agents can influence the state of objects in the environment, similar to sandbox games like The Sims. Users can reshape an agent's surroundings by changing the status of objects using natural language commands. For example, users can change a stove's status from "turned on" to "burning," prompting an agent to react accordingly. Agents may also respond to other changes in their environment, such as fixing a leaking shower.

#### Example "Day in the Life"

<div class="gallery" data-columns="2">
<img src="https://andlukyane.com/paper_reviews/ishb/2023-04-22_11-00-49.jpg">
<img src="https://andlukyane.com/paper_reviews/ishb/2023-04-22_11-02-19.jpg">
</div>

#### Emergent Social Behaviors

By interacting with each other, generative agents exchange information, form new relationships, and coordinate joint activities. These social behaviors are emergent rather than pre-programmed.

**Information Diffusion**. As agents notice each other, they may engage in dialogue—as they do so, information can spread from agent to agent.

**Relationship memory**. Agents in Smallville form new relationships and remember their interactions with other agents over time. For instance, Sam initially doesn't know Latoya, but after meeting her in Johnson Park and learning about her photography project, he later remembers their interaction and asks her about the project's progress during a subsequent conversation.

**Coordination**.

![Coordination](https://andlukyane.com/images/paper_reviews/ishb/2023-04-22_11-05-54.jpg)

In Smallville, generative agents coordinate with each other. When Isabella is initialized with the intent to plan a Valentine's Day party, she invites friends and customers as she encounters them. Isabella and her friend Maria, who has a crush on Klaus, decorate the cafe for the event. Maria later invites Klaus to join her at the party. On Valentine's Day, five agents, including Klaus and Maria, attend the party and enjoy the festivities. In this scenario, the user only sets Isabella's initial intent and Maria's crush on Klaus, while the agent architecture autonomously generates the social behaviors, such as spreading the word, decorating, asking each other out, and interacting at the party.

### Generative agent architecture

![Architecture](https://andlukyane.com/images/paper_reviews/ishb/2023-04-24_08-38-42.jpg)

Generative agents aim to create a framework for behavior in an open world, allowing them to interact with other agents and respond to environmental changes. The novel agent architecture combines a large language model with mechanisms for synthesizing and retrieving relevant information to condition the language model's output. The memory stream, a database maintaining a comprehensive record of an agent's experience, is central to this architecture.

The current implementation utilizes the gpt3.5-turbo version of ChatGPT due to GPT-4's API being invitation-only. The core elements of generative agents—memory, planning, and reflection—will likely remain consistent as language models improve, with newer models like GPT-4 further expanding the expressivity and performance of the prompts that drive generative agents.

#### Memory and Retrieval

![Memory stream](https://andlukyane.com/images/paper_reviews/ishb/2023-04-22_11-11-24.jpg)

The architecture uses a retrieval function to select a subset of the memory stream, considering an agent's current situation. It focuses on three main components: recency, importance, and relevance.

* Recency assigns a higher score to recently accessed memories, using an exponential decay function.
* Importance differentiates between mundane and core memories by assigning higher scores to important memories. The language model is used to output an integer score for importance.
* Relevance assigns a higher score to memories related to the current situation, conditioned on a query memory. The language model generates embedding vectors for the text description of each memory, and relevance is calculated as the cosine similarity between these vectors.

The final retrieval score is calculated by normalizing and combining the recency, relevance, and importance scores, with equal weighting. The top-ranked memories that fit in the language model's context window are included in the prompt.

#### Reflection

![Reflection](https://andlukyane.com/images/paper_reviews/ishb/2023-04-22_11-15-53.jpg)

Generative agents struggle to generalize or make inferences when only using raw observational memory. To address this challenge, a second type of memory called "reflection" is introduced. Reflections are higher-level, more abstract thoughts generated by the agent and are included in the retrieval process along with observations. Reflections are generated periodically, triggered when the sum of importance scores for recent events exceeds a certain threshold. In practice, agents reflect about two or three times a day, enabling them to make better decisions and generalizations.

The reflection process involves agents identifying salient questions based on their recent experiences. They query the large language model with the 100 most recent records from their memory stream and generate candidate questions. These questions are then used for retrieval, and relevant memories are gathered. The language model extracts insights and cites records as evidence for these insights. These insights are then stored as reflections in the memory stream, along with pointers to the cited memory objects. Reflections can be based on both observations and other reflections, creating a tree-like structure with more abstract and higher-level thoughts further up the tree.

Example:
* Take 100 most recent records (like “Klaus Mueller is reading a book on gentrification”, “Klaus Mueller is conversing with a librarian about his research project”);
* Prompt the language model, “Given only the information above, what are 3 most salient high-level questions we can answer about the subjects in the statements?” 
* An example of a question generated by the model: "What topic is Klaus Mueller passionate about?"
* Use these questions as queries for retrieval, and retrieve relevant memories;
* Then extract insights like this:

> Statements about Klaus Mueller
> 1. Klaus Mueller is writing a research paper
> 2. Klaus Mueller enjoys reading a book on gentrification
> 3. Klaus Mueller is conversing with Ayesha Khan about exercising [...]
> What 5 high-level insights can you infer from the above statements? (example format: insight (because of 1, 5, 3))

#### Planning and Reacting

The challenge is that large language models can generate plausible behaviors but struggle with maintaining long-term coherence. To address this, agents use plans to keep their behavior consistent over time. Plans describe a sequence of future actions, including location, starting time, and duration. Plans are stored in the memory stream, allowing agents to consider observations, reflections, and plans together when deciding how to behave. Agents can also change their plans midstream if necessary, ensuring a more believable sequence of actions.

To create realistic and interesting plans for agents, the approach starts top-down and recursively generates more detail. First, an initial plan is created, outlining the day's agenda based on the agent's summary description and previous day's activities. This plan is saved in the memory stream and then decomposed into finer-grained actions in hour-long chunks, and then further into 5-15 minute chunks. The level of granularity can be adjusted as desired, allowing for detailed and engaging plans for the agent's activities.

**Reacting and Updating Plans**. Generative agents operate in an action loop, perceiving the world around them and storing observations in their memory stream. The language model uses these observations to decide whether the agent should continue with their existing plan or react to the situation. When a reaction is needed, the context summary is generated using relevant queries. The agent's existing plan is regenerated starting from the time of the reaction. If the action involves interaction between agents, dialogue between them is generated.

An example of a prompt:

> [Agent’s Summary Description]
> 
> It is February 13, 2023, 4:56 pm. John Lin’s status: John is back home early from work.
> 
> Observation: John saw Eddy taking a short walk around his workplace.
> 
> Summary of relevant context from John’s memory: Eddy Lin is John’s Lin’s son. Eddy Lin has been working on a music composition for his class. Eddy Lin likes to walk around the garden when he is thinking about or listening to music.
> 
> Should John react to the observation, and if so, what would be an appropriate reaction?

**Dialogue**.

![Dialogue](https://andlukyane.com/images/paper_reviews/ishb/2023-04-22_11-27-51.jpg)

### Sandbox environment implementation

The Smallville sandbox game environment is built using the Phaser web game development framework, with imported visual sprites, environment and collision maps. A server supplements the framework to make the sandbox information available to generative agents, enabling them to move and influence the environment. The server maintains a JSON data structure with information about each agent and updates it with any changes. Agents receive information about objects and other agents within their visual range to react appropriately. End users initialize a new agent with a natural language description, which is split into a set of initial memories to determine the agent's behavior. As agents gain experience, their memory stream grows and their behavior evolves.

To ground generative agents' reasoning to the sandbox world, the environment is represented as a tree data structure, with edges indicating containment relationships. This tree is converted into natural language for agents. Agents build individual tree representations of the environment as they navigate, updating it when they perceive new areas. To determine the appropriate location for each action, the agent's environment tree is traversed and flattened into natural language to prompt the language model, recursively finding the most suitable area. Traditional game path algorithms animate the agent's movement to the selected location. When an agent executes an action on an object, the language model is prompted to determine the change in the object's state, such as a coffee machine switching from "off" to "brewing coffee".

### Controlled evaluation

Generative agents aim to produce believable behavior based on their environment and experiences. The evaluation of these agents is done in two stages. First, a controlled evaluation assesses individual agent responses to understand whether they generate believable behavior in specific contexts. Then, an end-to-end analysis of the agent community over two full days examines their emergent behavior as a collective, including errors and boundary conditions, to see if they can demonstrate information diffusion, relationship formation, and agent coordination.

#### Evaluation Procedure

To assess generative agents in Smallville, agents are "interviewed" to probe their abilities to maintain self-knowledge, retrieve memory, generate plans, react, and reflect. The dependent variable is the believability of the behavior. The interview includes five question categories, each designed to assess one of the key areas. Agents are sampled from the end of a two game-day simulation with the full architecture. To gather feedback on the believability of the responses, 100 human evaluators are recruited to watch a replay of a randomly chosen agent's life in Smallville. The evaluators rank the believability of the interview responses generated by four different agent architectures and a human author condition for the same agent, in a within-subjects design experiment.

#### Conditions

The generative agent architecture was compared to three ablated architectures and a human-generated condition. The ablated architectures had limited access to memory types like observations, reflections, and plans. The no observation, no reflection, no planning condition represents the previous state of the art for agents created through large language models. To provide a human baseline, a unique crowdworker roleplayed as an agent, watched a replay of the agent's sandbox life, and inspected its memory stream. They then authored responses to interview questions in the voice of the agent. The human-generated responses were manually inspected to ensure they met a baseline expectation in quality. This comparison aimed to determine whether the architecture passes a basic level of behavioral competency.

#### Analysis

The experiment produced 100 sets of rank data comparing the believability of each condition. TrueSkill ratings were calculated to provide an interpretable comparison. To investigate statistical significance, the Kruskal-Wallis test and Dunn post-hoc test were used, adjusting p-values with the Holm-Bonferroni method. Additionally, an inductive analysis was conducted by the first author to study qualitative distinctions between responses in each condition. Qualitative open coding was used in two phases: generating codes representing generated responses at the sentence level, and synthesizing codes to extract higher-level themes. These themes were used to compare the types of responses generated in the study.

### Results

![Results](https://andlukyane.com/images/paper_reviews/ishb/2023-04-22_11-37-08.jpg)

The findings suggest that the full generative agent architecture produces the most believable behavior among all study conditions. Performance degraded with the removal of each component in the ablation conditions.

Generative agents with a complete memory module can recall past experiences and answer questions consistently with their self-knowledge. However, their memory is not flawless. They can fail to retrieve correct instances, leading to incomplete or uncertain responses. While agents rarely fabricate knowledge, they might hallucinate embellishments to their knowledge or base their responses on the world knowledge encoded in the language model. This could lead to inaccuracies or inconsistencies in their responses.

Reflection is crucial for generative agents to synthesize their experiences and make informed decisions. Access to reflection memories allows agents to confidently respond to questions, drawing from past interactions and knowledge. In the example provided, Maria Lopez was able to suggest a birthday gift for Wolfgang Schulz based on his interests when she had access to reflection memories.

### Emergent Social Behaviors

![Diffusion](https://andlukyane.com/images/paper_reviews/ishb/2023-04-22_11-41-24.jpg)

The study investigates information diffusion among generative agents in a simulated world by tracking the spread of two pieces of information: Sam's candidacy for village mayor and Isabella's Valentine's Day party. At the start, only Sam and Isabella were aware of these events. After two game days, the 25 agents were interviewed, and their responses were analyzed to determine whether they were aware of the information. Responses were labeled "yes" if they indicated knowledge and "no" if they did not. The study also verified that agents did not hallucinate their responses by checking their memory streams. The percentage of agents holding the information at the end of the simulation was reported.

The study also examines relationship formation and coordination among agents. Agents were asked about their knowledge of other agents at the beginning and end of the simulation. Relationships were considered formed if both agents knew each other. An undirected graph was created with vertices representing agents and edges representing mutual knowledge. Network density was calculated to measure relationship formation.

Additionally, the study investigates agents' ability to coordinate for group activities, such as Isabella's Valentine's Day party. Agents needed to hear about the event and plan to attend at the right time and location. The number of agents who actually showed up to the party after hearing about it was reported.

The study observed emergent outcomes in the simulation. Without user intervention, agents who knew about Sam's mayoral candidacy increased from 4% to 32%, and those who knew about Isabella's party increased from 4% to 48%. Network density increased from 0.167 to 0.74, indicating relationship formation. Only 1.3% of agent responses regarding their awareness of other agents were hallucinated.

In terms of coordination, five out of twelve invited agents attended Isabella's party. Of the seven who didn't attend, three cited conflicts and four expressed interest but didn't plan to come on the day of the party.

###Boundaries and Errors

The study identified three common modes of erratic behavior in agents that could be addressed in future research:

* As agents learned about more locations, they faced challenges in choosing the most relevant information and appropriate space for their actions, which could make their behavior less believable over time. For example, at first they had lunch in the cafe, but later on, after learning about a nearby bar, they opted to go there.
* Erratic behaviors resulted from agents misclassifying proper behavior due to physical norms of locations not being properly conveyed in natural language. For example, several persons visited dorm bathrooms concurrently. This could be addressed by adding norms to the state of the locations.
* Instruction tuning made agents overly polite and cooperative, resulting in overly formal dialogues and agents rarely refusing suggestions from others, even if they didn't align with their interests. This also led to agents' interests being shaped by others over time.

### Discussion

#### Applications of Generative Agents

Generative agents have potential applications beyond sandbox demonstrations, including populating online forums, virtual reality metaverses, and even physical spaces as social robots. This could lead to powerful simulations of human behavior for testing social systems and theories and creating interactive experiences. Another application is in human-centered design processes, where generative agents can act as proxies for users, learning their behavior patterns and preferences. This would allow for more personalized and effective technological experiences, such as automatic coffee brewing, helping with morning routines, and adjusting ambient settings to match a user's mood.

#### Future Work and Limitations

Future research on generative agents can focus on improving the architecture, performance, cost-effectiveness, and real-time interactivity. Enhancements can include fine-tuning the retrieval module and parallelizing agents. As underlying models improve, so will the agents' performance. Evaluations should extend to longer timescales and explore varying models and hyperparameters. Addressing biases and ensuring value alignment are crucial, as well as accounting for data deserts that affect marginalized populations. Generative agents' robustness must be tested to tackle issues like prompt hacking, memory hacking, and hallucination. Adopting mitigations from large language models will help improve the resilience of generative agents.

#### Ethics and Societal Impact

Generative agents raise ethical concerns, including the risk of parasocial relationships, the impact of errors, exacerbation of existing AI risks, and over-reliance. To address these concerns:

* Generative agents should disclose their computational nature and be value-aligned to avoid inappropriate behaviors.
* In different application domains, it is important to follow best practices in human-AI design to understand errors and their impact on user experience.
* Mitigate risks associated with deepfakes, misinformation, and tailored persuasion by maintaining audit logs of inputs and outputs, allowing detection and intervention against malicious use.
* Generative agents should not displace human input in design processes but be used for early-stage prototyping or testing theories difficult or risky to test with human participants.
* Adhering to these principles ensures ethical and socially responsible deployment of generative agents.