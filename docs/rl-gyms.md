there are several open-source, long-horizon Reinforcement Learning (RL) gyms specifically designed for coding, software engineering (SWE), and LLM agent training. These platforms focus on multi-turn, complex tasks where agents must navigate file systems, run tests, and debug code over many steps. 
Here are the primary open-source coding long-horizon RL gyms:
1. AgentGym-RL (by Renmin University & others) https://www.emergentmind.com/topics/agentgym-r2e-gym

Overview: A comprehensive, open-source framework designed to train LLM agents from scratch using RL for long-horizon, multi-turn decision-making.
Focus: It focuses on real-world scenarios, including coding, web navigation, and tool use, supporting "Staged Training" to stabilize long-horizon training.
Key Features: Modular, scalable, and includes datasets for training, evaluation, and self-evolution. 
arXiv
arXiv
 +3
2. ARES (Agentic Research and Evaluation Suite) https://withmartian.com/post/ares-open-source-infrastructure-for-online-rl-on-coding-agents#:~:text=ARES%20is%20an%20open%2Dsource,ARES%20closes%20that%20gap.

Overview: An open-source, RL-first infrastructure specifically for training coding agents with online RL.
Focus: Designed to enable real exploration and fast feedback, supporting massively parallel asynchronous rollouts.
Key Features: Ships with tens of thousands of verifiable coding tasks, including SWE-Bench Verified. 
Withmartian
Withmartian
3. GEM (General Experience Maker)
https://arxiv.org/abs/2510.01051

Overview: A recently released open-source gym for agentic LLMs, designed to handle long-horizon, multi-turn tasks (over 100 turns).
Focus: It acts as a standard, unified interface for training, similar to OpenAI Gym, but for LLM agents.
Key Features: Includes support for Python execution, web search, and Model Context Protocol (MCP) tools. It works with popular RL frameworks like OpenRLHF, Verl, and Oat. 
arXiv
arXiv
 +4
4. DeepSWE (by Together AI & Agentica) 
Overview: A fully open-sourced suite for training coding agents, including the dataset, code, and evaluation logs.
Focus: Specifically targeting Software Engineering (SWE) tasks using RL (rLLM).
Key Features: Achieves high performance on SWE-Bench-Verified by using reinforcement learning to train models to navigate complex codebases. 
Together AI
Together AI
5. SWE-bench & Related Tools
https://arxiv.org/abs/2509.08755#:~:text=Developing%20autonomous%20LLM%20agents%20capable,encourage%20diverse%20problem%2Dsolving%20strategies.
Overview: While often used for benchmarking, many implementations (such as SWE-agent or those within ARES) use the SWE-bench environment for RL training.
Focus: Solving real-world GitHub issues.
Long-Horizon Nature: Agents must interact with a terminal, edit files, and run tests, which often involves 10-50+ steps to resolve a bug. 
Withmartian
Withmartian
 +4
Key Characteristics of These Gyms
Verifiable Rewards: Instead of a human-in-the-loop, these gyms use automated feedback, such as passing unit tests, as the reward signal.
Dockerized Sandboxes: To allow agents to execute code safely, these environments are often containerized.
Multi-Turn Interaction: Unlike single-prompt completion, these environments require long-horizon planning and reasoning over many steps. 
Patronus AI
Patronus AI
 +4