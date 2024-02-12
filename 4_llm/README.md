# Autogen AI Assistant

[Ollama](https://github.com/ollama/ollama) - Allows you to download and run LLMs locally.

[Autogen](https://microsoft.github.io/autogen/docs/Getting-Started) - Create personalized agents that specialize in specific task i.e AI coding assistant

[LiteLLM](https://litellm.ai/) - Provides embeddings, error handling, chat completion, function calling

LLMs - We can choose from Mistral, LLAMA2, GPT, Vicuna, Orca

<!--
-- Autogen Tutrial
https://www.youtube.com/watch?v=mUEFwUU0IfE
-- Initialize AutogenStudio
autogenstudio ui --port 8081

Sample:
Build llm that specializes in Quantitative analysis and FinTech, automize research multiple docs, provides auotametaded backtesting

Prompt:
"You are the best Quantitative Analyst in all the world, in fact the best Quant ever known to man, with that in mind please answer the prompts. Take into consideration the research articles you are trained on"


INSTRUCTIONS
Initialize AutoGen in Docker

-- Download Autogen docker img
docker build -f .devcontainer/full/Dockerfile -t autogen_full_img https://github.com/microsoft/autogen.git


-- Mount your current directory
docker run -it -v $(pwd)/myapp:/home/autogen/autogen/myapp autogen_base_img:latest python /home/autogen/autogen/myapp/main.py
docker run -it -v /Documents/GithubFiles/MyQuantsFinance/4_llm:/home/autogen/autogen/myapp autogen_base_img:latest python /home/autogen/autogen/myapp/main.py
docker run -it -v //c/Documents/GithubFiles/MyQuantsFinance/4_llm:/home/autogen/autogen/myapp autogen_base_img:latest python /home/autogen/autogen/myapp/main.py
docker run -it -v //c/Documents/GithubFiles/MyQuantsFinance/4_llm:/home/autogen/autogen/myapp autogen_full_img python /home/autogen/autogen/myapp/autogen_ai.py




 -->
