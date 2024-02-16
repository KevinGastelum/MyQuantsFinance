# Autogen AI Assistant

[Autogen](https://microsoft.github.io/autogen/docs/Getting-Started) - Create personalized agents that specialize in specific task i.e AI Quant Research assistant

```shell
pip install autogenstudio
autogenstudio ui --port 8081 # Access AutoGenStudio in your localhost
```

[Ollama](https://github.com/ollama/ollama) - Allows you to download and run LLMs locally. <!-- curl -fsSL https://ollama.com/install.sh | sh -->

```shell
pip install ollama
ollama serve
ollama run mistral # Download & installs Mistral LLM locally ~4gb size file
```

[LiteLLM](https://litellm.ai/) - Provides embeddings, error handling, chat completion, function calling

```bash
pip install litellm
litellm --model ollama/mistral # Launches Mistral LLM locally
```

LLMs - We can choose from Mistral, LLAMA2, GPT, Vicuna, Orca

<!--
-- Autogen Tutrial
https://www.youtube.com/watch?v=mUEFwUU0IfE

https://blog.finxter.com/how-to-set-up-autogen-studio-with-docker/

-- Initialize AutogenStudio
autogenstudio ui --port 8081

Sample:
Build llm that specializes in Quantitative analysis and FinTech, automize research multiple docs, provides auotametaded backtesting

Prompt:
"You are the best Quantitative Analyst in all the world, in fact the best Quant ever known to man, with that in mind please answer the prompts. Take into consideration the research articles you are trained on"


INSTRUCTIONS: Initializing AutoGen in Docker

-- Download Autogen docker img
docker build -f .devcontainer/full/Dockerfile -t autogen_full_img https://github.com/microsoft/autogen.git

-- MOUNT your current directory
docker run -it -v "$(pwd)":/home/autogen/project autogen_full_img

-- ENTER container
docker exec -it a282e3193d5e bash

-- START/STOP container
docker start a282e3193d5e
docker stop a282e3193d5e
docker rm a282e3193d5e
 -->

```

```
