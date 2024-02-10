<!-- Initialize AutoGen in Docker

-- Download Autogen docker img
docker build -f .devcontainer/full/Dockerfile -t autogen_full_img https://github.com/microsoft/autogen.git


-- Mount your current directory
docker run -it -v $(pwd)/myapp:/home/autogen/autogen/myapp autogen_base_img:latest python /home/autogen/autogen/myapp/main.py
docker run -it -v /Documents/GithubFiles/MyQuantsFinance/4_llm:/home/autogen/autogen/myapp autogen_base_img:latest python /home/autogen/autogen/myapp/main.py
docker run -it -v //c/Documents/GithubFiles/MyQuantsFinance/4_llm:/home/autogen/autogen/myapp autogen_base_img:latest python /home/autogen/autogen/myapp/main.py
docker run -it -v //c/Documents/GithubFiles/MyQuantsFinance/4_llm:/home/autogen/autogen/myapp autogen_full_img python /home/autogen/autogen/myapp/autogen_ai.py




 -->
