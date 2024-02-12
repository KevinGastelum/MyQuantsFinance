FROM python:3.11-slim-bookworm

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        software-properties-common sudo\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Setup a non-root user 'autogen' with sudo access
RUN adduser --disabled-password --gecos '' autogen
RUN adduser autogen sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER autogen
WORKDIR /home/autogen

# Set environment variable
ENV PATH="/home/autogen/.local/bin:$PATH"
# Follow the = with your OpenAI API key (no quotes or anything, just OPENAI_API_KEY=sk-lotsOfLettersAndNumbers)
ENV OPENAI_API_KEY=

# Pre-load popular packages
RUN pip install pyautogen==0.2.8 autogenstudio==0.0.34a0 numpy pandas matplotlib seaborn scikit-learn requests urllib3 nltk pillow pytest beautifulsoup4 litellm

# Expose port
EXPOSE 8081

# Start Command for AutoGen Studio
CMD ["autogenstudio", "ui", "--host", "0.0.0.0", "--port", "8081"]

# command to build the image:
    # docker build -t autogenstudio -f Dockerfile.base .
# command to run the container:
    # docker run -it --rm -p 8081:8081 --name autogenstudio autogenstudio
# Access AutoGen Studio at http://localhost:8081 make sure you don't click the 0.0.0.0:8081 link in the terminal, it won't work!