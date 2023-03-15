# FROM python:3.9
FROM tiangolo/uwsgi-nginx:python3.9

WORKDIR /code
RUN chmod 777 /code
RUN chmod 777 /etc/nginx/nginx.conf
RUN chmod 777 /etc/nginx/
RUN chmod 777 /etc/nginx/conf.d
# RUN chmod 777 /etc/nginx/conf.d/nginx.conf

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME

RUN git clone https://github.com/hazyresearch/meerkat.git
WORKDIR $HOME/meerkat/
RUN git checkout karan/deploy
RUN pip install --upgrade .

# PUT THIS BACK
# RUN pip install meerkat-ml

COPY --chown=user . $HOME/

# Set env variables
RUN mkdir $HOME/logs
RUN chmod 777 $HOME/logs
ENV MEERKAT_LOG_DIR=$HOME/logs
RUN chmod 777 $HOME/
ENV MEERKAT_CONFIG=$HOME/config.yaml
ENV SPACE_AUTHOR_NAME={{ space_author }}
ENV SPACE_REPO_NAME={{ space_repo }}

COPY --chown=user ./{{ script }}.py $HOME/{{ script }}.py

WORKDIR $HOME
CMD ["mk", "run", "/home/user/{{ script }}.py", "--host", "0.0.0.0", "--api-port", "7860"]