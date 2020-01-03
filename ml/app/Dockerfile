FROM python:3.7.2

WORKDIR /home/btc-tweets

ADD ./ ./

RUN pip install xgboost    
RUN pip install plotly
RUN pip install matplotlib
RUN pip install numpy
RUN pip install fire
RUN pip install tensorflow

EXPOSE 80

ENTRYPOINT [ "sh", "start.sh" ]
CMD [ "/bin/sh" ]