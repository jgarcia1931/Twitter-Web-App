FROM continuumio/anaconda3:4.4.0
COPY ./flask_demo /var/www/python/
EXPOSE 8000
WORKDIR /var/www/python/
RUN pip install -r requirements.txt
CMD python twitter_app.py