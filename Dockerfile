FROM supervisely/base-py-sdk:6.69.68

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt