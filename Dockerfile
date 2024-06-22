FROM apache/airflow:2.9.1

# Set the working directory
WORKDIR /opt/airflow

# Create necessary directories
RUN mkdir -p /opt/airflow/dags /opt/airflow/data

# Copy the DAGs folder to the container
COPY dags/ /opt/airflow/dags/

# Copy the data folder to the container
# COPY data/ /opt/airflow/data/

COPY config/ /opt/airflow/config/


# Copy the requirements file
COPY requirements.txt /opt/airflow/


# Install additional Python packages
RUN pip install -r /opt/airflow/requirements.txt

# RUN chmod 644 /opt/airflow/config/advance-anvil-425519-u2-b2800cb795f5.json

#Activate Environment of GCS in docker container with json file permission.
ENV GOOGLE_APPLICATION_CREDENTIALS=/opt/airflow/config/advance-anvil-425519-u2-b2800cb795f5.json

# To access this .json file in container we need to assign permission: 
# chmod 644 /opt/airflow/config/advance-anvil-425519-u2-b2800cb795f5.json


# Expose the port
EXPOSE 8080

# Run Airflow
CMD ["airflow", "webserver"]
