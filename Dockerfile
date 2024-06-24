FROM apache/airflow:2.9.1


# Install git and other necessary packages
USER root
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Switch back to the airflow user for application-specific commands
USER airflow
# Set the working directory
WORKDIR /opt/airflow

# Create necessary directories
RUN mkdir -p /opt/airflow/dags /opt/airflow/data1 
# change here data with data1 -----


# Copy the DAGs folder to the container
COPY dags/ /opt/airflow/dags/

COPY data1/ /opt/airflow/data1/  
#--------change data1 with data forr this is testing:


COPY config/ /opt/airflow/config/


# Copy the requirements file
COPY requirements.txt /opt/airflow/


# Install additional Python packages
RUN pip install -r /opt/airflow/requirements.txt

# RUN chmod 644 /opt/airflow/config/advance-anvil-425519-u2-b2800cb795f5.json

#Activate Environment of GCS in docker container with json file permission.
ENV GOOGLE_APPLICATION_CREDENTIALS=/opt/airflow/config/advance-anvil-425519-u2-b2800cb795f5.json

# To access this .json file in container we need to assign permission: 
# RUN chmod 644 /opt/airflow/config/advance-anvil-425519-u2-b2800cb795f5.json

# Initialize Git repository, add files, and commit
USER airflow
RUN git init && \
    git config user.email "fenilsavani55555@gmail.com" && \
    git config user.name "Fenilsavani-and" && \
    git config core.sharedRepository group && \
    git add . && \
    git commit -m "Initial commit"

# Initialize DVC repository, add files, and commit
RUN dvc init 
# change folder with data below:
RUN dvc remote add -d gcs_remote gs://donut-dataset/data1   
RUN git add .dvc/config && \
    git commit -m "DVC setup for data1."  
    
    # Stop tracking data/fenil.txt from Git
RUN git rm -r --cached data1 && \
    git commit -m "Stop tracking data/fenil.txt"

RUN dvc add data1 && \
    git add data1.dvc && \
    git commit -m "Added data file and tracked by git"

RUN dvc push

RUN  rm -rf data1
# RUN  rm -rf .dvc/cache

# # Pull the data from the remote storage

RUN dvc pull

# # Track the pulled data files with Git and commit
RUN git add -f data1 && \
    git commit -m "Pulled data from DVC and added to Git"


# RUN git rm -r --cached data1 && \
#     git commit -m "stop tracking data1"


# Add safe directory exception in Git config
RUN git config --global --add safe.directory /opt/airflow

# Expose the port
EXPOSE 8080

# Run Airflow
CMD ["airflow", "webserver"]

# COPY .dvc /opt/airflow/.dvc/

# COPY .dvcignore /opt/airflow/.dvcignore/

# Copy the .git directory
# COPY .git /opt/airflow/.git/

# COPY data.dvc /opt/airflow/data.dvc 

# Copy the data folder to the container
# COPY data/ /opt/airflow/data/
#------------------
# # Pull the data from the remote storage
# RUN dvc pull -r gcs_remote && \
#     dvc add data && \
#     dvc commit -m "Pulled data from remote storage"

# # Commit DVC metadata file to Git
# RUN git add data.dvc && \
#     git commit -m "Version control added for data.dvc"