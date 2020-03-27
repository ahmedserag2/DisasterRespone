# Disaster Response Pipeline Project

### Project Motivation:
This project is aimed to visualize and classify disaster messages from the DisasterResponse data sets.

### project phases:
In this project there were 3 phases:
1. build an ETL pipeline to load,clean and save data from the database
2. buid a ML pipeline to deploy a clasification model
3. the flask app which uses the ML model to classify messages and runs visualizations on the data set

### libraries and frameworks used
- pandas
- sklearn
- numpy
- plotly
- flask(framework)

### Acknowledgments
Udacity for making such a complete nanodegree
figure8 for providing the datasets and making this project possible

### runnning:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
