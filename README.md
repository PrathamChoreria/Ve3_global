# **Data Analysis and Visualization Django App**

## **Overview**
This Django application is designed to handle, preprocess and analyze data files uploaded by users. It provides various functionalities including:

#### Data Preprocessing: 
Handles missing values, removes duplicates, scales numerical features,encodes categorical variables and identifies and handles outliers in numerical columns.Designed for flexibility, the application can process a wide range of CSV files making it an invaluable tool for data preprocessing across different datasets and use cases.
#### Visualization:
Generates scatter plots, line plots, pie charts and histograms from the data. It supports both the raw and preprocessed datasets.**It creates visualization that shows every feature plotted against every other feature.**
#### Added a feature which stores the csv file temporarily and allows us to delete it from directory.

#### Pls find project snapshots in photos folder

## Features
### Summary of data
It shows summary of data such as the first few rows of the .csv file, describe the data, display mean median and mode for the features , the missing values in the .csv file , the number of duplicate data that is dealt and outliers in each feature(column).
### **File Upload** 
Users can upload CSV files for analysis.File is temporarily stored in media folder
### Data Cleaning
Automatically handles missing values and outliers.**Outliers are removed only if we want to remove them.Outliers can be handy in a few data preprocessing processes and anamoly detection so I give user the choice to remove them.**
### Visualization
Generates multiple types of plots for data analysis.It shows line plot, scatter plot, piecharts and also histograms.
### Scalability
Supports scaling of numerical features.
### Categorical Data Handling 
Encodes categorical variables so that they can be used in ML models if needed.

## File Structure
#### analysis/views.py: Handles the file upload, data processing, and visualization.
#### analysis/urls.py: URL routing for the app.
#### templates/: HTML templates for the index and results pages.
#### media/: Directory where uploaded and processed files are stored.**It also stores graph/plot/piecharts/histograms that are generated.It also stores preprocessed csv file along with preprocessed scaled csv file in it.**
#### analysis is the name of the app created in the project analysis_pro


## Setup
### 1. Prerequisites
#### Ensure you have the following installed:
##### 1. Python 3.x
##### 2.Django
##### 3.Pandas
##### 4.Matplotlib
##### 5.Seaborn
##### 6.Scikit-learn/sklearn
##### 7.SciPy

### 2. Installation
##### Clone the Repository

### 3. Create a Virtual Environment
##### python -m venv venv
##### source venv/bin/activate  
##### On Windows use `venv\Scripts\activate`

### 4.Install packages in requirements.txt

### 5. Create a Django project and app if not already done
##### django-admin startproject analysis_pro
##### cd analysis_pro
##### django-admin startapp analysis

### 6.Add the app to your project's settings in myproject/settings.py:

### 7. In analysis_pro/settings.py, add: 
##### MEDIA_URL = '/media/'
##### MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

### 8. Update analysis_pro/urls.py to include media URLs:
##### urlpatterns = [...] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

### 9. Apply migrations
##### python manage.py migrate

### 10. Run Development Environment
##### python manage.py runserver








