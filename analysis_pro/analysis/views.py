from django.shortcuts import render
import pandas as pd
import os
from django.conf import settings
import matplotlib.pyplot as plt
import seaborn as sns
from django.http import HttpResponseRedirect
from django.urls import reverse
from django import forms
from scipy.stats import normaltest
import matplotlib
import numpy as np
matplotlib.use('Agg')
from sklearn.preprocessing import MinMaxScaler  

class UploadFileForm(forms.Form):
    file = forms.FileField()

def generate_combinations(arr, r):
    def combine(start, path):
        if len(path) == r:
            result[tuple(path)] = None
            return
        for i in range(start, len(arr)):
            combine(i + 1, path + [arr[i]])
    
    result = {}
    combine(0, [])
    return list(result.keys())



def handle_uploaded_file(f):
    file_path = os.path.join(settings.MEDIA_ROOT, f.name)
    with open(file_path, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
    preprocessed_file_path=data_preprocessing(file_path,f)
    return file_path,preprocessed_file_path



def index(request):
    return render(request, 'index.html')



def data_preprocessing(file_path, f):
    df = pd.read_csv(file_path)
    
    missing_percentage = df.isnull().mean() * 100
    num_columns = len(df.columns)
    total_missing_percentage = missing_percentage.sum() / num_columns
    
    # Handle missing values based on the total missing values percentage
    if total_missing_percentage < 10:
        df.dropna(inplace=True)
    else:
        # Drop columns where the percentage of missing values is more than 30%
        columns_to_drop = missing_percentage[missing_percentage > 30].index
        df.drop(columns=columns_to_drop, inplace=True)
        
        for column in df.columns:
            if df[column].isnull().sum() > 0:
                if df[column].dtype in ['int64', 'float64']:
                    # Check if the column is normally distributed
                    _, p_value = normaltest(df[column].dropna())
                    if p_value > 0.05:
                        # Fill with mean if normally distributed
                        df[column].fillna(df[column].mean(), inplace=True)
                    else:
                        # Fill with median if not normally distributed
                        df[column].fillna(df[column].median(), inplace=True)
                else:
                    # Fill categorical columns with the most frequent value (mode)
                    df[column].fillna(df[column].mode()[0], inplace=True)
                    
    df.drop_duplicates(inplace=True)

    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

    preprocessed_file_path = os.path.join(settings.MEDIA_ROOT, 'preprocessed1', f.name)
    os.makedirs(os.path.dirname(preprocessed_file_path), exist_ok=True)
    df.to_csv(preprocessed_file_path, index=False)
    
    # Scale numerical features for machine learning models if needed
    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Encoding categorical variables if it is to be used for machine learning models
    df = pd.get_dummies(df)

    df = handle_outliers(df, numerical_cols)
    
    # Save the preprocessed file
    preprocessed_file_path = os.path.join(settings.MEDIA_ROOT, 'preprocessed', f.name)
    os.makedirs(os.path.dirname(preprocessed_file_path), exist_ok=True)
    df.to_csv(preprocessed_file_path, index=False)
    
    return preprocessed_file_path




def visuals(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            file_path,preprocessed_file = handle_uploaded_file(request.FILES['file'])
            df = pd.read_csv(file_path)
            df1 = pd.read_csv(preprocessed_file)

            numerical_columns = df.select_dtypes(include=['number','int64', 'float64']).columns
            numerical_columns1=df1.select_dtypes(include=['number','int64', 'float64']).columns
            categorical_columns = df.select_dtypes(include='object').columns
            datetime_columns = df.select_dtypes(include=['datetime']).columns
            categorical_columns = categorical_columns.difference(datetime_columns)
            categorical_columns1 = df1.select_dtypes(include='object').columns


            head = df.head().to_html()
            describe_data = df.describe().to_html()
            missing_values = df.isnull().sum().to_frame(name='Missing Values').to_html()
            duplicate_values =  df.duplicated().sum()
            outlier_counts = count_outliers(df, numerical_columns)
            outlier_counts_html = pd.DataFrame(outlier_counts.items(), columns=['Column', 'Number of Outliers']).to_html(index=False)

            

            mean_values = df[numerical_columns].mean().to_frame(name='Mean').to_html()
            median_values = df[numerical_columns].median().to_frame(name='Median').to_html()
            mode_values = df[categorical_columns].mode().iloc[0].to_frame(name='Mode').to_html()
            std_values = df[numerical_columns].std().to_frame(name='Standard Deviation').to_html()

            hist_paths = []
            hist_choices = []
            plot_paths = []
            plot_choices = []
            pie_paths = []
            pie_choices = []
            scaled_paths=[]
            scaled_choices=[]


            scatter_paths, scatter_choices = create_scatter_plots(df, numerical_columns,'others')
            scaled_scatter_paths,scaled_scatter_choices = create_scatter_plots(df1,numerical_columns1,'scaled')
            scaled_line_paths,scaled_line_choices=create_line_plots(df1,numerical_columns1,'scaled')
            line_paths, line_choices = create_line_plots(df, numerical_columns,'others')
            pie_paths, pie_choices = create_pie_charts(df, categorical_columns,'pie')
            histogram_paths, histogram_choices = create_histograms(df, numerical_columns,'histograms')
            scaled_histograms_path,scaled_histograms_choices=create_histograms(df1,numerical_columns1,'scaled')

            hist_paths.extend(histogram_paths)
            hist_choices.extend( histogram_choices)
            plot_paths.extend(scatter_paths + line_paths )
            plot_choices.extend(scatter_choices + line_choices)
            pie_paths.extend(pie_paths)
            pie_choices.extend(pie_choices)
            scaled_paths.extend(scaled_histograms_path+scaled_line_paths+scaled_scatter_paths)
            scaled_choices.extend(scaled_histograms_choices+scaled_line_choices+scaled_scatter_choices)

            # print(head)
            # print(describe_data)

            return render(request, 'analysis/results.html', {
                'head': head,
                'describe': describe_data,
                'missing_values': missing_values,
                'duplicated_values':duplicate_values,
                'mean_values': mean_values,
                'median_values': median_values,
                'mode_values': mode_values,
                'std_values': std_values,
                'hist_paths': hist_paths,
                'hist_choices': hist_choices,
                'plot_paths':plot_paths,
                'plot_choices': plot_choices,
                'pie_paths': pie_paths,
                'pie_choices':pie_choices,
                'scaled_paths':scaled_paths,
                'scaled_choices':scaled_choices,
                'outlier_counts': outlier_counts_html,
            })
    else:
        form = UploadFileForm()
    return render(request, 'analysis/index.html', {'form': form})

def create_scatter_plots(df, numerical_columns,name):
    plot_paths = []
    plot_choices = []
    col_pairs = list(generate_combinations(numerical_columns, 2))
    
    for col1, col2 in col_pairs:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x=col1, y=col2)
        plt.title(f'Scatter Plot: {col1} vs {col2}')
        plot_name = f'scatter_{col1}_vs_{col2}.png'
        plot_path = os.path.join(settings.MEDIA_ROOT, name, plot_name)
        plt.savefig(plot_path)
        plot_paths.append(plot_path)
        plot_choices.append((plot_name, f'Scatter Plot: {col1} vs {col2}'))
        plt.close()
    
    return plot_paths, plot_choices




def create_line_plots(df, numerical_columns,name):
    plot_paths = []
    plot_choices = []
    
    if 'Date' in df.columns:
        for col in numerical_columns:
            plt.figure(figsize=(10, 6))
            df.groupby('Date')[col].sum().plot(kind='line')
            plt.title(f'{col} Over Time')
            plt.ylabel(f'{col}')
            plot_name = f'line_{col}_over_time.png'
            plot_path = os.path.join(settings.MEDIA_ROOT,name, plot_name)
            plt.savefig(plot_path)
            plot_paths.append(plot_path)
            plot_choices.append((plot_name, f'{col} Over Time'))
            plt.close()
    
    return plot_paths, plot_choices




def create_pie_charts(df, categorical_columns,name):
    plot_paths = []
    plot_choices = []
    
    for col in categorical_columns:
        plt.figure(figsize=(8, 8))
        ax = df[col].value_counts().plot(kind='pie', autopct='%1.1f%%', legend=False)
        ax.get_yaxis().set_visible(False)
        plt.title(f'{col} Distribution')
        plot_name = f'pie_{col}.png'
        plot_path = os.path.join(settings.MEDIA_ROOT,name, plot_name)
        plt.savefig(plot_path)
        plot_paths.append(plot_path)
        plot_choices.append((plot_name, f'{col} Distribution'))
        plt.close()
    
    return plot_paths, plot_choices




def create_histograms(df, numerical_columns,name):
    plot_paths = []
    plot_choices = []
    num_bins_sqrt = int(np.sqrt(len(df)))
    
    for col in numerical_columns:
        print(col)
        plt.figure(figsize=(10, 6))
        df[col].plot(kind='hist', bins=num_bins_sqrt)
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plot_name = f'histogram_{col}.png'
        plot_path = os.path.join(settings.MEDIA_ROOT,name, plot_name)
        plt.savefig(plot_path)
        plot_paths.append(plot_path)
        plot_choices.append((plot_name, f'Histogram of {col}'))
        plt.close()
    
    return plot_paths, plot_choices





def delete_file(request):
    if request.method == 'POST':
        file_name = request.POST.get('file_name')
        if file_name:
            file_path = os.path.join(settings.MEDIA_ROOT, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
        return HttpResponseRedirect(reverse('index')) 




def handle_outliers(df, numerical_cols):
    for col in numerical_cols:
        # Calculate the first (Q1) and third (Q3) quartiles
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define the outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Clip outliers
        df[col] = np.clip(df[col], lower_bound, upper_bound)
    
    return df




def count_outliers(df, numerical_cols):
    outlier_counts = {}
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Count outliers
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_counts[col] = len(outliers)
    
    return outlier_counts


