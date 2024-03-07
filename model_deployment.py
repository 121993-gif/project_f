import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.pipeline import Pipeline
from category_encoders import BinaryEncoder

select_page = st.sidebar.radio('Select page', ['Intoduction','Analysis', 'Model Regression'])

if select_page == 'Intoduction':
    
    def main():
        st.title('Jobs and Salaries in Data Science')
        
        #st.image('')
        
        st.write('### introduction to my data :')
        
        st.write('''In an age dominated by data, the field of data science stands as a beacon of innovation and opportunity. Behind every groundbreaking discovery, every insightful trend, lies a team of dedicated professionals armed with the tools of analysis and the power of visualization. Welcome to our exploration of the dynamic landscape of data science careers and salaries, where numbers tell stories and trends reveal themselves in vibrant hues.

                ''')
        
        st.header('Dataset Feature Overview')
        
        st.write('''
                *work_year* : The year in which the data was recorded. This field indicates the temporal context of the data, important for understanding salary trends over time.
        
                *job_title* : The specific title of the job role, like 'Data Scientist', 'Data Engineer', or 'Data Analyst'. This column is crucial for understanding the salary distribution across various specialized roles within the data field. 
                
                *job_category*: A classification of the job role into broader categories for easier analysis. This might include areas like 'Data Analysis', 'Machine Learning', 'Data Engineering', etc.
                
                *salary_currency* : The currency in which the salary is paid, such as USD, EUR, etc. This is important for currency conversion and understanding the actual value of the salary in a global context.
                 
                *salary* : The annual gross salary of the role in the local currency. This raw salary figure is key for direct regional salary comparisons.
                
                *salary_in_usd* : The annual gross salary converted to United States Dollars (USD). This uniform currency conversion aids in global salary comparisons and analyses.
                
                *employee_residence* : The country of residence of the employee. This data point can be used to explore geographical salary differences and cost-of-living variations.

                *experience_level*: Classifies the professional experience level of the employee. Common categories might include 'Entry-level', 'Mid-level', 'Senior', and 'Executive', providing insight into how experience influences salary in data-related roles.

                *employment_type* : Specifies the type of employment, such as 'Full-time', 'Part-time', 'Contract', etc. This helps in analyzing how different employment arrangements affect salary structures.

                *work_setting* : The work setting or environment, like 'Remote', 'In-person', or 'Hybrid'. This column reflects the impact of work settings on salary levels in the data industry.
            
                *company_location* : The country where the company is located. It helps in analyzing how the location of the company affects salary structures.

                *company_size* : The size of the employer company, often categorized into small (S), medium (M), and large (L) sizes. This allows for analysis of how company size influences salary.

                ''')
        
    if __name__ == '__main__':
        main()

        
    
    


if select_page == 'Analysis':
    def main():
        cleaned_df = pd.read_csv('cleaned_df.csv')
        st.write('### Head of Dataframe')
        st.dataframe(cleaned_df.head(10))
        
        tab1, tab2, tab3 = st.tabs(['Univariate Analysis', 'Bivariate Analysis', 'Multivariate Analysis'])
        
        tab1.write('### Univariate Analysis with Histogram for each Feature')
        for col in cleaned_df.columns:
            tab1.plotly_chart(px.histogram(cleaned_df, x=col))
    
        tab2.write('### Does the job category affect the salary in USD?')
        tab2.plotly_chart(px.box(cleaned_df, x='job_category', y='salary_in_usd'))

        tab2.write('### Is there a correlation between work year and salary in USD?')
        tab2.plotly_chart(px.scatter(cleaned_df, x='work_year', y='salary_in_usd'))
        
        tab2.write('### How does employment type relate to job category?')
        tab2.plotly_chart(px.bar(cleaned_df, x='job_category', color='employment_type'))
        
        tab2.write('### Does the company size impact the work setting?')
        tab2.plotly_chart(px.box(cleaned_df, x='work_setting', y='company_size'))
        
        tab2.write('### Is there a difference in salaries based on employee residence?')
        sal_per_res = cleaned_df.groupby('employee_residence')['salary_in_usd'].sum().reset_index().sort_values(by= 'salary_in_usd', ascending= False).head(10)
        tab2.plotly_chart(px.bar(sal_per_res, x='employee_residence', y='salary_in_usd'))

        tab3.write('### How does work experience level, job category, and salary in USD correlate?')
        tab3.plotly_chart(px.histogram(cleaned_df, x='job_category', y='salary_in_usd',color='experience_level'))

        tab3.write('### Is there a relationship between company size, job category, and salary in USD?')
        tab3.plotly_chart(px.box(cleaned_df, x='company_size', y='salary_in_usd',color='job_category'))

        tab3.write('##### How does employment type, work setting, and salary in USD vary together?')
        tab3.plotly_chart(px.box(cleaned_df, x='employment_type', y='salary_in_usd', color='work_setting'))
        
        tab3.write('##### Does the combination of work year, job title, and job category have an impact on salaries in USD?')
        tab3.plotly_chart(px.box(cleaned_df, x='job_category', y='salary_in_usd', color='work_year'))

        tab3.write('##### How do job title, company location, and company size interact in terms of salaries in USD?')
        tab3.plotly_chart(px.scatter(cleaned_df, x='company_size', y='salary_in_usd',color='job_category',size='salary_in_usd'))

        
    if __name__=='__main__':
        main() 

elif select_page == 'Model Regression':
    
    def main(): 
        
        st.title('Model Regression')
        
        pipeline = joblib.load('gb_pipeline.pkl')

        def Prediction(work_year, job_title, job_category, employee_residence, experience_level, employment_type, work_setting, company_location, company_size):
            df = pd.DataFrame(columns=['work_year', 'job_title', 'job_category', 'employee_residence', 'experience_level', 'employment_type', 'work_setting', 'company_location', 'company_size'])
            df.at[0, 'work_year'] = work_year
            df.at[0, 'job_title'] = job_title
            df.at[0, 'job_category'] = job_category
            df.at[0, 'employee_residence'] = employee_residence
            df.at[0, 'experience_level'] = experience_level
            df.at[0, 'employment_type'] = employment_type
            df.at[0, 'work_setting'] = work_setting
            df.at[0, 'company_location'] = company_location
            df.at[0, 'company_size'] = company_size

            result = pipeline.predict(df)[0]
            return result

        # Now we will decide how the user can select each feature
        work_year = st.selectbox('Please provide the number of years', [2020,2021, 2022, 2023])
        job_title = st.text_input('Please write your job title')
        job_category = st.text_input('Please write your job category')
        employee_residence = st.text_input('Please write your employee residence')
        experience_level = st.selectbox('Please select your experience level',['Mid-level', 'Senior', 'Executive', 'Entry-level'] )
        employment_type = st.selectbox('Please select your employment type',['Full-time', 'Part-time', 'Contract', 'Freelance'] )
        work_setting = st.selectbox('Please select your work setting',['Hybrid', 'In-person', 'Remote'] )
        company_location = st.text_input('Please write your company location')
        company_size = st.selectbox('Please select your company size', ['L', 'M', 'S'] )

        if st.button('Predict'):
            result = Prediction(work_year, job_title, job_category, employee_residence, experience_level, employment_type, work_setting, company_location, company_size)

            st.write('### Prediction Result:')
            st.write(f'The predicted result is: {round(np.exp(result), 2)}')

    if __name__=='__main__':
        main() 

    
