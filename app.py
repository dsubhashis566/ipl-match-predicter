import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
              background-color: #cccccc;
             
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()

teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Delhi Daredevils',
       'Rajasthan Royals', 'Royal Challengers Bangalore',
       'Sunrisers Hyderabad', 'Deccan Chargers', 'Kings XI Punjab',
       'Mumbai Indians', 'Delhi Capitals']

cities = ['Cuttack', 'Chennai', 'Durban', 'Mumbai', 'Kolkata', 'Jaipur',
       'Delhi', 'Bangalore', 'Nagpur', 'Hyderabad', 'Kimberley', 'Raipur',
       'Chandigarh', 'Mohali', 'Ranchi', 'Bengaluru', 'Dharamsala',
       'Port Elizabeth', 'Visakhapatnam', 'Ahmedabad', 'Pune',
       'East London', 'Abu Dhabi', 'Indore', 'Johannesburg', 'Cape Town',
       'Sharjah', 'Centurion', 'Bloemfontein']

pipe = pickle.load(open('ipl_model.pkl','rb'))
st.title('IPL Win Predictor')


col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team',sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team',sorted(teams))

selected_city = st.selectbox('Select host city',sorted(cities))

target = st.number_input('Target',min_value=0,step=1)

col3,col4,col5 = st.columns(3)

with col3:
    score = st.number_input('Score',min_value=0,step=1)
with col4:
    overs = st.number_input('Overs completed',min_value=0,step=1,max_value=20)
with col5:
    wickets = st.number_input('Wickets out',min_value=0,step=1,max_value=10)

if st.button('Predict Probability'):
    if batting_team == bowling_team:
        st.header("Batting and Bowling team cannot be same.")
    elif score>=target:
        st.header(batting_team + "- " + str(round(1 * 100)) + "%")
        st.header(bowling_team + "- " + str(round(0 * 100)) + "%")
    elif overs>=20:
        if score<target:
            st.header(batting_team + "- " + str(round(0 * 100)) + "%")
            st.header(bowling_team + "- " + str(round(1 * 100)) + "%")
    else:
        runs_left = target - score
        balls_left = 120 - (overs*6)
        wickets = 10 - wickets
        crr = score/overs
        rrr = (runs_left*6)/balls_left

        input_df = pd.DataFrame({'batting_team':[batting_team],
                             'bowling_team':[bowling_team],'city':[selected_city],
                             'runs_left':[runs_left],'balls_left':[balls_left],
                             'wickets':[wickets],'target':[target],
                             'crr':[crr],'rrr':[rrr]})

        result = pipe.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]
        st.header(batting_team + "- " + str(round(win*100)) + "%")
        st.header(bowling_team + "- " + str(round(loss*100)) + "%")
        y = np.array(result[0])
        mylabel = [batting_team, bowling_team]
        #plt.pie(y, labels=mylabel)
        fig1, ax1 = plt.subplots()
        ax1.pie(y, labels=mylabel, startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        st.pyplot(fig1)