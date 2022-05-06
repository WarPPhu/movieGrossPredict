# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
from datetime import datetime, date
from model_predict import predict_model, perp_data

file_dir = ''

st.set_page_config(
             page_title="Fundamental Programming Project",
             layout="wide",
             initial_sidebar_state="expanded",
         )

class App:
    def __init__(self):
        self.page = {"Gross Prediction": self.MainPage, 
                     "Appendix" : self.PageAppendix}
        
        self.model = self.read_predict_model()
        self.prep= self.read_prep_data()
        self.universe_dict = self.read_prep_data().get_universe()
        
        self.column_name = ['Gross(M) $', 'Director', 'Actor', 'Writer', 'Rating', 'Genres',
                                                           'Country', 'Laguages', 'Filiming Locations', 'Run Times', 'Release Date',
                                                           'Have Official Site', 'Company']
        

        if 'num' not in st.session_state:
            st.session_state.num = 1
        if 'data_df' not in st.session_state:
            st.session_state.data_df = pd.DataFrame()

    def read_predict_model(self):
        model = predict_model()
        return model
    
    def read_prep_data(self):
        prep = perp_data()
        return prep

    def MainPage(self):
        st.markdown("## Gross Prediction")
        st.write("\n")
        
        director = st.selectbox('Director', options = self.universe_dict['Director'])
        writer = st.selectbox('writer', options = self.universe_dict['Writer'])
        
        actor_1_select, actor_2_select = st.columns((2, 2))
        actor_1 = actor_1_select.selectbox('Actor 1', options = self.universe_dict['Actor'])
        actor_2 = actor_2_select.selectbox('Actor 2', options = self.universe_dict['Actor'])
        
        rating = st.selectbox('rating', options = self.universe_dict['Rating'])
        genres = st.multiselect('Genres', options = self.universe_dict['Generes'])
        
        
        select_date = st.date_input("Released Date", datetime.now().date(), min_value = date(1990,1,1), max_value = date(2030,12,31))

        day = select_date.day
        month = select_date.month
        year = select_date.year
        
        countries_of_origin = st.selectbox('Country', options = self.universe_dict['Countries Of Origin'])
        languages = st.multiselect('Languages', options = self.universe_dict['Languages'])
        filming_locations = st.multiselect('Filming Locations', self.universe_dict['Filming Locations'])
        production_companies = st.selectbox('Company', options = self.universe_dict['Production Companies'])

        run_times = st.slider('Run Time Minutes', min_value = 60, max_value = 180, value = 120)
        official_sites = st.checkbox('Official Sites')

        data_df = self.prep.get_predict_data(official_sites, run_times, 
                                        director, actor_1, actor_2,
                                        rating, genres, day,
                                        month, year, countries_of_origin,
                                        languages, filming_locations, production_companies,
                                        writer)
        
        result = self.model.predict(data_df)[0]
        
        st.write('## Gross Worldwide Prediction ' + "{0:,.0f}".format(result) + ' $')
        
        with st.sidebar:
            st.write('## Gross Worldwide Prediction: ' + "{0:,.0f}".format(result) + ' $')
        show_df = pd.Series([("{0:,.2f}".format(result/1000000)), director, f'{actor_1}, {actor_2}',
                             writer, rating, ', '.join(genres), 
                             countries_of_origin, ', '.join(languages), 
                             ', '.join(filming_locations), run_times,
                             datetime(year, month, day).strftime("%Y-%m-%d"), 
                             'Yes' if official_sites else 'No', 
                             production_companies], index = self.column_name)
        show_df.name = st.session_state.num
        
        if st.button('Add'):
            st.session_state.data_df = pd.concat([st.session_state.data_df,show_df], axis = 1)
            st.session_state.num = st.session_state.num + 1
        if st.button('Clear'):
            st.session_state.data_df = pd.DataFrame()
            st.session_state.num = 1
        st.table(st.session_state.data_df.T)

    def PageAppendix(self):
        st.markdown("## Appendix")
        
        st.write('Ref Data')
        st.write('https://www.imdb.com/')
        
        st.write('Group Member')
        st.write('6480444026, ปภัช สุจิตรัตนันท์, Papat Sujitrattanun')
        st.write('6480448626, ปาลิตา เหลืองพัฒนผดุง, Palita Luengpattanapadung')
        st.write('6480459526, พิเชฐ ชาไชย, Pichet Chachai')
        st.write('6480469826, ภูริณัฐ จันทร์หอม, Phurinut Chanhom')
        st.write('6480484126, วัชรพล สร้างกุศล, Watcharapol Srangkusol')

    def start(self):
        st.title("Fundamental Programming Project")
        with st.sidebar:
            page = st.radio("Select page", tuple(self.page.keys()))
        self.page[page]()


if __name__ == "__main__":
    app = App()
    app.start()
