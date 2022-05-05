# -*- coding: utf-8 -*-

import numpy as np
import pickle
import pandas as pd
import xgboost as xgb

class predict_model:
    def __init__(self, model_path = "model.pkl", scaler_path = 'scaler.pkl'):
        self.model = self.load_model(model_path)
        self.scaler = self.load_model(scaler_path)

    def load_model(self, model_path):
        model = pickle.load(open(model_path, 'rb'))
        return model

    def polyFeatures2(self, X, p):
        p1_X = X.copy()
        cal_X = X.copy()
        for power in range(2,p+1):
            new_p_X = p1_X**power
            cal_X = np.concatenate((cal_X, new_p_X), axis=1)
        return cal_X     
    
    def predict(self, data_df):
        predict_data = self.polyFeatures2(data_df, 2)
        scaler = self.scaler
        predict_data_t = scaler.transform(predict_data)
        
        predictions = self.model.predict(predict_data_t)
        result = np.exp(predictions)
        return result

class perp_data:
    def __init__(self):
        self.actor_df = pd.read_csv('actor.csv')
        self.director_df = pd.read_csv('director.csv')
        self.writer_df = pd.read_csv('writer.csv')
        
    def get_director_info(self, director_name):
        if director_name == None:
            director_1_no_movie = np.mean(self.director_df['director_no_movie'])
            director_1_avg_rating = np.mean(self.director_df['director_avg_rating'])
            director_1_avg_votes = np.mean(self.director_df['director_avg_votes'])
            director_1_age = np.mean(self.director_df['director_age'])
        else:
            director_1_no_movie = float(self.director_df[self.director_df['Names'] == director_name]['director_no_movie'])
            director_1_avg_rating = float(self.director_df[self.director_df['Names'] == director_name]['director_avg_rating'])
            director_1_avg_votes = float(self.director_df[self.director_df['Names'] == director_name]['director_avg_votes'])
            director_1_age = float(self.director_df[self.director_df['Names'] == director_name]['director_age'])
        return director_1_no_movie, director_1_avg_rating, director_1_avg_votes, director_1_age 

    def get_actor_info(self, actor_name):
        if actor_name == None:
            actor_avg_rating = np.mean(self.actor_df['actor_avg_rating'])
            actor_avg_votes = np.mean(self.actor_df['actor_avg_votes'])
            actor_orcar_winner = np.mean(self.actor_df['oscar_winner'])
            actor_oscar_nominate = np.mean(self.actor_df['oscar_nominate'])
        else:
            actor_avg_rating = float(self.actor_df[self.actor_df['Names'] == actor_name]['actor_avg_rating'])
            actor_avg_votes = float(self.actor_df[self.actor_df['Names'] == actor_name]['actor_avg_votes'])
            actor_orcar_winner = float(self.actor_df[self.actor_df['Names'] == actor_name]['oscar_winner'])
            actor_oscar_nominate = float(self.actor_df[self.actor_df['Names'] == actor_name]['oscar_nominate'])
        return actor_avg_rating, actor_avg_votes, actor_orcar_winner, actor_oscar_nominate

    def get_writter_info(self, writter_name):
        if writter_name == None:
            writer_avg_votes = np.mean(self.writer_df['writer_avg_votes'])
            writer_avg_rating = np.mean(self.writer_df['writer_avg_rating'])
            writer_age = np.mean(self.writer_df['writer_age'])
        else:
            writer_avg_votes = float(self.writer_df[self.writer_df['Names'] == writter_name]['writer_avg_votes'])
            writer_avg_rating = float(self.writer_df[self.writer_df['Names'] == writter_name]['writer_avg_rating'])
            writer_age = float(self.writer_df[self.writer_df['Names'] == writter_name]['writer_age'])
        return writer_avg_votes, writer_avg_rating, writer_age

    def get_universe(self):
        universe_dict = dict()
        universe_dict['Director'] = self.director_df['Names'].tolist()
        universe_dict['Actor'] = self.actor_df['Names'].tolist()
        universe_dict['Rating'] = ['p', 'pg', 'pg-13', 'nc-17', 'r']
        universe_dict['Generes'] = ['action' ,'drama', 'documentary', 'comedy', 'thriller', 'horror', 'sci_fi', 'mystery', 
                                    'biography', 'adventure', 'romance', 'crime', 'animation', 'musical', 'fantasy', 
                                    'sport', 'western', 'war', 'family', 'music', 'other']
        
        universe_dict['Countries Of Origin'] = ['united states', 'united kingdom', 'france' ,'india', 'other']
        universe_dict['Languages'] = ['english', 'french', 'spanish', 'japanese', 'german', 'other']
        universe_dict['Filming Locations'] = ['california', 'new york', 'los angeles', 'canada', 'england', 'other']
        universe_dict['Production Companies'] = ['columbia', 'universal pictures', 'warner bros', 'paramount', 
                                                 'twentieth century fox', 'mg', 'new line cinema', 'walt disney', 'other']
        universe_dict['Writer'] = self.writer_df['Names'].tolist()
        return universe_dict


    def get_predict_data(self, official_sites, runtimeminutes, director_name, actor_1_name, actor_2_name, rating, genres_list,
                          day, month, year, countries_of_origin, languages_list, filming_locations_list, production_companies, writter_name):
        
        columns_name = ['official sites', 'runtimeminutes', 'director_1_no_movie',
               'director_1_avg_rating', 'director_1_avg_votes', 'director_1_age',
               'actor_1_avg_rating', 'actor_1_avg_votes', 'actor_2_avg_rating',
               'actor_2_avg_votes', 'actor_oscar_winner', 'actor_oscar_nominate',
               'rating_pg-13', 'rating_nc-17', 'rating_pg', 'rating_unrated',
               'rating_g', 'rating_13-15', 'rating_18-20', 'rating_passed',
               'rating_approved', 'rating_p', 'rating_(banned)', 'genres_action',
               'genres_drama', 'genres_documentary', 'genres_comedy',
               'genres_thriller', 'genres_horror', 'genres_sci-fi', 'genres_mystery',
               'genres_biography', 'genres_adventure', 'genres_romance',
               'genres_crime', 'genres_animation', 'genres_musical', 'genres_fantasy',
               'genres_sport', 'genres_western', 'genres_war', 'genres_family',
               'genres_music', 'day', 'month', 'year',
               'countries of origin_united states',
               'countries of origin_united kingdom', 'countries of origin_france',
               'countries of origin_india', 'languages_english', 'languages_french',
               'languages_spanish', 'languages_japanese', 'languages_german',
               'filming locations_california', 'filming locations_new york',
               'filming locations_los angeles', 'filming locations_canada',
               'filming locations_england', 'production companies_universal pictures',
               'production companies_warner bros', 'production companies_columbia',
               'production companies_paramount',
               'production companies_twentieth century fox', 'production companies_mg',
               'production companies_new line cinema',
               'production companies_walt disney', 'writer_avg_votes',
               'writer_avg_rating', 'writer_age']
        
        official_sites = float(official_sites)
        runtimeminutes = float(runtimeminutes)
        
        director_1_no_movie, director_1_avg_rating, director_1_avg_votes, director_1_age = self.get_director_info(director_name)

        actor_1_avg_rating, actor_1_avg_votes, actor_1_oscar_winner, actor_1_oscar_nominate = self.get_actor_info(actor_1_name)
        actor_2_avg_rating, actor_2_avg_votes, actor_2_oscar_winner, actor_2_oscar_nominate = self.get_actor_info(actor_2_name)

        actor_oscar_winner = actor_1_oscar_winner + actor_2_oscar_winner
        actor_oscar_nominate = actor_1_oscar_nominate + actor_2_oscar_nominate

        rating_pg_13 = 1 if rating == 'pg-13' else 0
        rating_nc_17 = 1 if rating == 'nc-17' else 0
        rating_pg = 1 if rating == 'pg' else 0
        rating_unrated = 1 if rating == 'unrated' else 0
        rating_g = 1 if rating == 'g' else 0
        rating_13_15 = 1 if rating == '13-15' else 0
        rating_18_20 = 1 if rating == '18-20' else 0
        rating_passed = 1 if rating == 'passed' else 0
        rating_approved = 1 if rating == 'approved' else 0
        rating_p = 1 if rating == 'p' else 0
        rating_banned = 1 if rating == 'banned' else 0

        genres_action = 1 if 'action' in genres_list else 0
        genres_drama = 1 if 'drama' in genres_list else 0
        genres_documentary = 1 if 'documentary' in genres_list else 0
        genres_comedy = 1 if 'comedy' in genres_list else 0
        genres_thriller = 1 if 'thriller' in genres_list else 0
        genres_horror = 1 if 'horror' in genres_list else 0
        genres_sci_fi = 1 if 'sci_fi' in genres_list else 0
        genres_mystery = 1 if 'mystery' in genres_list else 0
        genres_biography = 1 if 'biography' in genres_list else 0
        genres_adventure = 1 if 'adventure' in genres_list else 0
        genres_romance = 1 if 'romance' in genres_list else 0
        genres_crime = 1 if 'crime' in genres_list else 0
        genres_animation = 1 if 'animation' in genres_list else 0
        genres_musical = 1 if 'musical' in genres_list else 0
        genres_fantasy = 1 if 'fantasy' in genres_list else 0
        genres_sport = 1 if 'sport' in genres_list else 0
        genres_western = 1 if 'western' in genres_list else 0
        genres_war = 1 if 'war' in genres_list else 0
        genres_family = 1 if 'family' in genres_list else 0
        genres_music = 1 if 'music' in genres_list else 0

        day = int(day)
        month = int(month)
        year = int(year)
        
        countries_of_origin_united_states = 1 if countries_of_origin == 'united states' else 0
        countries_of_origin_united_kingdom = 1 if countries_of_origin == 'united kingdom' else 0
        countries_of_origin_france = 1 if countries_of_origin == 'france' else 0
        countries_of_origin_india = 1 if countries_of_origin == 'india' else 0

        languages_english = 1 if 'english' in languages_list else 0
        languages_french = 1 if 'french' in languages_list else 0
        languages_spanish = 1 if 'spanish' in languages_list else 0
        languages_japanese = 1 if 'japanese' in languages_list else 0
        languages_german = 1 if 'german' in languages_list else 0
        
        filming_locations_california = 1 if 'california' in filming_locations_list else 0
        filming_locations_new_york = 1 if 'new york' in filming_locations_list else 0
        filming_locations_los_angeles = 1 if 'los angeles' in filming_locations_list else 0
        filming_locations_canada = 1 if 'canada' in filming_locations_list else 0
        filming_locations_england = 1 if 'england' in filming_locations_list else 0

        production_companies_universal_pictures = 1 if 'universal pictures' == production_companies else 0
        production_companies_warner_bros = 1 if 'warner bros' == production_companies else 0
        production_companies_columbia = 1 if 'columbia' == production_companies else 0
        production_companies_paramount = 1 if 'paramount' == production_companies else 0
        production_companies_twentieth_century_fox = 1 if 'twentieth century fox' == production_companies else 0
        production_companies_mg = 1 if 'mg' == production_companies else 0
        production_companies_new_line_cinema = 1 if 'new line cinema' == production_companies else 0
        production_companies_walt_disney = 1 if 'walt disney' == production_companies else 0
        
        writer_avg_votes, writer_avg_rating, writer_age = self.get_writter_info(writter_name)

        return pd.DataFrame(data = [[official_sites, runtimeminutes, director_1_no_movie, director_1_avg_rating, 
                                     
                                     director_1_avg_votes, director_1_age, actor_1_avg_rating, actor_1_avg_votes,
                                     actor_2_avg_rating, actor_2_avg_votes, actor_oscar_winner, actor_oscar_nominate, rating_pg_13, 
                                     rating_nc_17, rating_pg, rating_unrated, rating_g, rating_13_15, rating_18_20, 
                                     rating_passed, rating_approved, rating_p, rating_banned, genres_action, 
                                     genres_drama, genres_documentary, genres_comedy, genres_thriller, genres_horror, 
                                     genres_sci_fi, genres_mystery, genres_biography, genres_adventure, genres_romance, 
                                     genres_crime, genres_animation, genres_musical, genres_fantasy, genres_sport, genres_western, 
                                     genres_war, genres_family, genres_music, day, month, year, countries_of_origin_united_states, 
                                     countries_of_origin_united_kingdom, countries_of_origin_france, countries_of_origin_india, 
                                     languages_english, languages_french, languages_spanish, languages_japanese, languages_german, 
                                     filming_locations_california, filming_locations_new_york, filming_locations_los_angeles,
                                     filming_locations_canada, filming_locations_england, production_companies_universal_pictures, 
                                     
                                     production_companies_warner_bros, production_companies_columbia, production_companies_paramount, 
                                     production_companies_twentieth_century_fox, production_companies_mg, production_companies_new_line_cinema, 
                                     production_companies_walt_disney, writer_avg_votes, writer_avg_rating, writer_age]], columns = columns_name)


