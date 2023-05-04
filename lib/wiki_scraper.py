# Eric Tria (emt4wf@virginia.edu) DS 5001 Spring 2023

import pandas as pd
import requests
import time
import re
from bs4 import BeautifulSoup

class WikiScraper:
    '''
    A class that generates a LIB_EPISODE table from TV Series on Wikipedia
    '''
    max_retries = 10

    quote_regex = '"([^"]*)"'
    parenthesis_regex = '\(([^)]*)\)'
    non_numeric_regex = '[^0-9]+'

    def __init__(self, user_agent):
        '''
        Purpose: Initiates the class
        
        INPUTS:
        user_agent - str user agent used for web scraping
        '''
        self.user_agent = user_agent
    
    def scrape_episode_list(self, series_url, column_map, series_id, season_limit = None):
        '''
        Purpose: Scrapes full details from the Wiki episode list page
        
        INPUTS:
        series_url - str Wiki episode list url
        column_map - dict mapping of the columns to be scraped
        series_id - str series id
        season_limit - int limit of seasons to scrape
        
        OUTPUTS:
        final_df - Pandas dataframe with the episode details
        '''
        r = requests.get(
            series_url, 
            headers = {'User-agent': self.user_agent}
        )
        soup = BeautifulSoup(r.text, features = 'lxml')

        season_tables = soup.find_all('table', {'class': 'wikitable plainrowheaders wikiepisodetable'})

        if season_limit:
            season_tables = season_tables[:season_limit]

        season_dfs = []
        for idx, season in enumerate(season_tables):
            season_details = self.__scrape_season_table(season, column_map)
            season_df = pd.DataFrame(season_details)
            season_df['season_id'] = idx + 1
            season_dfs.append(season_df)
        
        final_df = pd.concat(season_dfs)
        final_df['series_id'] = series_id
        final_df['episode_id'] = final_df['episode_id'].astype(int)
        final_df['us_viewers'] = final_df['us_viewers'].astype(float)
        final_df = final_df.set_index(['series_id', 'season_id', 'episode_id'])
        return final_df

    def __scrape_season_table(self, season_table, column_map):
        '''
        Purpose: Private method for scraping the season tables
        
        INPUT:
        season_table - bs4 soup of season table
        column_map - dict of column mapping
        
        OUTPUT:
        season_ep_details - list of dict
        '''
        season_eps = season_table.find_all('tr', {'class': 'vevent'})

        season_ep_details = []
        for season_ep in season_eps:
            season_ep_detail = self.__scrape_season_ep(season_ep, column_map)
            season_ep_details.extend(season_ep_detail)

        return season_ep_details
    
    def __scrape_season_ep(self, season_ep, column_map):
        '''
        Purpose: Private method for scraping the rows of the season table
        
        INPUT:
        season_ep - bs4 soup of episode row
        column_map - dict of column mapping
        
        OUTPUT:
        ep_details - dict of scraped information per episode
        '''
        ep_cols = season_ep.find_all('td')

        episode_idx = column_map['episode_id']
        title_idx = column_map['episode_title']
        director_idx = column_map['director']
        writers_idx = column_map['writers']
        date_idx = column_map['date']
        viewers_idx = column_map['us_viewers']

        try:
            # Handle two-part episodes
            split_char = '-'
            episode_id_soup = ep_cols[episode_idx]
            for split in episode_id_soup.find_all('hr'):
                    split.replace_with(split_char)
            
            episode_id = episode_id_soup.text
            writers = ep_cols[writers_idx].text
            title = ep_cols[title_idx].text
            date = ep_cols[date_idx].text

            ep_details = {
                'episode_title': re.search(self.quote_regex, title).group(1) if re.search(self.quote_regex, title) else title,
                'director': ep_cols[director_idx].text,
                'writers': writers,
                'first_writer': writers.split('&')[0].strip() if '&' in writers else writers,
                'date': re.search(self.parenthesis_regex, date).group(1) if re.search(self.parenthesis_regex, date) else date,
                'us_viewers': ep_cols[viewers_idx].text.split('[')[0],
            }


            episode_id = re.sub('[^0-9]+', split_char, episode_id)
            # Handle two-part episodes
            if split_char in episode_id:
                eps = episode_id.split(split_char)
                ep_1 = eps[0]
                ep_2 = eps[1]

                ep_details_2 = ep_details.copy()
                ep_details['episode_id'] = ep_1
                ep_details_2['episode_id'] = ep_2
                return [ep_details, ep_details_2]
            else:
                ep_details['episode_id'] = episode_id
                return [ep_details]

        except Exception as e:
            print('ERROR: ', ep_cols)
            raise(e)