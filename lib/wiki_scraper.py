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

    def __init__(self, user_agent):
        self.user_agent = user_agent
    
    def scrape_episode_list(self, series_url, column_map, season_limit = None):
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
        return final_df

    def __scrape_season_table(self, season_table, column_map):
        season_eps = season_table.find_all('tr', {'class': 'vevent'})

        season_ep_details = []
        for season_ep in season_eps:
            season_ep_detail = self.__scrape_season_ep(season_ep, column_map)
            season_ep_details.extend(season_ep_detail)

        return season_ep_details
    
    def __scrape_season_ep(self, season_ep, column_map):
        ep_cols = season_ep.find_all('td')

        episode_idx = column_map['episode_id']
        title_idx = column_map['episode_title']
        director_idx = column_map['director']
        writers_idx = column_map['writers']
        date_idx = column_map['date']
        viewers_idx = column_map['us_viewers']

        try:
            # Handle two-part episodes
            episode_id_soup = ep_cols[episode_idx]
            for split in episode_id_soup.find_all('hr'):
                    split.replace_with('|')
            
            episode_id = episode_id_soup.text
            writers = ep_cols[writers_idx].text
            title = ep_cols[title_idx].text
            date = ep_cols[date_idx].text

            ep_details = {
                # 'episode_id': ep_cols[episode_idx].text,
                'epsiode_title': re.search(self.quote_regex, title).group(1) if re.search(self.quote_regex, title) else title,
                'director': ep_cols[director_idx].text,
                'writers': writers,
                'first_writer': writers.split('&')[0].strip() if '&' in writers else writers,
                'date': re.search(self.parenthesis_regex, date).group(1) if re.search(self.parenthesis_regex, date) else date,
                'us_viewers': ep_cols[viewers_idx].text.split('[')[0],
            }


            # Handle two-part episodes
            if '|' in episode_id:
                eps = episode_id.split('|')
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