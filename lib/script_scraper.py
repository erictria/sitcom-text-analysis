import pandas as pd
import requests
import time
from bs4 import BeautifulSoup

class ScriptScraper:
    '''
    A class that generates a CORPUS from TV Series scripts

    Scrapes scripts from https://subslikescript.com/
    '''

    main_page = 'https://subslikescript.com'
    max_retries = 5

    def __init__(self, series_url, series_id, user_agent, acts=3, scenes=5):
        self.series_url = series_url
        self.series_id = series_id
        self.user_agent = user_agent
        self.acts = acts
        self.scenes = scenes
    
    def generate_corpus(self, season_limit=None):
        home_request = requests.get(
            self.series_url, 
            headers = {'User-agent': self.user_agent}
        )
        home_soup = BeautifulSoup(home_request.text, features = 'lxml')

        seasons = home_soup.find_all('div', {'class': 'season'})

        if season_limit:
            seasons = seasons[:season_limit]

        all_seasons = []
        for idx, season in enumerate(seasons):
            season_df = self.__scrape_season(
                season_text = season,
                season_id = idx + 1
            )
            all_seasons.append(season_df)
        
        series_df = pd.concat(all_seasons)
        series_df['series_id'] = self.series_id
        series_df = series_df.set_index(['series_id', 'season_id', 'episode_id', 'episode_title', 'scene_id', 'line_id'])
        
        self.corpus = series_df
        return self.corpus
    
    def generate_tokens(self):
        # generate tokens
        tokens_df = self.corpus.line.str.split(expand = True).stack().to_frame('token_str')
        tokens_df.index.names = ['series_id', 'season_id', 'episode_id', 'episode_title', 'scene_id', 'line_id', 'token_id']
        
        # clean up tokens
        tokens_df['term_str'] = tokens_df.token_str.str.replace(r'\W+', '', regex = True).str.lower()
        
        self.tokens = tokens_df
        return self.tokens
    
    def generate_vocab(self):
        vocab_df = self.tokens.term_str.value_counts().to_frame('n')
        vocab_df.index.name = 'term_str'
        
        self.vocab = vocab_df
        return self.vocab
        
    def __scrape_season(self, season_text, season_id):
        episodes = season_text.find_all('a', href = True, title = True)

        all_episodes = []
        for idx, ep in enumerate(episodes):
            episode_script = self.__scrape_episode(self.main_page + ep['href'])
            episode_lines = self.__split_episode_text_scenes(episode_script)
            episode_df = pd.DataFrame(episode_lines)
            episode_df['episode_id'] = idx + 1
            episode_df['episode_title'] = ep.text
            
            all_episodes.append(episode_df)
        
        season_df = pd.concat(all_episodes)
        season_df['season_id'] = season_id
        
        return season_df
    
    def __scrape_episode(self, episode_url):
        episode_script_text = ''
        retry = 0
        while retry < self.max_retries:
            try:
                episode_request = requests.get(
                    episode_url, 
                    headers = {'User-agent': self.user_agent}
                )
                episode_soup = BeautifulSoup(episode_request.text, features = 'lxml')
                episode_script = episode_soup.find('div', {'class': 'full-script'})
                
                for br in episode_script.find_all('br'):
                    br.replace_with('\n')
                
                episode_script_text = episode_script.text
                
                break
            except Exception as e:
                print('Error scraping {0}: {1}. Retry {2}'.format(episode_url, e, retry + 1))
                time.sleep((retry + 1) * 2)
                retry += 1
    
        return episode_script_text
    
    def __split_episode_text_scenes(self, episode_text):
        ep_lines = episode_text.split('\n')
        ep_lines = list(filter(None, ep_lines))
        
        # split by acts * scenes, index only by scene id
        num_acts = self.acts
        num_scenes = self.scenes
        split_factor = num_acts * num_scenes
        split_lines = list(self.__split(ep_lines, split_factor))
        
        all_splits = []
        curr_scene = 1
        for idx, scene in enumerate(split_lines):
            for idx, line in enumerate(scene):
                all_splits.append({
                    'scene_id': curr_scene,
                    'line_id': idx + 1,
                    'line': line
                })

            curr_scene += 1
        
        return all_splits
    
    def __split_episode_text_acts(self, episode_text):
        ep_lines = episode_text.split('\n')
        ep_lines = list(filter(None, ep_lines))
        
        # split by acts * scenes
        num_acts = self.acts
        num_scenes = self.scenes
        split_factor = num_acts * num_scenes
        split_lines = list(self.__split(ep_lines, split_factor))
        
        all_splits = []
        curr_act = 1
        curr_scene = 1
        for idx, act in enumerate(split_lines):
            if idx > (curr_act * num_scenes):
                curr_act += 1
                curr_scene = 1

            for idx, line in enumerate(act):
                all_splits.append({
                    'act_id': curr_act,
                    'scene_id': curr_scene,
                    'line_id': idx + 1,
                    'line': line
                })

            curr_scene += 1
        
        return all_splits

    def __split(self, a, n):
        k, m = divmod(len(a), n)

        return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
