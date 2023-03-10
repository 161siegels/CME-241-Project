import urllib.request
import tarfile
from pathlib import Path
from itertools import product
import pandas as pd
import numpy as np

def data_proc():
    """
    Loads the data and processes it into a dataframe
    Context columns:
    GameId: unique identifier for each game
    TimeElapsed: time elapsed in seconds
    Home: home indicator
    Team: team of player
    Opponent: opponent of team
    TeamRest: days rest for team
    OpponentTeamRest: days rest for opponent
    """
    df = pd.read_csv('data/NBA_PBP_2018-19.csv')
    df['GameId'] = df.groupby(['HomeTeam', 'AwayTeam', 'Date']).ngroup()
    # Time Elapsed
    df['TimeElapsed'] = np.where(df['Quarter']<=4, 720, 300) - df['SecLeft'] + (df['Quarter'].clip(1,4)-1)*720
    # Home Indicator
    df['Home'] = np.where(df['HomePlay'].isna(), 0, 1)
    # days rest by team for opponent
    df['Date'] = pd.to_datetime(df['Date'])
    df['Team'] = np.where(df['HomePlay'].isna(), df['HomeTeam'], df['AwayTeam'])
    team_rest = df.groupby(['GameId', 'Team']).first().sort_values('Date').groupby('Team')['Date'].diff().dt.days.fillna(0).clip(0,4).reset_index()
    df = df.merge(team_rest.rename(columns={'Date': 'TeamRest'}), on=['GameId', 'Team'])
    df['Opponent'] = np.where(df['HomePlay'].isna(), df['AwayTeam'], df['HomeTeam'])
    df = df.merge(team_rest.rename(columns={'Date': 'OpponentTeamRest'}), left_on=['GameId', 'Opponent'], right_on=['GameId', 'Team'], suffixes=('', '_y'))
    
    return df

def player_game_scoring(df: pd.DataFrame, intervals = 12):
    """
    Calculates the cumulative points for each player in each game
    """
    scoring_play = ~((df['FreeThrowOutcome'].isna()) | (df['FreeThrowOutcome'].isna()))
    df = df.loc[scoring_play]
    df['Player'] = np.where(~df['FreeThrowOutcome'].isna(), df['FreeThrowShooter'], df['Shooter'])
    df['PotentialPoints'] = np.where(df['ShotType'].str.startswith('3'), 3, 2)
    df['Points'] = ((df['FreeThrowOutcome'] =="make") | (df['ShotOutcome'] =="make")) * df['PotentialPoints']
    df['CumulativePoints'] = df.sort_values('TimeElapsed').groupby(['Player', 'GameId'])['Points'].cumsum()
    df = interval_metric(df, intervals, 'Points')
    return df

def interval_metric(df: pd.DataFrame, intervals: int, statistic: str):
    """
    Calculates the metric per interval time in game for each player
    Results in dataframe with columns:
    Player, GameId, Cumulative{statistic}_{i}, where i is the interval
    Extra columns for context as well
    """
        # Check Points every 4 minutes
    for i in range(intervals):
        # right now it is getting rid of OT points. We don't want this so will add in edge case
        max_time = (48/intervals) * (i+1) * 60
        if (i == intervals-1):
            max_time = np.float('inf')
        cum_pts = df.groupby(['Player', 'GameId']).apply(lambda x: x.loc[x['TimeElapsed'] <= max_time, f'Cumulative{statistic}'].max()).reset_index().fillna(0)
        df = df.merge(cum_pts.rename(columns={0: f'Cumulative{statistic}_{i}'}), on=['Player', 'GameId'])
    return df

if __name__ == "__main__":
    df = data_proc()
    df = player_game_scoring(df)
    df.to_csv('data/NBA_PBP_2018-19_processed_points.csv', index=False)
