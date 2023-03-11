import urllib.request
import tarfile
from pathlib import Path
from itertools import product
import pandas as pd
import numpy as np
from settings import FEATURES

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
    df = pd.read_csv('data/NBA_PBP_2017-18.csv')
    df['Season'] = 2017
    df2 = pd.read_csv('data/NBA_PBP_2018-19.csv')
    df2['Season'] = 2018
    df = pd.concat([df, df2], axis=0)
    df['GameId'] = df.groupby(['HomeTeam', 'AwayTeam', 'Date']).ngroup()
    # Time Elapsed
    df['TimeElapsed'] = np.where(df['Quarter']<=4, 720, 300) - df['SecLeft'] + (df['Quarter'].clip(1,4)-1)*720 + (df['Quarter']-5).clip(0)*300
    df['TimeRemaining'] = df['SecLeft'] + (4 - df['Quarter'].clip(1,4))*720
    # Home Indicator
    df['Home'] = np.where(df['HomePlay'].isna(), 0, 1)
    # days rest by team for opponent
    df['Date'] = pd.to_datetime(df['Date'])
    df['Team'] = np.where(df['HomePlay'].isna(), df['AwayTeam'], df['HomeTeam'])
    team_rest = df.groupby(['GameId', 'Team']).first().sort_values('Date').groupby('Team')['Date'].diff().dt.days.fillna(0).clip(0,4).reset_index()
    df = df.merge(team_rest.rename(columns={'Date': 'TeamRest'}), on=['GameId', 'Team'])
    df['Opponent'] = np.where(df['HomePlay'].isna(), df['AwayTeam'], df['HomeTeam'])
    df = df.merge(team_rest.rename(columns={'Date': 'OpponentTeamRest'}), left_on=['GameId', 'Opponent'], right_on=['GameId', 'Team'], suffixes=('', '_y'))
    # remove playoffs
    df = df.loc[df['GameType'] == "regular"]
    return df

def player_game_rolling_avg(df: pd.DataFrame, statistic: str):
    """
    Calculates the rolling EWM average for each player in each game.
    Must have columns:
    Player, GameId, Cumulative{statistic}
    """
    df = df.sort_values('Date')
    game_stats = df.groupby(['Player', 'GameId', 'Date'])[f'CumulativePlayer{statistic}'].max().reset_index()
    game_stats[f'RollingAvgPlayer{statistic}'] = game_stats.sort_values('Date').groupby(['Player'])[f'CumulativePlayer{statistic}'].transform(lambda x: x.ewm(span=5).mean())
    game_stats["PlayerGameNumber"] = game_stats.groupby(['Player']).cumcount()
    df = df.merge(game_stats[['Player', 'GameId', f'RollingAvgPlayer{statistic}', 'PlayerGameNumber']], on=['Player', 'GameId'], how='left')
    return df

def team_game_rolling_avg(df: pd.DataFrame, statistic: str):
    """
    Calculates the rolling EWM average for each team in each game.
    Must have columns:
    Team, GameId, CumulativeTeam{statistic}
    """
    df = df.sort_values('Date')
    home_stats = df.groupby(['HomeTeam', 'GameId', 'Date'])[[f'CumulativeTeam{statistic}', f'CumulativeOpponent{statistic}']].max().reset_index()
    home_stats[f'RollingAvgHomeTeam{statistic}'] = home_stats.sort_values('Date').groupby(['HomeTeam'])[[f'CumulativeTeam{statistic}']].transform(lambda x: x.ewm(span=5).mean())
    home_stats[f'RollingAvgHomeTeamAllowed{statistic}'] = home_stats.sort_values('Date').groupby(['HomeTeam'])[[f'CumulativeOpponent{statistic}']].transform(lambda x: x.ewm(span=5).mean())

    away_stats = df.groupby(['AwayTeam', 'GameId', 'Date'])[[f'CumulativeTeam{statistic}', f'CumulativeOpponent{statistic}']].max().reset_index()
    away_stats[f'RollingAvgAwayTeam{statistic}'] = away_stats.sort_values('Date').groupby(['AwayTeam'])[[f'CumulativeTeam{statistic}']].transform(lambda x: x.ewm(span=5).mean())
    away_stats[f'RollingAvgAwayTeamAllowed{statistic}'] = away_stats.sort_values('Date').groupby(['AwayTeam'])[[f'CumulativeOpponent{statistic}']].transform(lambda x: x.ewm(span=5).mean())

    df = df.merge(home_stats[['HomeTeam', 'GameId', f'RollingAvgHomeTeam{statistic}', f'RollingAvgHomeTeamAllowed{statistic}']], left_on=['HomeTeam', 'GameId'], right_on=['HomeTeam', 'GameId'], how='left')
    df = df.merge(away_stats[['AwayTeam', 'GameId', f'RollingAvgAwayTeam{statistic}', f'RollingAvgAwayTeamAllowed{statistic}']], left_on=['AwayTeam', 'GameId'], right_on=['AwayTeam', 'GameId'], how='left')
    df[f'RollingAverageTeam{statistic}'] = np.where(df['Home'] == 1, df[f'RollingAvgHomeTeam{statistic}'], df[f'RollingAvgAwayTeam{statistic}'])
    df[f'RollingAverageOpposingTeamAllowed{statistic}'] = np.where(df['Home'] == 0, df[f'RollingAvgHomeTeamAllowed{statistic}'], df[f'RollingAvgAwayTeamAllowed{statistic}'])
    return df


def player_game_scoring(df: pd.DataFrame, intervals = 2):
    """
    Calculates the cumulative points for each player in each game
    """
    scoring_play = ~((df['FreeThrowOutcome'].isna()) & (df['ShotOutcome'].isna()))
    df = df.loc[scoring_play]
    df['Player'] = np.where(~df['FreeThrowOutcome'].isna(), df['FreeThrowShooter'], df['Shooter'])
    df['PotentialPoints'] = np.where(df['ShotType'].str.startswith('3'), 3, 2)
    df['PotentialPoints'] = np.where(df['FreeThrowOutcome'].isna(), df['PotentialPoints'], 1)
    df['Points'] = ((df['FreeThrowOutcome'] =="make") | (df['ShotOutcome'] =="make")) * df['PotentialPoints']
    df['CumulativePlayerPoints'] = df.sort_values('TimeElapsed').groupby(['Player', 'GameId'])['Points'].cumsum()
    df['CumulativeTotalPoints'] = df.sort_values('TimeElapsed').groupby(['GameId'])['Points'].cumsum()
    df['CumulativeTeamPoints'] = df.sort_values('TimeElapsed').groupby(['Team', 'GameId'])['Points'].cumsum()
    df['CumulativeOpponentPoints'] =  df['CumulativeTotalPoints'] - df['CumulativeTeamPoints']
    # # in terms of team on offense. Positive means offense winning
    # df['ScoreMargin'] = df['CumulativeTeamPoints'] - df['CumulativeOpponentPoints']
    # # in terms of home team. Positive means home winning
    # df['HomeScoreMargin'] = np.where(df['HomePlay'].isna(), -df['ScoreMargin'], df['ScoreMargin'])
    # df['HomeScoreMarginxTimeRemaining'] = df['HomeScoreMargin'] * df['TimeRemaining']
    # df['HomeScoreMarginxTimeRemaining2'] = df['HomeScoreMargin'] * (df['TimeRemaining']**2)
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
        cum_pts = df.groupby(['Player', 'GameId']).apply(lambda x: x.loc[x['TimeElapsed'] <= max_time, f'CumulativePlayer{statistic}'].max()).reset_index().fillna(0)
        tot_pts = df.groupby(['GameId']).apply(lambda x: x.loc[x['TimeElapsed'] <= max_time, f'CumulativeTotal{statistic}'].max()).reset_index().fillna(0)
        team_pts = df.groupby(['Team', 'GameId']).apply(lambda x: x.loc[x['TimeElapsed'] <= max_time, f'CumulativeTeam{statistic}'].max()).reset_index().fillna(0)
        opponent_pts = df.groupby(['Team', 'GameId']).apply(lambda x: x.loc[x['TimeElapsed'] <= max_time, f'CumulativeOpponent{statistic}'].max()).reset_index().fillna(0)
        df = df.merge(cum_pts.rename(columns={0: f'CumulativePlayer{statistic}_{i}'}), on=['Player', 'GameId'], how='left')
        df = df.merge(tot_pts.rename(columns={0: f'CumulativeTotal{statistic}_{i}'}), on=['GameId'], how='left')
        df = df.merge(team_pts.rename(columns={0: f'CumulativeTeam{statistic}_{i}'}), on=['Team', 'GameId'], how='left')
        df = df.merge(opponent_pts.rename(columns={0: f'CumulativeOpponent{statistic}_{i}'}), on=['Team', 'GameId'], how='left')
    # df = df.groupby(['Player', 'GameId']).first()
    return df

def stack_intervals(df: pd.DataFrame, intervals: int, statistic: str):
    df_list = []
    for interval in range(intervals):
        interval_cols = [c for c in df.columns if f"{statistic}_" in c]
        # only include columns from this interval and the other feature columns
        interval_cols_other = [c for c in interval_cols if f"{statistic}_{interval}" not in c]
        this_interval_cols = [c for c in interval_cols if f"{statistic}_{interval}" in c]
        temp_df = df[[c for c in df.columns if c not in interval_cols_other]]
        temp_df = temp_df.rename(columns={c: c.replace(f'{statistic}_{interval}', f'{statistic}Interval') for c in this_interval_cols})
        temp_df["Interval"] = interval + 1
        time_remaining = 48*60 - ((48/intervals) * (interval+1) * 60)
        time_remaining = 0 if (interval == (intervals-1)) else time_remaining
        temp_df['ScoreMarginInterval'] = temp_df['CumulativeTeamPointsInterval'] - temp_df['CumulativeOpponentPointsInterval']
        temp_df['ScoreMarginxTimeRemainingInterval'] = abs(temp_df['ScoreMarginInterval']) * time_remaining
        temp_df['ScoreMarginxTimeRemaining2Interval'] = abs(temp_df['ScoreMarginInterval']) * time_remaining**2
        df_list.append(temp_df)
    # create synthetic for start of game
    start_df = temp_df.copy()
    start_df[this_interval_cols] = 0
    start_df["Interval"] = 0
    start_df['ScoreMargin'] = 0
    start_df['ScoreMarginxTimeRemaining'] = 0
    start_df['ScoreMarginxTimeRemaining2'] = 0
    start_df[[c for c in start_df.columns if 'Cumulative' in c]] = 0
    df = pd.concat(df_list + [start_df], axis=0)
    return df

if __name__ == "__main__":
    intervals = 4
    statistic = "Points"
    df = data_proc()
    df = player_game_scoring(df, intervals=intervals)
    df = player_game_rolling_avg(df, 'Points')
    df = team_game_rolling_avg(df, 'Points')
    df = df.groupby(['Player', 'GameId']).first().reset_index()
    df = stack_intervals(df, intervals, statistic)

    # only include players that we have at least 10 games on
    df = df.loc[(df["PlayerGameNumber"] >= 10) & (df['Season'] == 2018), FEATURES]
    df.to_csv('data/NBA_PBP_2018-19_processed_points.csv', index=False)
