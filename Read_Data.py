import pandas as pd
import numpy as np


# files = ['ipl_male_csv2']



def Processed_Data(df):
    df['over'] = df['ball'].astype(float).astype(int)+1
    df['over'] = pd.to_numeric(df['over'], errors="coerce").fillna(0).astype('int64')
    df['runs_off_bat'] = pd.to_numeric(df['runs_off_bat'], errors="coerce").fillna(0).astype('int64')
    df['extras'] = pd.to_numeric(df['extras'], errors="coerce").fillna(0).astype('int64')
    df['total_runs'] = df['extras'] + df['runs_off_bat'] 
    df['total_runs'] = pd.to_numeric(df['total_runs'], errors="coerce").fillna(0).astype('int64')
    df['legal_ball'] = np.where((df['wides'].isna() & df['noballs'].isna()), 1, 0)
    df['legal_ball'] = pd.to_numeric(df['legal_ball'], errors="coerce").fillna(0).astype('int64')

    df['current_score'] = df.groupby(['match_id', 'innings'])['total_runs'].apply(lambda x: x.cumsum()).reset_index()['total_runs']
    df['current_score'] = pd.to_numeric(df['current_score'], errors="coerce").fillna(0).astype('int64')

    df['wicket_no'] = np.where(df['player_dismissed'].isin([np.nan]), 0, 1)
    df['wicket_no'] = df.groupby(['match_id', 'innings'])['wicket_no'].apply(lambda x: x.cumsum()).reset_index()['wicket_no']
    df['wicket_no'] = pd.to_numeric(df['wicket_no'], errors="coerce").fillna(0).astype('int64')
    df['wides'] = pd.to_numeric(df['wides'], errors="coerce").fillna(0).astype('int64')
    df['noballs'] = pd.to_numeric(df['noballs'], errors="coerce").fillna(0).astype('int64')
    df['byes'] = pd.to_numeric(df['byes'], errors="coerce").fillna(0).astype('int64')
    df['legbyes'] = pd.to_numeric(df['legbyes'], errors="coerce").fillna(0).astype('int64')
    df['penalty'] = pd.to_numeric(df['penalty'], errors="coerce").fillna(0).astype('int64')
    df['innings'] = pd.to_numeric(df['innings'], errors="coerce").fillna(0).astype('int64')
    df['ball'] = df['ball'].astype(str)
    df['balls_left'] = 120 - df['ball'].apply(lambda x:int(x.split('.')[0])*6 + int(x.split('.')[-1]))
    df['wickets_left'] = 10 - df['wicket_no']

    df['Final_Score'] = df.groupby(['match_id', 'innings'])['total_runs'].transform(lambda x:x.sum())
    df['balls_left'] = df['balls_left'].apply(lambda x:0 if x<0 else x)

    df['run_rate'] = df.groupby(['match_id', 'innings']).apply(lambda x: x['current_score']/(120-x['balls_left'])).reset_index()[0]

    df = df[df['innings'].isin([1, 2])]
    df = df[['season','match_id','innings','Final_Score','striker', 'current_score', 'balls_left', 'wickets_left', 'over', 'run_rate', 'runs_off_bat', 'total_runs']]
    df.loc[:, 'run_rate(t-1)'] = df['run_rate'].shift(1)
    df = df.iloc[1:]
    df['runs_scored_from'] = df['Final_Score'] - df['current_score']

    return df