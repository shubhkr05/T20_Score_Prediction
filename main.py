import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K
from sklearn.metrics import r2_score
from Read_Data import Processed_Data
from tensorflow.python.keras.metrics import MeanMetricWrapper

print(pd.__version__)

np.random.seed(1234)
tf.random.set_seed(1234)

files = ['ipl','sat', 'ntb', 'psl', 'bbl', 't20is']

li = []
for f in files:
    df_ = pd.read_csv(fr"all_matches\{f}.csv")
    li.append(df_)
df = pd.concat(li).reset_index().drop(['index'], axis=1)



df1 = Processed_Data(df)
# print(df)

total_matches = len(df1['match_id'].unique())
print(total_matches)


train_features = df1[df1['match_id'].isin(df1['match_id'].unique()[400:total_matches])][['wickets_left', 'balls_left','current_score', 'runs_scored_from']]
val_features = df1[df1['match_id'].isin(df1['match_id'].unique()[200:400])][['wickets_left', 'balls_left', 'current_score', 'runs_scored_from']]
test_features=df1[df1['match_id'].isin(df1['match_id'].unique()[:200])][[ 'wickets_left', 'balls_left','current_score', 'runs_scored_from']]

# train_y = df1[df1['match_id'].isin(df1['match_id'].unique()[:-200])][['Final_Score']]
# test_y = df1[df1['match_id'].isin(df1['match_id'].unique()[-200:])][['Final_Score']]

print("train_features", train_features.shape)
# print("train_y", train_y.shape)
print("val_features", val_features.shape)
print("test_features", test_features.shape)
# print("test_y", test_y.shape)


train_labels = train_features.pop('runs_scored_from')
val_labels = val_features.pop('runs_scored_from')
test_labels = test_features.pop('runs_scored_from')

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))
first = np.array(train_features[:1])

with np.printoptions(precision=2, suppress=True):
  print('First example:', first)
  print()
  print('Normalized:', normalizer(first).numpy())


def custom_accuracy(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )



def build_and_compile_model(norm, act_func):
  model = keras.Sequential([
      norm,
      layers.Dense(64, activation=act_func),
      layers.Dense(64, activation=act_func),
      layers.Dense(1)
  ])

  return model



for act_func in ['tanh', 'relu']:
    
    save_format = 'tf'
    
    print(f"Using - {act_func}")
    model_with_function = build_and_compile_model(normalizer, act_func)
    model_with_function.compile(loss='mean_absolute_error',
                optimizer=tf.keras.optimizers.Adam(0.001),
               metrics=[custom_accuracy])
    
    history = model_with_function.fit(
    train_features,
    train_labels,
    epochs=40,
    validation_data=(val_features, val_labels),
    verbose=1,
    batch_size=500)

    test_predictions = model_with_function.predict(test_features).flatten()
    print(r2_score(test_labels, test_predictions))

    model_with_function.save(f'./final_score_model_with_function_{act_func}' + '.' + save_format, 
                             save_format=save_format)
    
    try:
        new_model = tf.keras.models.load_model(f'final_score_model_with_function_{act_func}' + '.' + save_format, 
                                               custom_objects={'custom_accuracy': custom_accuracy}, 
                                               compile=True)
        print("model_with_function loaded with the following metrics:")
        print(new_model.metrics[0], new_model.metrics[1])
    except Exception as e:
        print("model_with_function not loaded with the following error:")
        print(e)