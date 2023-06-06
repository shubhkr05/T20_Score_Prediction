import tensorflow as tf
from keras import backend as K

def custom_accuracy(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

save_format = 'tf'

for act_func in ['tanh', 'relu']:
    _model = tf.keras.models.load_model(f'model_with_function_{act_func}' + '.' + save_format, 
                                               custom_objects={'custom_accuracy': custom_accuracy}, 
                                               compile=True)

    print(_model.predict([10, 120]).flatten())
# print(new_model.metrics[1])