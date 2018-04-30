import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from datetime import datetime as dt
import pickle
from copy import copy, deepcopy
import gc
import json

from keras.optimizers import Adam
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense, Masking, TimeDistributed, RepeatVector, Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

gpu_description_substring = 'GeForce'
backend = 'tensorflow'
session_id_col = 'Request_ImpressionGuid'
time_col = 'TotalSeconds'
folder_path = r'C:\MSR Parkinson from mouse movements'
data_path = r'C:\data sets\MSR mouse movements to detect parkinson'
metadata_filename = r'session_metadata.csv'
num_features = len(3)
keep_longer_sessions = False #  Determines if session with less than max_session_events will be used.

opt_clip_val = 0.5
validation_quantile = 0.2
mask_val = 0  # None to infer
pad_beginning = True #  If False then the padding will be at the end of the session.
validation_quantile = 0.2

if backend == 'tensorflow':
    from tensorflow.python.client import device_lib
    if not any([gpu_description_substring in device.physical_device_desc for device in device_lib.list_local_devices()]):
        raise Exception("Couldn't find the GPU")
else:
    import imp
    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        imp.reload(K)
        assert K.backend() == backend

#%% Preprocessing
# Hyper-parameters are defined here for convenience.
# The parameters below are the ones we tuned while the ones
# defined above are fixed.

min_session_events = 5
max_session_events = 25
keep_longer_sessions = False #  Determines if session with less than max_session_events will be used.
num_epochs = 7
lstm_dim = 1024 #  [512, 1024]
batch_size = 256
opt_lr = np.logspace(-2.75,-2.75,1)[0]
use_metadata = False
meta_data_columns_to_use = ['TimeToFirstInteractionEvent'] #  metadata_columns
min_events_for_predictions = min_session_events - 1
padded_session_shape = (max_session_events - 1, num_features)
cur_alg_description = '25_samples' #  '25_samples_cutting_longer_sessions_with_first_interaction_metadata' #  

def scaling_unit_test(scaled_united_session):
    # median condition and IQR condition.
    conditions = [np.all(np.median(scaled_united_session, axis=0) == 0), np.all(np.percentile(scaled_united_session, 75, axis=0) - np.percentile(scaled_united_session, 25, axis=0) == 1)]
    return all(conditions)

def get_scaler_and_mask_val(session_arrays, max_session_events):
    united_session = np.concatenate([session[:max_session_events,:] for session in session_arrays])
    scaler = RobustScaler().fit(np.concatenate([session[:max_session_events,:] for session in session_arrays]))
    scaled_united_session = scaler.transform(united_session)
    assert scaling_unit_test(scaled_united_session)
    mask_val = np.ceil(np.max(np.max(scaled_united_session, axis = 0)*3))
    return scaler, mask_val

with open(data_path+os.sep+'-session_arrays.p', 'rb') as file_:
    session_arrays, valid_session_ids = pickle.load(file_)
session_metadata_df = pd.read_csv(os.sep.join([data_path, metadata_filename]), index_col='session_id')
session_metadata_df = session_metadata_df[meta_data_columns_to_use]

scaler, suggeted_mask_val = get_scaler_and_mask_val(session_arrays, max_session_events)
if not type(mask_val) == int: mask_val = suggeted_mask_val

scaled_session_arrays = []
min_session_events = max(min_session_events,2)
for session_ind, (session_id, session_array) in enumerate(zip(valid_session_ids, session_arrays)):
    #Checking that the session has at least min_session_events events and that there aren't any missing values.
    if session_array.shape[0] < min_session_events or np.any(~np.isfinite(session_array)):
        print('Error, an invalid session encountered in the list of valid sessions. Session serial %s, Session ID: %s' % (session_ind, session_id))
    else:
        cur_scaled_array = scaler.transform(session_array[:max_session_events,:])
        scaled_session_arrays.append(cur_scaled_array)
assert len(valid_session_ids) == len(scaled_session_arrays)

def pad_or_truncate(s, session_len, mask_value=0, pad_beginning=True):
    if s.shape[0] >= session_len:
        s_padded = s[:session_len,:] #  Always truncating at the end.
    elif pad_beginning:
        s_padded = np.concatenate((mask_value*np.ones((session_len - s.shape[0], s.shape[1])), s), axis=0)
    else:
        s_padded = np.concatenate((s, mask_value*np.ones((session_len - s.shape[0], s.shape[1]))), axis=0)
    return [s_padded[:-1,:], s_padded[1:,:]]

if keep_longer_sessions:
    x, y, session_id_list, session_unpadded_lens = zip(*[pad_or_truncate(s, max_session_events, mask_val, pad_beginning)+[id_, len(s)] for s, id_ in zip(scaled_session_arrays, valid_session_ids) if s.shape[0] >= min_events_for_predictions and np.all(np.isfinite(s))])
else:
    x, y, session_id_list, session_unpadded_lens = zip(*[pad_or_truncate(s, max_session_events, mask_val, pad_beginning)+[id_, len(s)] for s, id_ in zip(scaled_session_arrays, valid_session_ids) if min_events_for_predictions <= s.shape[0] <= max_session_events and np.all(np.isfinite(s))])
x = np.array(list(x))
y = np.array(list(y))

x_meta = []
for session_id in session_id_list:
    x_meta.append(session_metadata_df.loc[session_id])
x_meta = np.array(x_meta)

#%% Training the deep learning model
gc.collect()
training_documentations = []
            
training_start_time = dt.now()
unmasked_events = Input(shape = padded_session_shape)
events = Masking(mask_value=mask_val, input_shape = padded_session_shape)(unmasked_events)
metadata = Input(shape = tuple([len(meta_data_columns_to_use)]))
if use_metadata:
    repeated_metadata = RepeatVector(padded_session_shape[0])(metadata)
    events = Concatenate()([events, repeated_metadata])
encoding = LSTM(lstm_dim, return_sequences=True)(events)
output = TimeDistributed(Dense(num_features))(encoding)
model = Model([unmasked_events, metadata], output)
optimizer = Adam(opt_lr, clipvalue=opt_clip_val)
model.compile(loss='mean_squared_error', optimizer=optimizer)

model.summary()
model.fit([x, x_meta], y, epochs=num_epochs, batch_size=batch_size, shuffle=True, validation_split=validation_quantile)
training_end_time = dt.now()
training_duration = (training_end_time - training_start_time).total_seconds()

mask_vect = np.ones((num_features)) * mask_val
yh = model.predict([x, x_meta])
est_mse = np.mean(np.array([np.sum(np.array([np.mean((y[s_ind,t,:]-yh[s_ind,t,:])**2) for t in range(y.shape[1]) if not np.all(y[s_ind,t,:] == mask_vect)])) for s_ind in range(y.shape[0])]))
norm = np.mean(np.array([np.sum(np.array([np.mean(y[s_ind,t,:]**2) for t in range(y.shape[1]) if not np.all(y[s_ind,t,:] == mask_vect)])) for s_ind in range(y.shape[0])]))
explained_var = 1-est_mse/norm
print('Learning rate: %s' % (opt_lr))
print('Explained variance: %s%s.' % (round((explained_var)*100,2),'%'))

explained_vars = [doc['explained_var'] if 'explained_var' in doc.keys() else 0 for doc in training_documentations if 'explained_var' in doc.keys()]
val_losses = [min(doc['model_performance']['val_loss']) for doc in training_documentations if 'explained_var' in doc.keys()]
model.save(folder_path+os.sep+cur_alg_description+'.h5')
