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
backend = 'tensorflow'  # 'cntk'
session_id_col = 'Request_ImpressionGuid'
time_col = 'TotalSeconds'
metadata_columns = ['TimeToFirstInteractionEvent', 'TimeToFirstKeyboardEvent', 
           'TimeToFirstPointerEvent', 'TimeToFirstScrollEvent']
folder_path = r'C:\Python 3 scripts\MSR Parkinson from mouse movements'
data_path = r'C:\Python 3 scripts\data sets\MSR mouse movements to detect parkinson'
metadata_filename = r'session_metadata.csv'
session_fields_to_use = [time_col, 'X', 'Y']
num_features = len(session_fields_to_use)
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

#%% Saving every session to its own csv
# The the sessions themselves aren't provided, we chose to leave the code we used to
# parse the session files for completeness sake.
# It is commented out to avoid cnfustion as of which parts of the code to run.

#def is_monthly_xl_file(file_name):
#    return len(file_name) >= 32 and file_name[:27] == 'pointer_and_keyboard_events' and file_name[-5:] =='.xlsx'
#
#print('Separating sessions into individual files.', dt.now())
#all_session_ids = []
#session_metadata = []
#last_session_count = 0
#session_metadata = []
#session_delta_to_print = int(1e4)
#monthly_xl_file_names = [file_ for file_ in os.listdir(data_path) if os.path.isfile(os.sep.join([data_path, file_])) and is_monthly_xl_file(file_)]
#for file_ in monthly_xl_file_names:
#    df = pd.ExcelFile(os.sep.join((data_path, file_))).parse('Sheet1')
#    print('read a file')
#    cur_sessions_ids = set(df[session_id_col])
#    all_session_ids.extend(list(cur_sessions_ids))
#    for session_id in cur_sessions_ids:
#        # Saving session metadata
#        # The metadata should be the same in all cells, but median is used to 
#        # avoid typos\errors.
#        cur_metadata_dict = dict(df[df[session_id_col] == session_id][[session_id_col] + metadata_columns].median())
#        cur_metadata_dict['session_id'] = session_id
#        session_metadata.append(cur_metadata_dict)
#        
#        # Saving the session mouse events
#        session_df = df[df[session_id_col] == session_id].sort_values(time_col)[session_fields_to_use]
#        session_df.to_csv(os.sep.join([data_path, ''.join(['session_', session_id, '.csv'])]))
#    if len(all_session_ids) >= last_session_count + session_delta_to_print:
#        last_session_count = len(all_session_ids)
#        print('separated %s sessions into individual csv files' % (last_session_count))
#print('Done separating sessions into individual files.', dt.now())
#
#if len(metadata_filename) >= 5:
#    if len(metadata_filename) >= 6 and metadata_filename[-5:] == '.json':
#        with open(os.sep.join([data_path, metadata_filename]), 'w') as file_:
#            json.dump(session_metadata, file_)
#    else:
#        if not metadata_filename[-4:] == '.csv':
#            metadata_filename += '.csv'
#        session_metadata_df = pd.DataFrame(session_metadata)
#        session_metadata_df.set_index('session_id', inplace=True)
#        session_metadata_df.to_csv(os.sep.join([data_path, metadata_filename]))
#else:
#    pass

#%% Cleaning and parsing sessions, and saving the data into a single pickle file
#def is_session_file(file_name):
#    return len(file_name) >= 12 and file_name[:8] == 'session_' and file_name[-4:] =='.csv'
#all_session_files, all_session_ids = zip(*[(file_, file_[8:-4]) for file_ in os.listdir(data_path) if os.path.isfile(os.sep.join([data_path, file_])) and is_session_file(file_)])
#test = True
#for ind in range(len(all_session_ids)):
#    if not all_session_files[ind][8:-4] == all_session_ids[ind]:
#        test = False
#        break
#assert(test)
#
#max_session_len = 0
#session_arrays = []
#all_time_diffs = np.array([])
#all_x = np.array([])
#all_y = np.array([])
#all_session_ids = sorted(all_session_ids)
#all_session_files = sorted(all_session_files)
#
#valid_session_ids = []
#for session_serial, cur_session_file in enumerate(all_session_files):
#    try:
#        cur_session_array = pd.read_csv(os.sep.join([data_path, cur_session_file]), encoding='utf-8')[session_fields_to_use].values
#        cur_session_array = np.diff(cur_session_array, axis=0)
#        cur_session_array[:,0] = np.log10(cur_session_array[:,0])
#        session_len = cur_session_array.shape[0]
#        if session_len >= min_session_events and not np.any(~np.isfinite(cur_session_array)):
#            max_session_len = max(session_len, max_session_len)
#            session_arrays.append(cur_session_array)
#            valid_session_ids.append(all_session_ids[session_serial])
#    except Exception as e_:
#        print(session_serial, cur_session_file, e_)
#
#with open(data_path+os.sep+'-session_arrays.p', 'wb') as file_:
#    pickle.dump((session_arrays, valid_session_ids), file_)

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

#%% Saving features to CSV
# Once again commented out but kept for completeness sake.

#save_end_of_session_batchs = True
#save_full_session = True
#max_full_sessions_to_save = int(1e4)
#
#def get_feature_extractor(model):
#    return Model(inputs=model.input, outputs=model.get_layer(index=-2).output)
#
#try:
#    feature_extractor = get_feature_extractor(model)
#except NameError:
#    feature_extractor = load_model(folder_path+os.sep+cur_alg_description+'.h5')
#    feature_extractor = get_feature_extractor(model)
#
#def save_batch_to_csv(feature_extractor, x, x_meta, session_id_list, session_unpadded_lens, start_ind, end_ind, file_serial, cur_alg_description):
#    feature_tensor = feature_extractor.predict([x[start_ind:end_ind], x_meta])
#    # session_id_list-2 is used because its -1 for the last index being the length-1, and another -1 because the last sample is only used in y.
#    session_features = [feature_tensor[session_serial, session_len-2, :] for session_serial, session_len in enumerate(session_unpadded_lens[start_ind:end_ind])]
#    session_features = np.array(session_features)
#    feature_df = pd.DataFrame(session_features, index=session_id_list[start_ind:end_ind])
#    feature_df.to_csv(folder_path+os.sep+'-session_features_%s_%s.csv' % (cur_alg_description, file_serial), index=True)
#
#def save_sessions_to_csvs(model, x, x_meta, session_id_list, session_unpadded_lens, start_ind, end_ind, file_serial, cur_alg_description):
#    feature_tensor = model.predict([x[start_ind:end_ind], x_meta])
#    # session_id_list-2 is used because its -1 for the last index being the length-1, and another -1 because the last sample is only used in y.
#    for session_serial, (session_len, session_id) in enumerate(zip(session_unpadded_lens[start_ind:end_ind], session_id_list[start_ind:end_ind])):
#        session_features = feature_tensor[session_serial, :, :]
#        feature_df = pd.DataFrame(session_features)
#        feature_df.to_csv(folder_path+os.sep+'-full_session_%s_%s.csv' % (cur_alg_description, session_id), index=True)
#
#print('Starting to save sessions', dt.now())
#sessions_per_batch = int(1e4)
#start_ind = 0
#end_ind = sessions_per_batch
#file_serial = 0
#while end_ind <= x.shape[0]:
#    if save_full_session and end_ind <= max_full_sessions_to_save:
#        save_sessions_to_csvs(model, x, x_meta, session_id_list, session_unpadded_lens, start_ind, end_ind, file_serial, cur_alg_description)
#    if save_end_of_session_batchs:
#        save_batch_to_csv(feature_extractor, x, x_meta, session_id_list, session_unpadded_lens, start_ind, end_ind, file_serial, cur_alg_description)
#    start_ind = copy(end_ind)
#    end_ind += sessions_per_batch
#    file_serial += 1
#end_ind = x.shape[0]
#if save_full_session and end_ind <= max_full_sessions_to_save:
#    save_sessions_to_csvs(feature_extractor, x, x_meta, session_id_list, session_unpadded_lens, start_ind, end_ind, file_serial, cur_alg_description)
#if save_end_of_session_batchs:
#    save_batch_to_csv(feature_extractor, x, x_meta, session_id_list, session_unpadded_lens, start_ind, end_ind, file_serial, cur_alg_description)
