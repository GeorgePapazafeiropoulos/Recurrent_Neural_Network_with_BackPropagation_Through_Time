%% Local function: data_prepare_training
function [X_train, X_val, y_train, y_val, input_scaler, output_scaler] = data_prepare_training(...
    input_data, output_data, sequence_length, val_size)
% Reshape for normalization
X = input_data(:);
y = output_data(:);
% Standardize the input data
[X_scaled, input_scaler] = data_standardization(X, [0, 1]);
% Standardize the output data
[y_scaled, output_scaler] = data_standardization(y, [0, 1]);
% Create sequences of data
[X_seq, y_seq] = data_create_sequences(X_scaled, y_scaled, sequence_length);
% Split into train and test sets
[X_train, X_val, y_train, y_val] = data_train_validation_split(X_seq, y_seq, val_size);
end

