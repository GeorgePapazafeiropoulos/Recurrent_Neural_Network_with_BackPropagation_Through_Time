%% Local function: data_prepare_testing
function [X_test, y_test] = data_prepare_testing(...
    input_data, output_data, sequence_length, input_scaler, output_scaler)
% Reshape for normalization
X = input_data(:);
y = output_data(:);
% Apply the same normalization as training data
%X_scaled = data_apply_standardization(X, input_scaler);
X_scaled = (X - input_scaler.mean) / input_scaler.std;
%y_scaled = data_apply_standardization(y, output_scaler);
y_scaled = (y - output_scaler.mean) / output_scaler.std;
% Create sequences
[X_test, y_test] = data_create_sequences(X_scaled, y_scaled, sequence_length);
end

