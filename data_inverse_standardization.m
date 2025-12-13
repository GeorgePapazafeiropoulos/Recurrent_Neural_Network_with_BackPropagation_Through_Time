%% Local function: data_inverse_standardization
function original_data = data_inverse_standardization(scaled_data, scaler)
% Inverse standardization of data with statistics from other data
original_data = scaled_data * scaler.std + scaler.mean;
end

