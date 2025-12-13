%% Local function: data_standardization
function [scaled_data, scaler] = data_standardization(data, range)
% Standardization of data
mean_val = mean(data);
std_val = std(data);
% Avoid division by zero
if std_val == 0
    std_val = 1;
end
scaled_data = (data - mean_val) / std_val;
% Store parameters
scaler.mean = mean_val;
scaler.std = std_val;
scaler.range = range; % Keep for compatibility
end

