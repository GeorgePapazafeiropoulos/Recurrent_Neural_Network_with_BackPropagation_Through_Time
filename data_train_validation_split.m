%% Local function: data_train_validation_split
function [X_train, X_val, y_train, y_val] = data_train_validation_split(X, y, val_size)
% Split data into training and validation data
n_samples = size(X, 1);
val_samples = round(n_samples * val_size);
indices = randperm(n_samples);
val_indices = indices(1:val_samples);
train_indices = indices(val_samples+1:end);
X_train = X(train_indices, :);
X_val = X(val_indices, :);
y_train = y(train_indices, :);
y_val = y(val_indices, :);
end

