%% Local function: rnn_evaluate
function [mse, mae, predictions, y_true] = rnn_evaluate(X_test, y_test,...
    W1, W2, W3, b1, b2, output_scaler)
% Function for making predictions with RNN
[~, test_outputs, ~] = rnn_forward(X_test, W1, W2, W3, b1, b2);
% Use last output for prediction
predictions_scaled = test_outputs(:, end);
y_test_scaled = y_test;
% Inverse transform
predictions = data_inverse_standardization(predictions_scaled, output_scaler);
y_true = data_inverse_standardization(y_test_scaled, output_scaler);
% Calculate metrics
mse = mean((predictions - y_true).^2);
mae = mean(abs(predictions - y_true));
end

