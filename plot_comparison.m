%% Local function: plot_comparison
function plot_comparison(train_loss, val_loss, ...
    y_true_val, y_pred_val, y_true_test, y_pred_test, m, k, c)
sample_size = min(500, length(y_true_val));
figure('Position', [50, 50, 1400, 900]);
% Plot training history
subplot(3, 3, 1);
semilogy(train_loss, 'b-', 'LineWidth', 2);
hold on;
semilogy(val_loss, 'r-', 'LineWidth', 2);
legend('Training Loss', 'Validation Loss', 'Location', 'best');
title('Training History');
xlabel('Epoch');
ylabel('MSE Loss (log scale)');
grid on;
% Plot validation data predictions
subplot(3, 3, 2);
plot(y_true_val(1:sample_size), 'b-', 'LineWidth', 2, 'DisplayName', 'True');
hold on;
plot(y_pred_val(1:sample_size), 'r--', 'LineWidth', 1.5, 'DisplayName', 'Predicted');
title('Validation Data');
xlabel('Time Step');
ylabel('Displacement');
legend('show', 'Location', 'best');
grid on;
% Plot testing data predictions
subplot(3, 3, 3);
new_sample_size = min(sample_size, length(y_true_test));
plot(y_true_test(1:new_sample_size), 'b-', 'LineWidth', 2, 'DisplayName', 'True');
hold on;
plot(y_pred_test(1:new_sample_size), 'r--', 'LineWidth', 1.5, 'DisplayName', 'Predicted');
title('Testing Data (Sinusoids)');
xlabel('Time Step');
ylabel('Displacement');
legend('show', 'Location', 'best');
grid on;
% Plot scatter for validation data
subplot(3, 3, 4);
scatter(y_true_val, y_pred_val, 10, 'b', 'filled', 'MarkerFaceAlpha', 0.3);
hold on;
plot([min(y_true_val), max(y_true_val)], [min(y_true_val), max(y_true_val)], ...
    'r-', 'LineWidth', 2, 'DisplayName', 'Perfect prediction');
xlabel('True Displacement');
ylabel('Predicted Displacement');
title('Validation: Prediction vs Truth');
grid on;
axis equal;
% Plot scatter for testing data
subplot(3, 3, 5);
scatter(y_true_test, y_pred_test, 10, 'r', 'filled', 'MarkerFaceAlpha', 0.3);
hold on;
plot([min(y_true_test), max(y_true_test)], [min(y_true_test), max(y_true_test)], ...
    'k-', 'LineWidth', 2, 'DisplayName', 'Perfect prediction');
xlabel('True Displacement');
ylabel('Predicted Displacement');
title('Testing: Prediction vs Truth');
grid on;
axis equal;
% Plot error comparison
subplot(3, 3, 6);
errors_val = abs(y_pred_val - y_true_val);
errors_test = abs(y_pred_test - y_true_test);
plot(errors_val(1:sample_size), 'b-', 'LineWidth', 1, 'DisplayName', 'Validation Error');
hold on;
plot(errors_test(1:min(sample_size, length(errors_test))), 'r-', 'LineWidth', 1, 'DisplayName', 'Testing Error');
xlabel('Time Step');
ylabel('Absolute Error');
title('Prediction Errors');
legend('show');
grid on;
% Plot error distributions
subplot(3, 3, 7);
histogram(errors_val, 30, 'FaceColor', 'blue', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
hold on;
histogram(errors_test, 30, 'FaceColor', 'red', 'FaceAlpha', 0.5, 'EdgeColor', 'none');
xlabel('Prediction Error');
ylabel('Frequency');
title('Error Distributions');
legend('Validation', 'Testing');
grid on;
% Add comprehensive statistics
mse_val = mean((y_pred_val - y_true_val).^2);
mae_val = mean(abs(y_pred_val - y_true_val));
mse_test = mean((y_pred_test - y_true_test).^2);
mae_test = mean(abs(y_pred_test - y_true_test));
% Calculate R-squared
SS_res_val = sum((y_pred_val - y_true_val).^2);
SS_tot_val = sum((y_true_val - mean(y_true_val)).^2);
R2_val = 1 - SS_res_val/SS_tot_val;
SS_res_test = sum((y_pred_test - y_true_test).^2);
SS_tot_test = sum((y_true_test - mean(y_true_test)).^2);
R2_test = 1 - SS_res_test/SS_tot_test;
% Natural frequency info
f_n = sqrt(k/m) / (2*pi);
zeta = c / (2*sqrt(m*k));
% Create the text showing the results of the RNN training and testing
stats_text1 = sprintf(['SDOF System:            \n f_n=%.2f Hz\n zeta=%.3f',...
    '\n\n\n\n\n\n\n\n\n\n\n'], ...
    f_n, zeta);
stats_text2 = sprintf(['TRAINING (White Noise):\n' ...
    'MSE:  %.3e\nMAE:  %.3e\nR²:   %.3f\n\n' ...
    'TESTING (Sinusoids):\n' ...
    'MSE:  %.3e\nMAE:  %.3e\nR²:   %.3f\n\n' ... 
    'GENERALIZATION:\n' ...
    'MSE increase: +%.1f%%\n' ...
    'MAE increase: +%.1f%%\n' ...
    'R² decrease:  %.3f'], ...
    mse_val, mae_val, R2_val, ...
    mse_test, mae_test, R2_test, ...
    100*(mse_test/mse_val - 1), ...
    100*(mae_test/mae_val - 1), ...
    R2_val - R2_test);
subplot(3, 3, 8);
text(0.1, 0.9, stats_text1, 'FontSize', 10, 'VerticalAlignment', 'top', ...
    'BackgroundColor', [0.95 0.95 0.95], 'EdgeColor', 'black', ...
    'HorizontalAlignment', 'left');
axis off;
title('Model Summary');
subplot(3, 3, 9);
text(0.1, 0.9, stats_text2, 'FontSize', 10, 'VerticalAlignment', 'top', ...
    'BackgroundColor', [0.95 0.95 0.95], 'EdgeColor', 'black', ...
    'HorizontalAlignment', 'left');
axis off;
title('Performance Summary');
end

