%% Local function: rnn_train
function [W1, W2, W3, b1, b2, train_loss, val_loss] = rnn_train(...
    X_train, y_train, X_val, y_val, W1, W2, W3, b1, b2, learning_rate,...
    epochs, batch_size, window_length,patience)
n_train = size(X_train, 1);
n_batches = ceil(n_train / batch_size);
% Initialize output
train_loss = zeros(epochs, 1);
val_loss = zeros(epochs, 1);
% Training parameters
best_val_loss = inf;
patience_counter = 0;
current_lr = learning_rate;
% Store best weights
best_W1 = W1; best_W2 = W2; best_W3 = W3;
best_b1 = b1; best_b2 = b2;
for epoch = 1:epochs
    epoch_loss = 0;
    % Learning rate scheduling
    if epoch > 50 && mod(epoch, 10) == 0
        current_lr = current_lr * 0.95;
    end
    % Shuffle data
    indices = randperm(n_train);
    X_train_shuffled = X_train(indices, :);
    y_train_shuffled = y_train(indices, :);
    for batch = 1:n_batches
        % Get batch
        start_idx = (batch-1) * batch_size + 1;
        end_idx = min(batch * batch_size, n_train);
        batch_size_actual = end_idx - start_idx + 1;
        X_batch = X_train_shuffled(start_idx:end_idx, :);
        y_batch = y_train_shuffled(start_idx:end_idx);
        % Forward pass
        [h, outputs, ~] = rnn_forward(X_batch, W1, W2, W3, b1, b2);
        % Compute loss
        batch_loss = mean((outputs(:, end) - y_batch).^2);
        epoch_loss = epoch_loss + batch_loss;
        % Backward pass with gradient checking
        [dW1, dW2, dW3, db1, db2] = rnn_backward(...
            X_batch, y_batch, h, outputs, W1, W2, W3, b1, b2, ...
            batch_size_actual, window_length);
        % Update weights
        W1 = W1 - current_lr * dW1;
        W2 = W2 - current_lr * dW2;
        W3 = W3 - current_lr * dW3;
        b1 = b1 - current_lr * db1;
        b2 = b2 - current_lr * db2;
    end
    train_loss(epoch) = epoch_loss / n_batches;
    % Validation
    [~, val_outputs, ~] = rnn_forward(X_val, W1, W2, W3, b1, b2);
    val_loss(epoch) = mean((val_outputs(:, end) - y_val).^2);
    % Monitor outputs and weights
    if mod(epoch, 10) == 0 || epoch <= 5
        output_range = [min(val_outputs(:, end)), max(val_outputs(:, end))];
        target_range = [min(y_val), max(y_val)];
        fprintf('Epoch %d - LR: %.4f, Output: [%.3f, %.3f], Targets: [%.3f, %.3f]\n', ...
            epoch, current_lr, output_range(1), output_range(2), target_range(1),...
            target_range(2));
    end
    % Early stopping
    if val_loss(epoch) < best_val_loss
        best_val_loss = val_loss(epoch);
        best_W1 = W1; best_W2 = W2; best_W3 = W3;
        best_b1 = b1; best_b2 = b2;
        patience_counter = 0;
    else
        patience_counter = patience_counter + 1;
    end
    if patience_counter >= patience
        fprintf('Early stopping at epoch %d\n', epoch);
        W1 = best_W1; W2 = best_W2; W3 = best_W3;
        b1 = best_b1; b2 = best_b2;
        break;
    end
end
% Trim loss histories
if epoch < epochs
    train_loss = train_loss(1:epoch);
    val_loss = val_loss(1:epoch);
end
end

