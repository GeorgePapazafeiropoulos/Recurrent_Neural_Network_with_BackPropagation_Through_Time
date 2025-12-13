%% Local function: data_create_sequences
function [X_seq, y_seq] = data_create_sequences(X, y, sequence_length)
n_sequences = length(X) - sequence_length;
X_seq = zeros(n_sequences, sequence_length);
y_seq = zeros(n_sequences, 1);
for i = 1:n_sequences
    X_seq(i, :) = X(i:(i + sequence_length - 1));
    y_seq(i) = y(i + sequence_length);
end
end

