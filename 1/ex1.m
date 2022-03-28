clc;
close all;
clear all;

data_path = "data.txt";
label_path = "labels.txt";

epsilon = 1e-5;
maxiter = 1000;

data = load(data_path);
labels = load(label_path);

data = [data ones(size(data, 1), 1)];

test_data = data(2001:4601, :);
test_labels = labels(2001:4601);

train_sizes = [200 500 800 1000 1500 2000];
performances = [];

for i = (1:length(train_sizes))
    %get training data and label
    train_data = data(1:train_sizes(i), :);
    train_labels = labels(1:train_sizes(i));
    weights = logistic_train(train_data, train_labels, epsilon, maxiter);

    %test
    results = 1 * (sigmoid(test_data * weights)>0.5);
    performances = [performances sum(results==test_labels)/2601];
    
end

figure;
plot(train_sizes, performances);
xlabel("Training Data Size n");
ylabel("Performance on Testing Data");
title("Experiment 1");