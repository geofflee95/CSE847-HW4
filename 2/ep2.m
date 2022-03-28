clc;
close all;
clear all;

cd 'SLEP-master'
root = cd;
addpath(genpath([root '/SLEP']))

data_path = "ad_data.mat";
feature_path = "feature_name.mat";

load(data_path);

pars = [0.000000001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];

aucs = []
num_f = []

for i = (1:length(pars))
    %training
    [w, c] = logistic_l1_train(X_train, y_train, pars(i));

    %number of feature selected
    num_f = [num_f sum(w ~= 0)];

    %auc
    predictions = 1 * sigmoid(X_test * w + c);
    [~,~,~,auc] = perfcurve(y_test, predictions, 1);
    aucs = [aucs auc];

end

figure;
plot(pars, aucs)
xlabel("L1 regularization parameter");
ylabel("AUC")

figure;
plot(pars, num_f)
xlabel("L1 regularization parameter")
ylabel("Number of Feature Selected");