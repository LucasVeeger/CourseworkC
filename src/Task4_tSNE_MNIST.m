load ('mnist.mat');


% crop data to x samples
samples = 10000;
feats = 28*28;
digits_train = double(digits_train(:,:,1:samples));
digits_train = double(reshape(digits_train, [28*28,samples])');

Y = tsne(digits_train);

%%
gscatter(Y(:,1), Y(:,2), labels_test(1:samples))