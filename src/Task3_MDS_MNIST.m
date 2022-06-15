load ('mnist.mat');


% crop data to x samples
samples = 1000;
feats = 28*28;
digits_train = double(digits_train(:,:,1:samples));
digits_train = double(reshape(digits_train, [28*28,samples])');

C = center_datapoints(digits_train);

sim_mat = zeros(samples,samples);
for i = 1:samples
    for j = 1:samples
        % euclidean distance 
%         diff = C(i, :) - C(j, :);
%         diff = sqrt(diff * diff');
        diff = pdist2(C(i, :), C(j, :), 'euclidean');
        % similarity matrix
        sim_mat(i,j) = diff;
    end 
end
% make squared distance
sim_mat = sim_mat.*sim_mat;

center_mat = eye(samples,samples) - (1/samples)*ones(samples,samples);

gram = -0.5*center_mat*sim_mat*center_mat;

% gram = ...
%     -1/2*((eye(samples,samples)-1/samples*ones(samples,1)*ones(samples,1)')*...
%     sim_mat(:,:)*(eye(samples,samples)-1/samples*ones(samples,1)*ones(samples,1)'));

%% 4. Compute eigenvalues and eigenvectors of the GRAM matrix

% calculate eigenvals and sort 
[evs_unsorted, l_unsorted]=eig(gram, 'matrix');
[l, idx]=sort(diag(l_unsorted),'descend');
evs = evs_unsorted(idx,:);

% reduce to d dimensions
d = 2;
x_mds = diag(l(1:d))*evs(:, 1:d)';
Y = mdscale(sim_mat, d);

figure
gscatter(x_mds(2,:), x_mds(1,:), ...
    labels_train(idx,1))
figure
gscatter(Y(:, 2), Y(:, 1), ...
    labels_train(1:samples))

%% Stresses 

stress = (squareform(pdist(x_mds')) - sim_mat)^2;


%%
function [c] = center_datapoints(datapoints)
    c = datapoints - mean(datapoints);
end