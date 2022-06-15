load ('mnist.mat');

% crop data to x samples
samples = 10000;
feats = 28*28;
digits_train = digits_train(:,:,1:samples);
digits_train = double(reshape(digits_train, [28*28,samples])');
% center
C = center_datapoints(digits_train);

% pick landmark coordinates
lmc = 1:9:784; 
lmc_size = size(lmc,2);
digit_lmc = digits_train(:, lmc); 

cov_mat = (digit_lmc' * digit_lmc)./samples;    % examples show {/(m-1)}?

%% 4. Compute eigenvalues and eigenvectors of the covariance matrix

% calculate eigenvals and sort 
[evs_unsorted, l_unsorted]=eig(cov_mat, 'matrix');
[l, idx]=sort(diag(l_unsorted),'descend');
evs = evs_unsorted(:,idx); 

% reduce to d dimensions
d = 2;
l_redu = l(1:d);
evs_redu = evs(:,1:d);
digits_projected = evs_redu'*digit_lmc';

% explained variance (lamda_i/sum(lamda))
l_expl = zeros(lmc_size,1);
l_sum = sum(l);
for i = 1:lmc_size
    l_expl(i) = sum(l(1:i))/l_sum;
end

figure
gscatter(digits_projected(2,:), digits_projected(1,:), ...
    labels_train(1:samples))
figure
axes('LineWidth',0.6,...
    'FontName','Helvetica',...
    'FontSize',8,...
    'XAxisLocation','Origin',...
    'YAxisLocation','Origin')
xlim([1 300])
line(1:lmc_size,l_expl);


%%
function [c] = center_datapoints(datapoints)
    c = datapoints - mean(datapoints);
end

