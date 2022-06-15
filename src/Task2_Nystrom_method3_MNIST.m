load ('mnist.mat');

% crop data to x samples
samples = 60000;
feats = 28*28;
digits_train = digits_train(:,:,1:samples);
digits_train = double(reshape(digits_train, [28*28,samples])');
% center
C = center_datapoints(digits_train);

% pick landmark coordinates
all_idx = 1:1:feats;
lmc = 1:9:feats; 
lmc_size = size(lmc,2);
digit_lmc = digits_train(:, lmc); 
unsort_idx = [lmc setdiff(all_idx, lmc)];
digit_rest = digits_train(:, setdiff(all_idx, lmc));
matrix_AB = [digit_lmc digit_rest];

cov_mat = (matrix_AB' * matrix_AB)./samples;    % examples show {/(m-1)}?
mat_A = cov_mat(1:lmc_size, 1:lmc_size);
mat_B = cov_mat(lmc_size + 1:end, 1:lmc_size);

%% 4. Compute eigenvalues and eigenvectors of the covariance matrix

% calculate eigenvals and sort 
[evs_unsorted, l_unsorted]=eig(mat_A, 'matrix');
[l, idx]=sort(diag(l_unsorted),'descend');
l_diag = l.*eye(size(l,1),size(l,1));
evs = evs_unsorted(:,idx); 

% PCA modes
U_unsorted = [evs; mat_B*evs*inv(l_diag)];
U = U_unsorted(unsort_idx, :);

% reduce to d dimensions
d = 2;
l_redu = l(1:d);
U_redu = U(:,1:d);
digits_projected = U_redu'*digits_train';

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

