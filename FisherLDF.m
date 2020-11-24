% % clear all
%% Training
% Loading Images 
load('data_batch_1.mat')
training_imgs = imageDatastore('CIFAR_10_TRAINING');
training_labels = (labels);
load('test_batch.mat')
test_imgs = imageDatastore('CIFAR_10_TEST');
test_labels = (labels);
% Image Parameter Instantiation
img_size  = size(readimage(training_imgs,1));
flattened_img_size = img_size(1)*img_size(2)*img_size(3);
num_of_classes = 10;
mean_vector = zeros(num_of_classes, flattened_img_size);
A = zeros(flattened_img_size,flattened_img_size); % sum of covariances
cov_vector = [];
% Find mean vector, cov vector, and A features from the 10 training images
for class_index = 1 : num_of_classes % per class
    class_label = class_index-1; % because matlab does not have 0 indexing and the labels are from  0-9
    class_imgs_index = find(training_labels == class_label); % finding the indices of the imgs that correspond to a class
    num_of_imgs = length(class_imgs_index);
    class_imgs  = zeros(num_of_imgs,flattened_img_size);
    for i= 1 : num_of_imgs % extracting images that correspond to the class at hand
        img = readimage(training_imgs,class_imgs_index(i)); % reading the img given the index for the class at hand 
        class_imgs(i,:) = reshape(img,flattened_img_size,1);
    end 
    cov_vector{class_index} = cov(class_imgs);
    A = A +  cov(class_imgs); % adding covariances at each iteration
    mean_vector(class_index,:) = mean(class_imgs);
end
% Finding overall mean and B 
overall_mean = mean(mean_vector);
B = zeros(flattened_img_size,flattened_img_size); % sum of covariances
for i= 1 : num_of_classes
   B = B + ((mean_vector(i,:) - overall_mean)'*(mean_vector(i,:) - overall_mean));
end
[H,D] = eig(inv(A)*B); % LDF vector
H = H(:,1:9); % selecting the 9 top eig values 9 -> num of rank 

%% Classification after projecting to FISHER LDF SPACE
confmtrx = zeros(10,10); 
num_of_imgs = 1000;
for i = 1 : num_of_imgs
    image_label = test_labels(i) + 1; % because matlab does not have 0 indexing and the labels are from  0-9
    for class_index = 1 : num_of_classes % per class
         img = readimage(test_imgs,i); % reading 
         feature_vector = double(reshape(img,flattened_img_size,1));
         % Projecting to Fisher Space
         fisher_LDF_vector = H'*feature_vector;
         fisher_mean = H'*mean_vector(class_index,:)';
         fisher_cov = H'*cov_vector{class_index}*H;
         all_distances(class_index) = my_mahalanobis( fisher_LDF_vector, fisher_mean, fisher_cov);
    end 
    [min_value, min_index] = min(all_distances); % finding the min distances 
    confmtrx (image_label, min_index) = confmtrx (image_label, min_index) + 1;
    i
end