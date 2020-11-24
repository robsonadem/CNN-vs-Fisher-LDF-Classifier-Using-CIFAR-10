%Download CIFAR-10 Image Data
cifar10Data = tempdir;
url = 'https://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz';
helperCIFAR10Data.download(url,cifar10Data);
% Load the CIFAR-10 training and test data. 
[trainingImages,trainingLabels,testImages,testLabels] = helperCIFAR10Data.load(cifar10Data);
% Each image is a 32x32 RGB image and there are 50,000 training samples.
size(trainingImages)
% CIFAR-10 has 10 image categories. List the image categories:
numImageCategories = 10;
categories(trainingLabels)

% Display a few of the training images.
figure
thumbnails = trainingImages(:,:,:,1:100);
montage(thumbnails)

layers = [
    % 32 x 32 x 3 images with 'zerocenter' normalization
    imageInputLayer([32 32 3])
    % 32 5 x 5 convolutions with stride= 1 and padding = 2
    convolution2dLayer(5,32,'Stride',1,'Padding',2)
    
    %     A batch normalization layer normalizes each input channel across a mini-batch. 
    %     To speed up training of convolutional neural networks and reduce the sensitivity
    %     to network initialization, use batch normalization layers between convolutional 
    %     layers and nonlinearities, such as ReLU layers.
    
    batchNormalizationLayer
    % ActivationFunction ReLU
    reluLayer
    % 2x2 max pooling with stride= 2
    maxPooling2dLayer(2,'Stride',2)
    % 32 5 x 5 convolutions with stride= 1
    convolution2dLayer(5,32,'Stride',1)
    batchNormalizationLayer
    % Activation Function ReLU
    reluLayer
    % 2x2 max pooling with stride= 2
    maxPooling2dLayer(2,'Stride',2)
    % 64 5 x 5 convolutions with stride= 1
    convolution2dLayer(3,32,'Padding','same')
    % Use batch normalization layers between convolutional 
    % layers and nonlinearities, such as ReLU layers.
    batchNormalizationLayer
    % ActivationFunction ReLU
    reluLayer
    % 2x2 max pooling with stride = 2
    maxPooling2dLayer(2,'Stride',2)
    % Fully Connected Layer64 
    fullyConnectedLayer(64)
    % Activation Function ReLU
    reluLayer
    % Fully Connected Layer64 
    fullyConnectedLayer(10)
    % Activation Function Soft max
    % A Softmax function is a type of squashing function. 
    % Squashing functions limit the output of the function into the range 0 to 1.
    % This allows the output to be interpreted directly as a probability.
    softmaxLayer
    classificationLayer];

opts = trainingOptions('sgdm', ...
    'Momentum', 0.9, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', 40, ...
    'MiniBatchSize', 128, ...
    'Verbose', true);
% A trained network is loaded from disk to save time when running the
% example. Set this flag to true to train the network.
doTraining = true;

if doTraining    
    % Train a network.
    cifar10Net = trainNetwork(trainingImages, trainingLabels, layers, opts);
else
    % Load pre-trained detector for the example.
    load('rcnnStopSigns.mat','cifar10Net')       
end 

% Extract the first convolutional layer weights
w = cifar10Net.Layers(2).Weights;

% rescale the weights to the range [0, 1] for better visualization
w = rescale(w);

figure
montage(w)

% Run the network on the test set.
YTest = classify(cifar10Net, testImages);

% Calculate the accuracy.
accuracy = sum(YTest == testLabels)/numel(testLabels);

% figure; 
plotconfusion(YTest,testLabels)

