
clear all;
%%%%%%%%%%% make data-mnistVariations%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% mnist-basic;
% download from the web---http://www.iro.umontreal.ca/~lisa/icml2007data/mnist.zip
% after you extract mnist.zip, you will see 
% mnist_train.amat and mnist_test.amat
% train_d: train data
% train_t: train target
% val_d: val data
% val_t: val target
% test_d: test data
% test_t: test target

digtrain = load('mnist_train.amat');
digtest  = load('mnist_test.amat');
[train_d,train_t,val_d,val_t,test_d,test_t]=createdata(digtrain,digtest);
% i=200; Img=reshape(train_d(i,:),28,28); 
% imshow(Img);
save mnistbasic train_d train_t val_d val_t test_d test_t;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%% test  mnistbasic %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%% parameter setup %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
load mnistbasic;
opts.size        = [784 2000 2000 2000 10]; % input-hidden-output layer sizes
opts.numcases    = 100;        % minibatch size
opts.lambda      = 0.00001;    % L2 weight decay
opts.alpha       = 0.2;       % learning rate
opts.updateLayers=1:length(opts.size)-1; % layers to be updated
opts.inputnoise  = 0.2;        % add noise to inputdata
opts.piecenum    = 2;          % the number of segments
opts.lengthaf    = 1;          % interval length of the activation functions
opts.maxslope    = 2;          % initial slope starts at a value of maxslope 
                               % and linearly decreases to 0 over piecenum 
opts.logl2       = 0;          % 0: squared loss function
                               % 1: negative log-likelihood loss function
opts.numepochs   = 5;        % all training epoch
opts.tinterepoch = 1;          % interval epoch of test
opts.useGPU      = false;      % whether to use GPU

% supervised training
nnmnistbasic = dnntrain(train_d, train_t, val_d, val_t, test_d, test_t,opts); 




