function net = dnntrain(trainx, trainy, valx, valy, testx, testy, opts)
%  dnntrain: train deep neural-network using SGD
%  net=dnntrain(trainx,trainy, testx, testy, opts)
%
%  return net: trained deep neural-net
%
%  trainx: training input features
%  trainy: training output vector
% 
%  testx: test input features
%  testy: test output vector
%
%  opts: training options (examples below)
%  opts.size        = [784 1000 1000 1000 10]; % input-hidden-output layer sizes
%  opts.numcases    = 100;        % minibatch size
%  opts.lambda      = 0.00001;    % L2 weight decay
%  opts.alpha       = 0.01;       % learning rate
%  opts.updateLayers=1:length(opts.size)-1; % layers to be updated
%  opts.inputnoise  = 0;          % add noise to inputdata
%  opts.piecenum    = 2;          % the number of segments
%  opts.lengthaf    = 1;          % interval length of the activation functions
%  opts.maxslope    = 2;          % initial slope starts at a value of maxslope 
%                                 % and linearly decreases to 0 over piecenum 
%  opts.logl2       = 0;          % 0: squared loss function
%                                 % 1: negative log-likelihood loss function
%  opts.numepochs   = 200;        % all training epoch
%  opts.tinterepoch = 1;          % interval epoch of test
%  opts.useGPU      = true;       % whether to use GPU
%


%%%%%% print training parameters %%%%%%
fprintf(1,'nn training parameters:\n');

fprintf(1,'network size=');
fprintf(1,' %d',opts.size); fprintf(1,'\n');

fprintf(1,'  layers to be trained:');
fprintf(1,' %d',opts.updateLayers); fprintf(1,'\n');

if (opts.logl2==0)
   fprintf(1,'  Objective function = squared loss function\n');
else
   fprintf(1,'  Objective function = negative log-likelihood loss function\n');
end
fprintf(1,'epochs=%d\n',opts.numepochs);
fprintf(1,'minibatch=%d\n',opts.numcases);
fprintf(1,'learning-rate=%d\n',opts.alpha);
fprintf(1,'L2-regularization=%d\n',opts.lambda);

best_val_err=1e1;

%%%%%% initialize training %%%%%%%
net.size         = opts.size;
net.nlayers      = numel(net.size);
net.logl2        = opts.logl2;
net.inputnoise   = opts.inputnoise; 
net.piecenum     = opts.piecenum;
net.trainerror   = [];
net.valerror     = [];
net.testerror    = [];
net.trainloss    = [];
    
numcases         = opts.numcases;
allnumber        = size(trainx,1);
numbatches       = ceil(allnumber/numcases);

if (opts.useGPU)
  trainx         = gpuArray(trainx);
  trainy         = gpuArray(trainy);
  valx           = gpuArray(valx);
  valy           = gpuArray(valy);
  testx          = gpuArray(testx);
  testy          = gpuArray(testy);
end

net.intersegment = opts.lengthaf/opts.piecenum;
net.interslope   = opts.maxslope/opts.piecenum;

for i = 2 : net.nlayers
    %  initialize weights      
    %%% use random weights
    net.b{i - 1} = zeros(1,net.size(i));  %  biases
    net.W{i - 1} = 0.01 * randn(net.size(i-1),net.size(i));  % weights 
    if i < net.nlayers
       %%%%%%% initial segment points
       net.segment{i-1}  = [];
       for ii = 1:net.piecenum+1
           net.segment{i-1}= [net.segment{i-1} ...
                             (ii-1)*net.intersegment+zeros(1,net.size(i))]; 
       end 
       %%%%%%% initial random slopes   
       net.slope{i-1}    = [];
       for ii = 1:net.piecenum
           net.slope{i-1}= [net.slope{i-1} opts.maxslope-(ii-1)*...
                            net.interslope+0.01*randn(1,net.size(i))]; 
       end
    end     
    % grad momentum
    db{i - 1}            = zeros(size(net.b{i-1}));
    dW{i - 1}            = zeros(size(net.W{i-1}));
    if i<net.nlayers
       dslope{i - 1}     = zeros(size(net.slope{i-1}));
       dsegment{i - 1}   = zeros(size(net.segment{i-1}));    
    end
   
  % whether to use GPU 
  if (opts.useGPU) 
     net.W{i-1}          = gpuArray(net.W{i-1});
     dW{i-1}             = gpuArray(dW{i-1}); 
     net.b{i-1}          = gpuArray(net.b{i-1});
     db{i-1}             = gpuArray(db{i-1}); 
     if i<net.nlayers
        net.slope{i-1}   = gpuArray(net.slope{i-1});
        dslope{i-1}      = gpuArray(dslope{i-1});
        net.segment{i-1} = gpuArray(net.segment{i-1});
        dsegment{i-1}    = gpuArray(dsegment{i-1});
    end
  end
end

%%%%  start training for opts.numepochs epochs %%%%%
for epoch = 1 : opts.numepochs
    tic;
    ffcs = 0;
    %%%% set momentum and learning rate %%%%
    if epoch<min(15,opts.numepochs/2)
       momentum = 0.5+0.8*epoch/opts.numepochs;    
    else
       momentum = 0.90;
    end
    eta         = opts.alpha*(1-momentum);    
    index       = randperm(allnumber);    
    %%% go through minibatches
for batch = 1 : numbatches
    data        = trainx(index((batch-1)*numcases+...
                         1:min([batch*numcases allnumber])),:);
    targets     = trainy(index((batch-1)*numcases+...
                         1:min([batch*numcases allnumber])),:);   
    %%% add noise to inputdata 
    if net.inputnoise~=0
       if (opts.useGPU) 
          data  = data.*(+(gpuArray.rand(size(data)) > net.inputnoise));  
       else
          data  = data.*(rand(size(data)) > net.inputnoise);  
       end
    end    
        
	%%% evaluate gradient %%%
	grad        = dnngrad(net,data,targets,false);
 
	%%% outdate gradients for layers to be updated 
	for layer = opts.updateLayers
        %%%% update weights 
        dW{layer}       = momentum * dW{layer} - eta * (grad.W{layer} + ...
                                    opts.lambda * net.W{layer});                    
	    net.W{layer}    = net.W{layer} + dW{layer};
        %%%% update biases
        db{layer}       = momentum * db{layer} - eta * (grad.b{layer});     
        net.b{layer}    = net.b{layer} + db{layer};
       if layer < net.nlayers-1 
          %%%% update slopes
          dslope{layer} = momentum * dslope{layer} - eta * (grad.slope{layer});   
          net.slope{layer} = net.slope{layer} + dslope{layer}; 
          %%%% update segment points
          aunm=net.size(layer+1);  
          dsegment{layer}(aunm+1:end) = momentum * dsegment{layer}(aunm+1:end) ...
                                         - eta * (grad.segment{layer});   
          net.segment{layer}(aunm+1:end) = net.segment{layer}(aunm+1:end)...
                                           + dsegment{layer}(aunm+1:end); 
       end        
    end    
    ffcs=ffcs+grad.loss;
end     
    net.trainloss=[net.trainloss ffcs];
    t = toc;
    disp(['epoch ' num2str(epoch) '/' num2str(opts.numepochs) ' (' ...
            num2str(t) ' seconds)' ': obj=' num2str(net.trainloss(end)) ]);
    [train_loss,train_err] = dnntest(net,trainx,trainy,numcases);
    [val_loss,val_err]   = dnntest(net,valx,valy,numcases);
    net.trainerror         = [net.trainerror train_err];
    net.valerror           = [net.valerror val_err];
    
    if (best_val_err>val_err)
        best_val_err = val_err;
        net_best     = net;
    end
    %%%% output progress on training and val data %%%%%%
    if (mod(epoch,opts.tinterepoch)==0)
       disp(['train-loss= ' num2str(train_loss) ...
              '. val-loss= ' num2str(val_loss)]);
       disp(['train-error=' ...
	         num2str(train_err) '; val-error=' num2str(val_err)]);
    end
end

%%%% output progress on training, val and test data %%%%%%
[train_loss,train_err] = dnntest(net_best,trainx,trainy,numcases);
[val_loss,val_err]   = dnntest(net,valx,valy,numcases);
[test_loss,test_err]   = dnntest(net_best,testx,testy,numcases);
disp(['train-loss= ' num2str(train_loss) 'val-loss= ' num2str(val_loss)...
      '. test-loss= ' num2str(test_loss)]);
disp(['train-error=' num2str(train_err) '; val-error=' num2str(val_err)...
      '; test-error=' num2str(test_err)]);


%%%% finish %%%%%
if (opts.useGPU)
for layer=1:net.nlayers-1
    net.W{layer}          = gather(net.W{layer});
    net.b{layer}          = gather(net.b{layer});
    if layer<net.nlayers-1
       net.slope{layer}   = gather(net.slope{layer});
      net.segment{layer}  = gather(net.segment{layer});
    end
end
end
end



