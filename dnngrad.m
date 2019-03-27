function grad= dnngrad(nn,x,y,nograd)
%  dnngrad: evaluate and compute gradient
%  grad=dnngrad(nn,x,y,nograd,act,nograd)
%
%  nn: deep neural-net
%  nn.logl2 (=0: squared loss function
%            =1: negative log-likelihood loss function)
%  x: input features of size minibatch-size x feature-dimension
%  y: output vectors of size minibatch-size x output-dimension
%  nograd: if true, then do not compute gradient
%
% return grad:
%        grad.loss: loss function
%        grad.out: output (if gradient not computed)
%        grad.W: layer-by-layer weight gradient matrices
%        grad.b: layer-by-layer bias gradient vectors
%        grad.slope: layer-by-layer slope gradient vectors
%        grad.segment: layer-by-layer segment gradient vectors

numcases = size(x,1);  % minibatch size
n        = nn.nlayers;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Learning Sparseness-Inducing Activation Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%% feed forward %%%
a{1} = x; 
for layer = 1 : n-2 
    a1{layer+1} = a{layer} *nn.W{layer}+repmat(nn.b{layer}, numcases, 1); 
    aunm        = nn.size(layer+1);
    a2{layer+1} = min(repmat(nn.segment{layer}(aunm+1:end)-...
                  nn.segment{layer}(1:aunm*nn.piecenum), numcases, 1),...
                  max(0,repmat(a1{layer+1},1,nn.piecenum)-...
                  repmat(nn.segment{layer}(1:aunm*nn.piecenum), numcases,1)));
    bb          = repmat(nn.slope{layer}, numcases, 1).*a2{layer+1};
    a{layer+1}  = bb(:,1:aunm);
    for jj = 2:nn.piecenum
      a{layer+1} = a{layer+1}+bb(:,(jj-1)*aunm+1:jj*aunm);
    end
end

%%% output layer
if (nn.logl2 == 1)
   a{n}        = exp(repmat(nn.b{n-1}, numcases, 1) + a{n-1} * nn.W{n-1}); 
   a{n}        = a{n}./repmat(sum(a{n},2),1,nn.size(n));
   grad.loss   = -sum(sum(y.*log(a{n})));
else
   a{n}        = (repmat(nn.b{n-1}, numcases, 1) + a{n-1} * nn.W{n-1});
   grad.loss   = sum(sum((a{n}-y).^2));
end

%%% no need to compute grad
if (nograd)
   grad.out   = a{n};
   return;
end
%%%  back prop
if (nn.logl2 == 1)
    d{n}      = (a{n}-y); 
else
    d{n}      = 2*(a{n}-y); 
end
for layer = (n - 1) : -1 : 2
    d1{layer} = (d{layer + 1} * nn.W{layer}');  
    aunm      = nn.size(layer);
    d2{layer} = repmat(d1{layer},1,nn.piecenum).* repmat(nn.slope{layer-1},...
                       numcases, 1).*(a2{layer}>0).*...
                (a2{layer}<repmat(nn.segment{layer-1}(aunm+1:end)-...
                 nn.segment{layer-1}(1:aunm*nn.piecenum), numcases, 1));
    d{layer}  = d2{layer}(:,1:aunm);
    for jj = 2:nn.piecenum
       d{layer} = d{layer}+d2{layer}(:,(jj-1)*aunm+1:jj*aunm);
    end
end

%%%% compute grad
for layer = 1 : (n-1)
    grad.W{layer}         = (a{layer}'*d{layer + 1}) / numcases; 
    grad.b{layer}         = sum(d{layer + 1}, 1) / numcases;
    if layer < n-1
       % In this paper the learning rate of the slope is 50*opts.alpha. 
       % you can adjust the learning rate of the slope.
       grad.slope{layer}  = 50*sum(a2{layer+1}.*repmat(d1{layer + 1},...
                                 1,nn.piecenum),1)/ numcases; 
       grad.segment{layer}= -sum(d2{layer + 1}, 1) / numcases;
    end
end
end
