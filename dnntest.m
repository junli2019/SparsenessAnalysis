function [err_loss,err_class]= dnntest(nn,x,y,numcases)
% nntest: test deep neural-net
%  [err_loss,err_class]=dnntest(nn,x,y,numcases)
%
%  nn: deep neural-net
%  x: input
%  y: output
%  numcases: minibatch size
%
%  return: loss and classification-error
%

allnumber=size(x,1);
numbatches=ceil(allnumber/numcases);
nn.test = 1;
err_loss=0;
err_class=allnumber;

for batcht = 1 : numbatches
  data    = x((batcht-1)*numcases+1:min([batcht*numcases allnumber]),:);
  targets = y((batcht-1)*numcases+1:min([batcht*numcases ...
	allnumber]),:);
  value=dnngrad(nn,data,targets,true);
  targetout=value.out;

  [I J]=max(targetout,[],2);
  [I1 J1]=max(targets,[],2);
  err_class=err_class-length(find(J==J1));
  err_loss = err_loss + value.loss;
end
nn.test =0;
err_class=err_class/size(x,1);
err_loss=err_loss/size(x,1);
