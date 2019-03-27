function [train_d,train_t,val_d,val_t,test_d,test_t] = createdata(digtrain,digtest)

trainnumber = size(digtrain,1);
testnumber  = size(digtest,1);
dnumber     = size(digtrain,2);
lablenumber = max(digtrain(:,end));

valnumber   = floor(0.1*trainnumber);

train_dtemp     = zeros(trainnumber,dnumber-1);      % all train data
train_ttemp     = zeros(trainnumber,lablenumber+1);  % all train target

test_d      = zeros(testnumber,dnumber-1);       % test data
test_t      = zeros(testnumber,lablenumber+1);   % test target

for i=1:trainnumber
    if mod(i,500)==0
      fprintf(1,'train example number %d',i); fprintf(1,'\n');  
    end
    train_dtemp(i,:)   = digtrain(i,1:dnumber-1);
    j              = digtrain(i,end);
    train_ttemp(i,j+1) = 1;      % [0 0 1 0 0 0 0 0 0 0] when j=2
end
index       = randperm(trainnumber);
train_d     = train_dtemp(index(valnumber+1:trainnumber),:); % train data
train_t     = train_ttemp(index(valnumber+1:trainnumber),:);  % train target

val_d       = train_dtemp(index(1:valnumber),:);  % val data
val_t       = train_ttemp(index(1:valnumber),:);  % val target

for i=1:testnumber
    if mod(i,500)==0
      fprintf(1,'test example number %d',i); fprintf(1,'\n');  
    end
    test_d(i,:)    = digtest(i,1:dnumber-1);
    j              = digtest(i,end);
    test_t(i,j+1)  = 1;      % [0 0 0 0 0 1 0 0 0 0] when j=5             
end
end