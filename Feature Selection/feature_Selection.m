function feature_selection()
    clc;
    filename = input('Type the name of the file to test: ','s');
    Data = load (filename);
    %Data = load ('Input_23.txt', '-ascii');
    Y = Data(:,end);
    X = Data(:,1:end-1);
    fprintf('\nType the number of the algorithm you want to run\n');
    fprintf('1) Forward Approach Method\n');
    fprintf('2) Backward Approach Method\n');
    ip = input('Input: ');
    
    [m n] = size(X);
    
    fprintf('\n\nThis dataset has %d features (not including the class attribute), with %d instances.\n',n,m);
    fprintf('\n');
    fprintf('Please wait while I normalize the data....');
    fprintf('Done!\n\n');   
    model = fitcknn(X,Y);
    cvmodel = crossval(model,'KFold',m);
    kfloss = kfoldLoss(cvmodel);
    cvacc = 1.00 - kfloss;
    
    fprintf('Running nearest neighbor with %d features, using "leaving-one-out" evaluation, I get an accuracy of %.1f %%\n\n',n,cvacc*100);
    fprintf('Beginning Search\n\n');
    if ip == 1
        forward_selection(X,Y);   
    elseif ip == 2
        backward_selection(X,Y);
    end
end
