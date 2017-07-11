clc;
clear all;
close all;

disp(' ## PROGRAM FOR ASSIGNMENT 4: FACE RECOGNITION ## '); 

%file directory
    
directory=char(pwd);
TrainingDirectory = 'training\';
TestingDirectory = 'testing\';


TrainingFiles = dir(TrainingDirectory);
TestingFiles = dir(TestingDirectory);



% Training process

tt= cputime;

% For image training

training_image_num=1;
count=0;

for iFile = 3:size(TrainingFiles,1);
     
    %loading the image and converting into vector
    sub_dir=dir([TrainingDirectory TrainingFiles(iFile).name]);
    count=count+1;
    
    for iFile1 = 3:size(sub_dir,1);
    
        origIm=imread([TrainingDirectory TrainingFiles(iFile).name '\' sub_dir(iFile1).name]);
        origIm=rgb2hsv(origIm);
        vIm=reshape(origIm,[3600 1]);
        train_matrix(:,training_image_num)=vIm;
        train_subject_number(training_image_num,1)=count;
        
        training_image_num=training_image_num+1;
        
    end
        
end

train_matrix1=normalizeIm1(train_matrix,training_image_num);
mean_train=zeros(3600,1);

for iFile = 1:training_image_num-1;
    
    mean_train=mean_train+train_matrix1(:,iFile);

end

mean_train=mean_train/training_image_num;
new_train_matrix = bsxfun(@minus, train_matrix1, mean_train);
disp('Mean Calculated!');

covar_train=zeros(3600,3600);

for iFile = 1:training_image_num-1;
    
     a=new_train_matrix(:,iFile);
     b=(transpose(a));
     covar_train_numerator=a*b;
     covar_train=covar_train + covar_train_numerator;
end    

covar_train=covar_train/training_image_num;
disp('Covariance Calculated!');
%[Eigen_Vectors,Eigen_Values] = eig(covar_train);
%disp('Eigen Vectors Calculated!');


[evectors, score, evalues] = pca(train_matrix1'); % calculate the ordered eigenvectors and eigenvalues
num_eigenfaces = 56; % Selecting K
evectors = evectors(:, 1:num_eigenfaces);
feature_reference = evectors' * new_train_matrix; % project the images into the subspace to generate the feature vectors
disp('feature_reference created!');

% Creating testing image matrix


testing_image_num=1;
count=0;

for iFile = 3:size(TestingFiles,1);
     
    %loading the image and converting into vector
    sub_dir=dir([TestingDirectory TestingFiles(iFile).name]);
    count=count+1;
    
    
    for iFile1 = 3:size(sub_dir,1);
    
        origIm=imread([TestingDirectory TestingFiles(iFile).name '\' sub_dir(iFile1).name]);
        origIm=rgb2hsv(origIm);
        vIm=reshape(origIm,[3600 1]);
        test_matrix(:,testing_image_num)=vIm;
        test_subject_number(testing_image_num,1)=count;
        
        testing_image_num=testing_image_num+1;
    
    end
        
end

test_matrix1=normalizeIm1(test_matrix,testing_image_num);


% INFERENCE:

new_test_matrix = bsxfun(@minus, test_matrix1, mean_train); % Substracting mean face
feature_test = evectors' * new_test_matrix; % project the testing images into the subspace to generate the feature vectors
disp('feature_test created!');

X=zeros(num_eigenfaces,2);

for iFile = 1:testing_image_num-1;
    
    X(:,1)=feature_test(:,iFile);
    
    for iFile1 = 1:training_image_num-1;
        
        
        X(:,2)=feature_reference(:,iFile1);
        distances(iFile1,1) = pdist(X'); % Euclidian distance
        
    end
    
    minimum_distance=min(distances);
    [row,col]=find(distances==minimum_distance);
    matched_matrix(iFile,1)=row;
    
end   

% Selection of K value:

var = evalues / sum(evalues);
figure;
plot(cumsum(var));
xlabel('No. of eigenvectors'), ylabel('Percentage of variance considered (Normalized to 1)');
grid on;


% Visualization

% 1. Some matched images

% figure;
% for iFile = 1:20;
%     
%     vIm=reshape(test_matrix(:,iFile),[40 30 3]);
%     subplot(2,ceil(20),iFile);
%     showIm=vIm;
%     imshow(showIm);
%     %title(' Test Face ');
%     num=matched_matrix(iFile,1);
%     vIm1=reshape(train_matrix(:,num),[40 30 3]);
%     subplot(2,ceil(20),((20)+iFile));
%     showIm=vIm1;
%     imshow(showIm);
%     %title(' Matched Face ');
%     
% end
% 

% 2. Eigenfaces

% vIm = uint8(round(evectors(:,5)*255));
% vIm=reshape(vIm,[40 30 3]);
% showIm=vIm;
% imshow(showIm);

% Accuracy calculation:

% 1. Matched Subject Accuracy:

matched_suject_num=0;

for iFile = 1:testing_image_num-1;
    
    m=matched_matrix(iFile,1);
    a=train_subject_number(m,1);
    b=test_subject_number(iFile,1);
    
    if a==b;
        
        matched_suject_num=matched_suject_num+1;
        
    else
        continue
        
    end
end
        
Matched_Subject_Accuracy=matched_suject_num/(testing_image_num-1)*100;
    