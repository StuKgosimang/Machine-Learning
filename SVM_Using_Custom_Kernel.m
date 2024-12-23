rng(1); % for reproducibility
n=100; %number of class per quadrant

r1=sqrt(rand(2*n,1)); %radial distances
t1=[pi/2*rand(n,1);pi/2*rand(n,1)+pi]; %angles for q1 and q3
x1=[r1.*cos(t1),r1.*sin(t1)]; %polar to cartesian

r2=sqrt(rand(2*n,1)); 
t2 = [pi/2*rand(n,1)+pi/2;pi/2*rand(n,1)-pi/2];
x2=[r2.*cos(t2),r2.*sin(t2)];

%create a vector for classification
X=[x1;x2];
Y=ones(4*n,1);
Y(2*n+1:end)=-1;

%plot the data
figure
gscatter(X(:,1),X(:,2),Y);
title('scatter diagram for simulated data')

%Model
mdl1=fitcsvm(X,Y,'KernelFunction','rbf','Standardize',true);

%compute the scores over the grid
d=0.02;
[x1Grid, x2Grid]=meshgrid(min(X(:,1)):d:max(X(:,1)),min(X(:,2)):d:max(X(:,2)));
xGrid=[x1Grid(:),x2Grid(:)];
[~,scores1]=predict(mdl1,xGrid);

%Plot the data, SVs and Decision Boundary
figure
h(1:2) = gscatter(X(:,1),X(:,2),Y);
hold on
h(3) = plot(X(mdl1.IsSupportVector,1),X(mdl1.IsSupportVector,2),'ko','MarkerSize',15);
contour(x1Grid, x2Grid,reshape(scores1(:,2),size(x1Grid)),[0 0],'k'); %boundary
title('Scatter Diagram with Decision Boundary')
legend({'-1','1','SV'},'location','best');
hold off

%Misclassification
CVmdl1 = crossval(mdl1);
misclass1 = kfoldLoss(CVmdl1);
misclass1;


