%for repeatbility
rng(1);
r = sqrt(rand(100,1)); % radius
t = 2*pi*rand(100,1); % angle
d1=r.*cos(t);
d2 = r.*sin(t);
data1 = [d1,d2];

%Generate 100 points distributed in the annulus
r2 = sqrt(rand(100,1)+1);
t2 = 2*pi*rand(100,1);
d3=r2.*cos(t2);
d4 = r2.*sin(t2);
data2 = [d3,d4];

%Plot the points and circles of radii 1 and 2 for comparison
figure
plot(data1(:,1),data1(:,2),'r.','MarkerSize',15);
hold on
plot(data2(:,1),data2(:,2),'b.','MarkerSize',15);
fplot(@(t)sin(t),@(t)cos(t));
fplot(@(t)2*sin(t),@(t)2*cos(t));
axis equal
hold off

%Create a vector of classifications
data3 = [data1; data2]; %combine data
class=ones(200,1);
class(1:100)=-1;

%train the SVM Model
cl = fitcsvm(data3,class,'KernelFunction','rbf','BoxConstraint',Inf,'ClassNames',[-1,1]);

%predict scores of over the grid
d=0.02;
[x1Grid,x2Grid]=meshgrid(min(data3(:,1)):d:max(data3(:,1)),min(data3(:,2)):d:max(data3(:,2)));
xGrid=[x1Grid(:),x2Grid(:)];
[~,scores] = predict(cl,xGrid);

%plot the data and the decision boundary
figure
h(1:2) = gscatter(data3(:,1),data3(:,2),class,'rb','.');
hold on
fplot(@(t)sin(t),@(t)cos(t));
h(3)=plot(data3(cl.IsSupportVector,1),data3(cl.IsSupportVector,2),'ko');
contour(x1Grid,x2Grid,reshape(scores(:,2),size(x1Grid)),[0 0],'k');
legends(h,['-1','+1','Support Vectors']);
axis equal
hold off







