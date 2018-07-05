close all, clear all, clc
y_all=xlsread('F:\3-project_time series data_students.xlsx');
%y_all=xlsread("D:\3-project_time series data_students.xlsx");
%F:\Courses\EEE 511- Artificial Neural Computation\Project 2\3-time series prediction project due Oct 6
y_train = y_all(1:246)';
y_test = y_all(246:275);
T_train  = 1:246;
T_test = (246:275);
subplot(2,1,1)
plot(y_all,'b');
hold on;
plot(y_train,'r');
%plot(y_test,'g');
legend('testing data','training data');
xlabel('Time');
ylabel('y values');
y_train=con2seq(y_train);

T=con2seq(y_train');
T_op =con2seq(y_all(226:end)');

rng(15);
b=(sqrt(6))/sqrt(10+2);
net.iw{1,1} = (-b + (2*b)*rand(10,0));

%net.trainParam.goal =1*10^-6;
%net1.b =[0,0]
net=narnet(1:24,10);
net.trainFcn='trainlm';
[Xs,Xi,Ai,Ts] = preparets(net,{},{},y_train);

net.divideParam.trainRatio = 75/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 10/100;

net = train(net,Xs,Ts,Xi,Ai);
view(net)


[netc,xic,aic] = closeloop(net,Xi,Ai);
[y2,xfc,afc] = netc(cell(0,30),xic,aic);
y2=cell2mat(y2);

ytest = y_test';
figure(2);
plot(T_test,y2,'b');
hold on;
plot(T_test,ytest,'r');
legend('prediction','actual output');


e = ytest - y2;
figure(3)
subplot(2,1,1);
xlabel('Timesteps in years');
ylabel('Value');
plot(246:275,y2,'g')
hold on;
plot(246:275,ytest,'r')
legend('Predicted data','Actual Data');
subplot(2,1,2)
xlabel('Timesteps in Years')
ylabel('Error')
plot(246:275,e,'b');
legend('error');


y_train=cell2mat(y_train);
figure(4)
plot(y_train,'g');
hold on;
plot(246:275,y2,'b');
plot(246:275,ytest,'r');
legend('Training Data','Predicted Data','Actual Data');
xlabel('Timesteps in years');
ylabel('Value');




