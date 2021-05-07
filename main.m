clc;clear;close all;
% This code predict epileptic seizures using an on-line neuro-fuzzy model
%
% Reference paper: Shokouh Alaei H, Khalilzadeh MA, Gorji A.
% "Optimal selection of SOP and SPH using fuzzy inference system for on-line
% epileptic seizure prediction based on EEG phase synchronization". 
% Australas Phys Eng Sci Med. 2019 Dec;42(4):1049-1068. 
%
% Written by: Hesam Shokouh Alaei
%
load X
load Y
load Z
% Suppose that phase synchronization features of a patients's EEG is
% extracted. In this example we earlier extracted 75 features from each
% time-window. Next, optimal features were selected by a statistical test.
% Obtained matrix called X along with their corresponding label
% (seizure, non-seizure) or Y.
% Z is all phase synchronization features in non-evaluative segment
% Note that non-evaluative segment contains 3 or 4 ictal and inter-ictal samples. 

sop = 10; 
sph = 30;
% Choose any SOP and SPH length in minutes

N=16;  %length of time-window (second)
M=3.2;   %length of overlap (second)

m=2;    % fuzziness of clusters
lambda=1;   % forgetting factor in RLS update rule
th1=0.2; % threshold for adding new clusters
th2=0.3; % threshold for merging the similar clusters

%Classification by evolve neuro fuzzy model (ENFM)
% Here, the ENFM predicts SOP minutes ahead SPH. The estimated output is Yhat 
Yhat = ENFM(X, Y, m, lambda, th1,th2, sph, sop, N, M);

%Postprocessing
% Controlling of false predictions by applying MA filter on ENFM output
window=15; % length of moving average filter
M = N-M;
delay = ceil(((window*60*256)-(N*256))/(M*256));  % number of frames
h = ceil(((120*60*256)-(N*256))/(M*256));  % number of frames
s = find(Y==1, 7, 'first')+h;
s = s(:,end);
nb=floor((length(Y)-s)/delay);
nb1=floor((length(Z)-s)/delay);
temp=zeros(1,delay+1);
for i=1:nb
    temp=Yhat(s+(i-1)*delay:s+i*delay);
    avg(1,i)=mean(temp);
end
output=-1.*ones(1,nb);
for i=1:(nb-8)
    if output(i)==-1
        if avg(i)<avg(i+1)&&avg(i+1)<avg(i+2)&&avg(i+2)<avg(i+3)&&avg(i+3)<avg(i+4)
            output(i+4)=1;
            output(i)=0;
            stack(i)=i;
        elseif avg(i)<avg(i+1)&&avg(i+1)>avg(i+2)&&avg(i+1)<avg(i+3)&&avg(i+3)>avg(i+4)&&avg(i+3)<avg(i+5)&&avg(i+5)>avg(i+6)&&avg(i+5)<avg(i+7)
            output(i+7)=1;
            output(i)=0;
            stack(i)=i;
        elseif avg(i)<avg(i+1)&&avg(i+1)<avg(i+2)&&avg(i+2)>avg(i+3)&&avg(i+2)<avg(i+4)&&avg(i+4)<avg(i+5)
            output(i+5)=1;
            output(i)=0;
            stack(i)=i;
        elseif avg(i)<avg(i+1)&&avg(i+1)<avg(i+2)&&avg(i+2)>avg(i+3)&&avg(i+2)<avg(i+4)&&avg(i+4)>avg(i+5)&&avg(i+4)<avg(i+6)
            output(i+6)=1;
            output(i)=0;
            stack(i)=i;
        elseif avg(i)<avg(i+1)&&avg(i+1)>avg(i+2)&&avg(i+2)<avg(i+3)&&avg(i+3)<avg(i+4)&&avg(i+4)<avg(i+5)
            output(i+5)=1;
            output(i)=0;
            stack(i)=i;
        elseif avg(i)<avg(i+1)&&avg(i+1)>avg(i+2)&&avg(i+1)<avg(i+3)&&avg(i+3)>avg(i+4)&&avg(i+3)<avg(i+5)&&avg(i+5)<avg(i+6)
            output(i+6)=1;
            output(i)=0;
            stack(i)=i;
        elseif avg(i)<avg(i+1)&&avg(i+1)<avg(i+2)&&avg(i+2)<avg(i+3)&&avg(i+3)>avg(i+4)&&avg(i+3)<avg(i+5)
            output(i+5)=1;
            output(i)=0;
        elseif avg(i)<avg(i+1)&&avg(i+1)>avg(i+2)&&avg(i+1)<avg(i+3)&&avg(i+3)<avg(i+4)&&avg(i+4)>avg(i+5)&&avg(i+4)<avg(i+6)
            output(i+6)=1;
            output(i)=0;
            stack(i)=i;
        else
            output(i)=0;
        end
    elseif output(i)==1&&output(i+1)==-1
        if avg(i+1)<avg(i)&&avg(i+2)>avg(i)
            output(i+2)=1;
            output(i+1)=1;
        elseif avg(i+1)>=avg(i)&&avg(i+1)<=avg(i+2)
            output(i+2)=1;
            output(i+1)=1;
        else
            output(i+2)=0;
            output(i+1)=1;
        end
    end
end

stack(stack==0)=[];
opt=stack*delay;
s1=length(opt);
for i=1:s1
    time_opt(i) = ((M*opt(i))+N)/3600;
end
Ytest=Y(s:nb*delay+s);
f=find(Ytest==1);
f=f(2:end);
Ytest(f)=0;
sph = ceil(((sph*60*256)-(N*256))/(M*256));  % number of frames
sop = ceil(((sop*60*256)-(N*256))/(M*256));  % number of frames
L=length(Ytest)-sph-sop;
for i=1:L        
   d(i) = any(Ytest(1,i+sph:i+sph+sop)>0);
end
Ytest(Ytest==-1)=0;
szr=find(Ytest==1);
nb2=ceil(szr/delay);
d=double(d);
nb=floor(length(d)/delay);
nb=nb-1;
frame=zeros(1,delay+1);
for i=1:nb
    frame(i,:)=d(1+(i-1)*delay:1+i*delay);
end
output = output(1:nb);
TN = 0;  %true negative
TP = 0;  %true positive
FP = 0;  %false positive
FN = 0;  %false negative
tn = zeros(1,nb);tp = zeros(1,nb);fn = zeros(1,nb);fp = zeros(1,nb);
for i=1:nb
    a=any(frame(i,:));
    if a==1 && output(1,i)==1 && i>nb1
        TP = TP +1;
        tp(1,i)=1;
    elseif a==1 && output(1,i)==1 && i<=nb1
        FP = FP +1;
        fp(1,i)=1;
    elseif a==1 && output(1,i)==0 && i>nb1 && i<nb2
        FN = FN +1;
        fn(1,i)=1;
     elseif a==1 && output(1,i)==0 && i>nb1 && i>=nb2
        TN = TN + 1;
        tn(1,i)=1;
    elseif a==1 && output(1,i)==0 && i<=nb1
        TN = TN + 1;
        tn(1,i)=1;
    elseif a==0 && output(1,i)==0
        TN = TN + 1;
        tn(1,i)=1;
    elseif a==0 && output(1,i)==1
        FP = FP +1;
        fp(1,i)=1;
    end
end

S1 = (TP/(TP+FN))*100;  %Sensitivity
disp(['Sensitivity: ',num2str(S1),'%'])
S2 = (TN/(TN+FP))*100;  %Specificity
disp(['Specificity: ',num2str(S2,2),'%'])
PPV = TP/(TP+FP)*100;   % Precision
disp(['Precision: ',num2str(PPV,2),'%'])
A = ((TP+TN)/(TN+FP+TP+FN))*100;    %Accuracy
disp(['Accuracy: ',num2str(A,2),'%'])
FPR = (FP/(TN+FP));   %False prediction rate per hour
disp(['FPR: ',num2str(FPR,2)])

time_szr = ((M*szr)+N)/3600;
time_test = ((M*L)+N)/3600;
t = linspace(0,time_test,nb);

% Graphical results of model output in evaluative segment
% Evaluative segment contains the last ictal and interictal samples
figure(1)
bar(t,tn,'b','FaceAlpha',0.5)
hold on
bar(t,tp,'g','FaceAlpha',0.5)
hold on
bar(t,fn,'y','FaceAlpha',0.5)
hold on
bar(t,fp,'r','FaceAlpha',0.5)
hold on 
stem(time_szr,Ytest(szr),'h','MarkerSize',15,'LineWidth',2,'LineStyle','--','Color','k','MarkerfaceColor','r','MarkerEdgeColor','k')
title('Result of the adaptive prediction system for patient 1','FontName','Arial','FontWeight','Bold','LineWidth',2,'FontSize',16)
xlabel('Time (hour)','FontName','Arial','FontWeight','Bold','LineWidth',2,'FontSize',12.5)
ylabel('Prediction Outcome','FontName','Arial','FontWeight','Bold','LineWidth',2,'FontSize',12.5)
legend('True Negative','True Positive','False Negative','False Positive','Seizure Onset')
ylim([-0.5, 1.5])
a=findobj(gcf);
allaxes=findall(a,'Type','axes');
set(allaxes,'FontName','Arial','FontWeight','Bold','LineWidth',2,...
'FontSize',12.5);