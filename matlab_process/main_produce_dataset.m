clc;clear; close all;

%% 生成距离和速度数组

n_t=400;
r_range=[1100,2900];%可调节
v_range=[-20,20];%可调节
v_list=[linspace(-16,-3,10),linspace(3,16,10),0,0,0,0];%可调节
r_start_list=[1500,2000,2500,1300,2200,2800];%可调节
dt=64.88*1e-6*1536;%一帧的时间间隔，prt=64.88us,脉冲积累数=1536
t=linspace(0,dt*n_t,n_t);
r_label=[];
v_label=[];
for i=1:12000/n_t
    v=zeros(1,n_t);
    v(1:n_t/2)=v_list(randi([1,length(v_list)],[1,1]));
    v(n_t/2:n_t)=v_list(randi([1,length(v_list)],[1,1]));
    r_start=r_start_list(randi([1,length(r_start_list)],[1,1]));
    r=r_start+v.*t;
    r_label=[r_label,r];
    v_label=[v_label,v];
end

for i=1:12000
    if(r_label(i)>r_range(2)-100 | r_label(i)<r_range(1)+100)
        r_label(i)=0;v_label(i)=0;
    end
    
    
    if(v_label(i)==0)
        r_label(i)=0;v_label(i)=0;
    end
end

R=r_label;
V=v_label;
% save(['label.mat'],'R','V');
%% Generate MTD per frame
tic;SCR=-60;%dB
load('label.mat');
mkdir(['./matlab_dataset/RDmaps_SCR',num2str(SCR),'dB/']);
for frameRInd=0:length(R)
V(frameRInd+1)
R(frameRInd+1)
[MTD_simu]=fun_MTD_produce(V(frameRInd+1),R(frameRInd+1),SCR);
MTD_simu=MTD_simu(696:842,166:500);%只取-20-20m/s,1000-3000m,减小计算量

save(['./matlab_dataset/RDmaps_SCR',num2str(SCR),'dB/frame_',num2str(frameRInd),'.mat'],'MTD_simu');
frameRInd
toc
end





