function [MTD_Signal_simu]=fun_MTD_produce(V,R,SCR)



show_PC=0;          %脉冲压缩结果显示
prtNum = 1536;       %每帧信号的脉冲数
graph=0;%画图开关
%% 常数定义
cj = sqrt(-1);
c  =  2.99792458e8;       % 电磁波传播速度

PI2 = 2*pi;
MHz = 1e+6;     % frequency unit(MHz)
us  = 1e-6;     % time unit(us)
ns  = 1e-9;     % time unit(ns)
KHz = 1e+3;     % frequency unit(KHz)
GHz = 1e+9;     % frequency unit(GHz)

%% 系统参数
fs = 25*MHz;                % 生成原始信号的采样频率
ts = 1/fs;
deltaR = c*ts/2;
point_prt=1000;
tao3  = 6.4*us;             % 脉冲3脉宽
f0=0*MHz;
fc=5500*MHz;
prt   = 64.88*us;
prf   = 1/prt;
wavelength=c/fc;
B     = 10*MHz;             % 带宽
K3    = -B/tao3;            % 长脉冲调频斜率
t3=-tao3/2:ts:tao3/2-ts; 
%% 生成发射信号
%画出chirp信号时域波形
pulse3=exp(cj*2*pi*(f0*t3+0.5*K3*(t3.^2)));%160个点


%% 仿真模拟某一距离处的运动目标回波
%高斯噪声
% echoData_Frame=awgn(zeros(prtNum,point_prt),SCR);
%真实杂波
echoData_Frame=load(['./matlab_dataset/BasebandRawData_mat/frame_',num2str(randi([0,10])),'.mat']);
echoData_Frame=echoData_Frame.echoData_Frame_0(:,1031-point_prt:1031);
if((V==0)||(R==0))
    Echo_simu_clutter=echoData_Frame;%真实杂波
    %如果速度太小或距离太近就不仿真了，当作背景杂波
else
    Velocity=V*(-1);
    Range=R;
    Echo_simu=fun_SimulateTarget(Velocity,Range,prtNum,point_prt,pulse3);
    Echo_simu=gather(Echo_simu);
    Echo_simu=fun_SCR(prtNum,Echo_simu,echoData_Frame,SCR);
    %加杂波
    Echo_simu_clutter= fun_add_clutter(Echo_simu,echoData_Frame);%
end

Echo_simu_clutter=gather(Echo_simu_clutter);
%% 脉冲压缩
[Echo_simu_clutter]=fun_lss_pulse_compression(Echo_simu_clutter,show_PC,pulse3);

%% MTD
[m,n]=size(Echo_simu_clutter);
MTD_Signal_simu = fun_Process_MTD( Echo_simu_clutter,n,m );
MTD_Signal_simu=fun_0v_pressing(MTD_Signal_simu);%压制0速附近的峰值





%% 画出MTD三维图
if(graph==1)
    figure(8);
    R_point=6;%两点间距6m
    r0=0:R_point:point_prt*R_point-R_point;%距离轴
    fd=linspace(-prf/2,prf/2,prtNum);
    v0=fd*wavelength/2;%速度轴
    MTD_Signal_simu_log= 20*log10(abs(MTD_Signal_simu)/max(max(abs(MTD_Signal_simu))));
    mesh(r0,v0,MTD_Signal_simu_log);xlabel('距离');ylabel('速度m/s');zlabel('幅度dB');title('0波束仿真信号MTD');

    % 画出速度维
    MTD_max=max(max(MTD_Signal_simu_log));
    [vindex,rindex]=find(MTD_Signal_simu_log(:,:)==MTD_max);
    vindex,rindex
    figure(9);
    rindex=417;
    plot(v0,(MTD_Signal_simu_log(:,rindex))),xlabel('速度m/s'),ylabel('幅度dB');title('速度维');
    figure(10);
    plot(v0,(MTD_Signal_simu(:,rindex))),xlabel('速度m/s'),ylabel('幅度');title('速度维');
    
%     画出距离维
    figure(11);
    vindex=812;
    plot(r0,(MTD_Signal_simu_log(vindex,:))),xlabel('距离'),ylabel('幅度dB');title('距离维');
    figure(12);
    plot(r0,(MTD_Signal_simu(vindex,:))),xlabel('距离'),ylabel('幅度');title('距离维');
    
end

end