function [MTD_Signal_simu]=fun_MTD_produce(V,R,SCR)



show_PC=0;          %����ѹ�������ʾ
prtNum = 1536;       %ÿ֡�źŵ�������
graph=0;%��ͼ����
%% ��������
cj = sqrt(-1);
c  =  2.99792458e8;       % ��Ų������ٶ�

PI2 = 2*pi;
MHz = 1e+6;     % frequency unit(MHz)
us  = 1e-6;     % time unit(us)
ns  = 1e-9;     % time unit(ns)
KHz = 1e+3;     % frequency unit(KHz)
GHz = 1e+9;     % frequency unit(GHz)

%% ϵͳ����
fs = 25*MHz;                % ����ԭʼ�źŵĲ���Ƶ��
ts = 1/fs;
deltaR = c*ts/2;
point_prt=1000;
tao3  = 6.4*us;             % ����3����
f0=0*MHz;
fc=5500*MHz;
prt   = 64.88*us;
prf   = 1/prt;
wavelength=c/fc;
B     = 10*MHz;             % ����
K3    = -B/tao3;            % �������Ƶб��
t3=-tao3/2:ts:tao3/2-ts; 
%% ���ɷ����ź�
%����chirp�ź�ʱ����
pulse3=exp(cj*2*pi*(f0*t3+0.5*K3*(t3.^2)));%160����


%% ����ģ��ĳһ���봦���˶�Ŀ��ز�
%��˹����
% echoData_Frame=awgn(zeros(prtNum,point_prt),SCR);
%��ʵ�Ӳ�
echoData_Frame=load(['./matlab_dataset/BasebandRawData_mat/frame_',num2str(randi([0,10])),'.mat']);
echoData_Frame=echoData_Frame.echoData_Frame_0(:,1031-point_prt:1031);
if((V==0)||(R==0))
    Echo_simu_clutter=echoData_Frame;%��ʵ�Ӳ�
    %����ٶ�̫С�����̫���Ͳ������ˣ����������Ӳ�
else
    Velocity=V*(-1);
    Range=R;
    Echo_simu=fun_SimulateTarget(Velocity,Range,prtNum,point_prt,pulse3);
    Echo_simu=gather(Echo_simu);
    Echo_simu=fun_SCR(prtNum,Echo_simu,echoData_Frame,SCR);
    %���Ӳ�
    Echo_simu_clutter= fun_add_clutter(Echo_simu,echoData_Frame);%
end

Echo_simu_clutter=gather(Echo_simu_clutter);
%% ����ѹ��
[Echo_simu_clutter]=fun_lss_pulse_compression(Echo_simu_clutter,show_PC,pulse3);

%% MTD
[m,n]=size(Echo_simu_clutter);
MTD_Signal_simu = fun_Process_MTD( Echo_simu_clutter,n,m );
MTD_Signal_simu=fun_0v_pressing(MTD_Signal_simu);%ѹ��0�ٸ����ķ�ֵ





%% ����MTD��άͼ
if(graph==1)
    figure(8);
    R_point=6;%������6m
    r0=0:R_point:point_prt*R_point-R_point;%������
    fd=linspace(-prf/2,prf/2,prtNum);
    v0=fd*wavelength/2;%�ٶ���
    MTD_Signal_simu_log= 20*log10(abs(MTD_Signal_simu)/max(max(abs(MTD_Signal_simu))));
    mesh(r0,v0,MTD_Signal_simu_log);xlabel('����');ylabel('�ٶ�m/s');zlabel('����dB');title('0���������ź�MTD');

    % �����ٶ�ά
    MTD_max=max(max(MTD_Signal_simu_log));
    [vindex,rindex]=find(MTD_Signal_simu_log(:,:)==MTD_max);
    vindex,rindex
    figure(9);
    rindex=417;
    plot(v0,(MTD_Signal_simu_log(:,rindex))),xlabel('�ٶ�m/s'),ylabel('����dB');title('�ٶ�ά');
    figure(10);
    plot(v0,(MTD_Signal_simu(:,rindex))),xlabel('�ٶ�m/s'),ylabel('����');title('�ٶ�ά');
    
%     ��������ά
    figure(11);
    vindex=812;
    plot(r0,(MTD_Signal_simu_log(vindex,:))),xlabel('����'),ylabel('����dB');title('����ά');
    figure(12);
    plot(r0,(MTD_Signal_simu(vindex,:))),xlabel('����'),ylabel('����');title('����ά');
    
end

end