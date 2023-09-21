function Echo_simu= fun_SimulateTarget( Velocity,Range,prtNum,point_prt,pulse3)
%RangeΪ��ʼ����
fc=5500e6;
prt=64.88e-6;
c=2.99792458e8;

[x,len_pulse3]=size(pulse3);%160
pulse3=gpuArray(pulse3);
%% ��������ز�

Echo_simu=zeros(prtNum,point_prt);
Echo_simu=gpuArray(Echo_simu);


for i = 1:prtNum
DelayTime=2*(Range+i*Velocity*prt)/c;%�ӳ�ʱ��
RealTime_Range=Range+i*Velocity*prt;
point_delay=round(RealTime_Range/6);
%����ƽ��
Echo_simu(i,point_delay:point_delay+len_pulse3-1) = pulse3.*exp(-1i*2*pi*fc*DelayTime);
end

end