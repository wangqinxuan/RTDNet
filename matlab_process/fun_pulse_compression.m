function signal_PC=fun_pulse_compression(s0,s_echo)

%ƥ���˲�
h=conj(s0(1,end:-1:1));         %ƥ���˲����ĵĳ弤��Ӧ����
[m,n]=size(h);
point_pulse=n;

%�Ӵ�
hamm=hamming(point_pulse)';%���ɺ�����
h=h.*hamm;
% kaisa=kaiser(point_pulse,20)';
% h=h.*kaisa;

[m,n]=size(s_echo);
point_prt=n;
point_signal_PC=point_pulse+point_prt-1;            %�˲��൱�ڻز��ź���ƥ���˲������߾����YNΪ�����ĵ���

%����FFTʵ�ֿ��پ��
S=fft(s_echo,point_signal_PC);                     %�ֱ����s(t)��h(t)��Ƶ��
H=fft(h,point_signal_PC);
Y=S.*H;                          %Ƶ����˼�Ϊʱ����
y=ifft(Y,point_signal_PC);                    %��Ƶ��˻�������Ҷ���任����ʱ���źż�Ϊƥ���˲������ú�����
signal_PC=y;
end




