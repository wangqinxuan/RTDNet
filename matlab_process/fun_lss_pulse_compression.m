function s_PC=fun_lss_pulse_compression(Echo_simu_clutter,show_PC,pulse3)

[x,len_pulse3]=size(pulse3);%160
%����������ֿ�
[m,n]=size(Echo_simu_clutter);
prtNum=m;
point_prt=n;


s_PC=zeros(prtNum,point_prt);%����ѹ����Ľ��


%����ѹ��
for i_prt=1:prtNum
    signal_PC_3=fun_pulse_compression(pulse3,Echo_simu_clutter(i_prt,:));%������ѹ��
    s_PC(i_prt,:)=signal_PC_3(len_pulse3:end);%��������

    if show_PC==1
    figure(5)
    plot(20*log10(abs(s_PC(i_prt,:)))),title('������ѹ��ʾ');
    pause(0.05)
    end
end
end