function s_PC=fun_lss_pulse_compression(Echo_simu_clutter,show_PC,pulse3)

[x,len_pulse3]=size(pulse3);%160
%将三个脉冲分开
[m,n]=size(Echo_simu_clutter);
prtNum=m;
point_prt=n;


s_PC=zeros(prtNum,point_prt);%脉冲压缩后的结果


%脉冲压缩
for i_prt=1:prtNum
    signal_PC_3=fun_pulse_compression(pulse3,Echo_simu_clutter(i_prt,:));%长脉冲压缩
    s_PC(i_prt,:)=signal_PC_3(len_pulse3:end);%向量对齐

    if show_PC==1
    figure(5)
    plot(20*log10(abs(s_PC(i_prt,:)))),title('波束脉压显示');
    pause(0.05)
    end
end
end