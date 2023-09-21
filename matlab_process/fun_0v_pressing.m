%压制MTD_0速附近的波形
function MTD=fun_0v_pressing(MTD)%无噪回波幅度控制
[prtNum,point_prt]=size(MTD);
zero_v_pos=round(prtNum/2);
MTD(zero_v_pos-round(prtNum/200):zero_v_pos+round(prtNum/200),:)=mean(mean(abs(MTD)));

end