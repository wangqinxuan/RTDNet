%ѹ��MTD_0�ٸ����Ĳ���
function MTD=fun_0v_pressing(MTD)%����ز����ȿ���
[prtNum,point_prt]=size(MTD);
zero_v_pos=round(prtNum/2);
MTD(zero_v_pos-round(prtNum/200):zero_v_pos+round(prtNum/200),:)=mean(mean(abs(MTD)));

end