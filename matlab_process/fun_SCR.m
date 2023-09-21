

function Echo_simu= fun_SCR(prtNum,Echo_simu,echoData_Frame,SCR)

  
SCR_=10^((SCR)/10);

P_s=mean(Echo_simu(1,:).^2)+eps;
    for i=1:prtNum
        P_echo=mean(echoData_Frame(i,:).^2);
        P_s_need= P_echo*SCR_;%需要的信号功率
        g=P_s_need/P_s;
        Echo_simu(i,: )=Echo_simu(i,:)*sqrt(g);
    end    
       
end