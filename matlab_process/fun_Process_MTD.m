function MTD_Signal = fun_Process_MTD( ProSignal,Len_PRT,Num_PRTperFrame )
%fun_Process_MTD MTD������
%   MTD��������ÿ�����뵥Ԫ�����ݽ��� FFT, abs, max ����
%  ���뺯����ProSignal              ���������� ������ʽ��Num_PRTperFrame*Len_PRT
%            Len_PRT                �ź��������ݳ���
%            Num_PRTperFrame        �ź���λ��۵���
%  ���������MTD_Signal_R           MTD����ά������ ʸ����ʽ��1*Len_PRT
%            MTD_Signal_V           MTD�ٶ�ά������  ʸ����ʽ��1*Num_PRTperFrame
%% MTD����
% ���ɴ����� - ���δ�
betaMTD = 8;
WindowData= kaiser(Num_PRTperFrame,betaMTD);
% WindowData = ones(Num_PRTperFrame,1);                                    %%���δ� �����Ӵ�
MTD_Signal = zeros(Num_PRTperFrame,Len_PRT);                               %%MTD������
MTD_Signal_R = zeros(1,Len_PRT);                                           %%MTD����ά������
% ÿ�����뵥Ԫ�����ݽ��� MTD ����
for Index=1:Len_PRT
    % �Ӵ�����
    Signal_Win = ProSignal(:,Index) .* WindowData; 
    % FFT����
    FFT_Signal = fftshift(fft(Signal_Win, Num_PRTperFrame));
    % ��ģֵ
    Abs_Signal = abs(FFT_Signal);
    % ���뵥Ԫѡ��
    [MTD_Signal_R(Index),~] = max(Abs_Signal);  
    MTD_Signal(:,Index) = Abs_Signal;  
end
% �ٶȵ�Ԫѡ��
MTD_Signal_V = max(MTD_Signal,[],2)';
end

