import torch
import torch.utils.data.dataset as Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import math

#创建Dataset子类
class subDataset(Dataset.Dataset):
    #初始化，定义数据内容和标签
    def __init__(self, Data, Label):
        self.Data = Data
        self.Label = Label
        
    #返回数据集大小
    def __len__(self):
        return len(self.Data)
    #得到数据内容和标签
    def __getitem__(self, index):
        data = torch.Tensor(self.Data[index])
        label= torch.Tensor(self.Label[index])
        
        return  data, label

def loss_class_pos(confmap_preds,confmaps,threshold,criterion):
    record=[[],[],[],[]]
    record=ClassPosRecord(record,confmap_preds,confmaps,threshold)
    record_class_preds,record_pos_preds,record_class,record_pos=record[0],record[1],record[2],record[3]
    record_class_preds=torch.Tensor(np.array(record_class_preds))
    record_pos_preds=torch.Tensor(np.array(record_pos_preds).astype('float32'))
    record_class=torch.Tensor(np.array(record_class))
    record_pos=torch.Tensor(np.array(record_pos).astype('float32'))
    loss_class=criterion(record_class_preds.float().cuda(), record_class.float().cuda())  # MSELoss
    loss_pos=criterion(record_pos_preds.float().cuda(), record_pos.float().cuda())  # MSELoss
    return loss_class,loss_pos

#预测类别，预测位置记录
def ClassPosRecord(record,confmap_preds,confmaps,threshold):
    confmaps=confmaps.numpy()
    confmap_preds=confmap_preds.cpu().detach().numpy()
    record_class_preds,record_pos_preds,record_class,record_pos=record[0],record[1],record[2],record[3]
        
    for i in range(confmap_preds.shape[0]):
        for j in range(confmap_preds.shape[2]):
            if(np.max(confmap_preds[i,0,j,:,:])>threshold):
                record_class_preds.append(1)
                pos=np.where(confmap_preds[i,0,j,:,:]==np.max(confmap_preds[i,0,j,:,:]))
                pos= np.array([*pos])[:,0]#(2,1)
                record_pos_preds.append(pos)
            else:
                record_class_preds.append(0)
                record_pos_preds.append(np.array([0,0]))
            
            if(np.max(confmaps[i,0,j,:,:])>threshold):
                record_class.append(1)
                pos=np.where(confmaps[i,0,j,:,:]==np.max(confmaps[i,0,j,:,:]))
                pos= np.array([*pos])[:,0]#(2,1)
                record_pos.append(pos)
            else:
                record_class.append(0)
                record_pos.append(np.array([0,0]))
    record=[record_class_preds,record_pos_preds,record_class,record_pos]
    return record

#分类正确率
def ClassAccuracy(record_class_preds,record_class):
    record_class_preds=np.array(record_class_preds)
    record_class=np.array(record_class)
    acc=sum(np.equal(record_class_preds,record_class))/len(record_class_preds)
    return  acc

#虚警率
def FaRate(record_class_preds,record_class):
    fa_cnt,no_fa_cnt=0,0
    for i in range(len(record_class_preds)):
        if((record_class_preds[i]==1)&(record_class[i]==0)):
            fa_cnt=fa_cnt+1
        if((record_class_preds[i]==0)&(record_class[i]==0)):
            no_fa_cnt=no_fa_cnt+1
    if((fa_cnt+no_fa_cnt)!=0):
        fa=fa_cnt/(fa_cnt+no_fa_cnt)
    else:
        fa=1
        
    return fa

#检测率
def DetectionRate(record_class_preds,record_class):
    dr_cnt,no_dr_cnt=0,0
    for i in range(len(record_class_preds)):
        if((record_class_preds[i]==1)&(record_class[i]==1)):
            dr_cnt=dr_cnt+1
        if((record_class_preds[i]==0)&(record_class[i]==1)):
            no_dr_cnt=no_dr_cnt+1
    if((dr_cnt+no_dr_cnt)!=0):
        dr=dr_cnt/(dr_cnt+no_dr_cnt)
    else:
        dr=0
    return dr

#定位误差
def PosError(record,confmaps_test,R_range,V_target_max):
    #在检测有目标正确的基础上，即p(1|1),来进行定位精度的计算。
    record_class_preds,record_pos_preds,record_class,record_pos=record[0],record[1],record[2],record[3]
    index_p_1_1=[]#所有分类p(1|1)的标号
    pos_preds=[]
    pos=[]
    for i in range(len(record_class_preds)):
        if((record_class_preds[i]==1)&(record_class[i]==1)):
            index_p_1_1.append(i)
    for index in index_p_1_1:
        pos_preds.append(record_pos_preds[index])#只取出分类p(1|1)的事件
        pos.append(record_pos[index])#只取出分类p(1|1)的事件

    pos_preds=np.array(pos_preds)
    pos=np.array(pos)
    if(pos_preds.shape[0]==0):    
        pos_err=[0.5,6]
    else:
        pos_err=sum(abs(pos-pos_preds))/pos_preds.shape[0]
        pos_err[0]=pos_err[0]*(V_target_max*2)/confmaps_test.shape[3]
        pos_err[1]=pos_err[1]*(R_range[1]-R_range[0])/confmaps_test.shape[4]

    return pos_err

def RDI(drate,fa,pos_err,dv_base=5,dr_base=50,w=[0.4,0.3,0.3]):
    '''dr_base,dv_base是允许速度与距离偏移的基准值'''
    dv_base=dv_base*50
    dv,dr=pos_err[0]*50,pos_err[1]
    l=dv**2+dr**2
    l_base=dv_base**2+dr_base**2
    if(l<l_base):
        pof=1-l/l_base
    if(l>l_base):
        pof=math.exp(1-l/l_base)-1
    
    rdi=w[0]*drate+w[1]*(1-fa)+w[2]*pof
    return rdi,pof

def ScaleSum(loss_conf,loss2,loss3):
    x=loss2/loss_conf
    
    for i in [1,1e1,1e2,1e3,1e4,1e5,1e6,1e7]:
        if(1<=int(x/i)<=10):
            break
    loss2=loss2/(i*10)
    x=loss3/loss_conf

    for i in [1,1e1,1e2,1e3,1e4,1e5,1e6,1e7]:
        if(1<=int(x/i)<=10):
            break
    loss3=loss3/(i*10)
    
    loss=loss_conf+loss2+loss3
    return loss

def TestConfmap(test_data,test_confmaps,criterion,model,threshold,R_range,V_target_max):
    dataset = subDataset(test_data, test_confmaps) 
    dataloader_test = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    test_loss,test_acc,record=0,0,[[],[],[],[]]
    for batch, dataset_batch in enumerate(tqdm(dataloader_test)):
        data_test,confmaps_test=dataset_batch
        confmap_preds = model(data_test.float().cuda())
        #confmap_preds=[batch_size, class, win_size, h, w]     
        record=ClassPosRecord(record,confmap_preds,confmaps_test,threshold)
        loss_confmap = criterion(confmap_preds, confmaps_test.float().cuda())  # MSELoss
        loss_class,loss_pos= loss_class_pos(confmap_preds,confmaps_test,threshold,criterion)
        loss=ScaleSum(loss_confmap,loss_class,loss_pos)
        
        test_loss = loss.item()+test_loss

    test_loss=test_loss/(batch+1)
    record_class_preds,record_pos_preds,record_class,record_pos=record[0],record[1],record[2],record[3]
    test_acc=ClassAccuracy(record_class_preds,record_class)
    fa=FaRate(record_class_preds,record_class)
    dr=DetectionRate(record_class_preds,record_class)
    pos_err=PosError(record,confmaps_test,R_range,V_target_max)
    return test_loss,test_acc,fa,dr,pos_err