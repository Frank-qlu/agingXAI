import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.nn as pyg_nn
import os
import numpy as np
import pandas as pd
from torchmetrics import MeanAbsoluteError,PearsonCorrCoef
from scipy.stats import pearsonr
import torch.nn.functional as F
import os
data_dir='/data/NK/'
def read_single_csv(input_path):
    df_chunk=pd.read_csv(input_path,sep=",",chunksize=3000)  #The hunksize parameter enables batch reads (this parameter is used to set how many rows of data are read into each batch)
    res_chunk=[]
    for chunk in df_chunk:
        res_chunk.append(chunk)
    res_df=pd.concat(res_chunk)
    return res_df

data_load = torch.load('./NK_pyg.pt')

Log_normalized_matrix_of_naive_cd4=read_single_csv(os.path.join(data_dir, f'expression_of_NK.csv'))
id=Log_normalized_matrix_of_naive_cd4.iloc[3487:,:].index
data_adj=pd.read_csv(os.path.join(data_dir, f'ppi_of_NK.csv'),sep=",")
in_dim=len(data_adj)
test_dataset = data_load[3487:] # test datasets
print(len(test_dataset))

class DeepRNAGenConv(nn.Module):
    
    def __init__(self, in_dim,node_features_dim=None, node_embedding_dim=None, num_layers=None, node_output_features_dim=None, convolution_dropout=0.1, dense_dropout=0.0):
        super(DeepRNAGenConv, self).__init__()
        
        self.node_encoder = torch.nn.Linear(node_features_dim, node_embedding_dim)
        
        self.gcn_layers = torch.nn.ModuleList()
        self.in_dim=in_dim
        self.hidden1 = nn.Linear(in_features=in_dim, out_features=1000)
        
        self.drop1 = nn.Dropout(0.0)
        self.drop2 = nn.Dropout(0.0)
        
        self.hidden2 = nn.Linear(1000, 100)

        for i in range(num_layers):
            convolution =  pyg_nn.GENConv(in_channels=node_embedding_dim, out_channels=node_embedding_dim)
            norm = torch.nn.LayerNorm(node_embedding_dim)
            activation = torch.nn.ReLU()
            layer = pyg_nn.DeepGCNLayer(conv=convolution, norm=norm, act=activation, dropout=convolution_dropout)
            self.gcn_layers.append(layer)

        self.dropout = torch.nn.Dropout(p=dense_dropout)
        self.decoder = torch.nn.Linear(node_embedding_dim, node_output_features_dim)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        xs=x.view(-1,self.in_dim)
        xs=F.leaky_relu( self.drop1(self.hidden1(xs)))
        xs=F.leaky_relu( self.drop2(self.hidden2(xs)))
        x = self.node_encoder(x)
        for layer in self.gcn_layers:
            x = layer(x, edge_index)
        x = self.dropout(x)
        x = pyg_nn.global_mean_pool(x, data.batch)
        x=0.5*xs+0.5*x
        x= self.decoder (x)
        return x[:, 0]
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
epochs = 20  
lr = 0.001  # LR
num_node_features = 1 
num_classes =1 
test_loader = torch_geometric.loader.DataListLoader(test_dataset, batch_size=64, shuffle=False)
net=DeepRNAGenConv(
    in_dim=in_dim,
    node_features_dim=1,
    node_embedding_dim=100,
    num_layers=2,
    node_output_features_dim=1,
    convolution_dropout=0.2,
    dense_dropout=0.2)
net.initialize()
model = pyg_nn.DataParallel(net).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
loss_function =  torch.nn.MSELoss(reduction='mean')
model.load_state_dict(torch.load('models/NK_best_model.pth')) #load the best model

mean_squared_error = MeanAbsoluteError().to(device)
pearson = PearsonCorrCoef().to(device)

def run():
 model.eval()
 with torch.no_grad():

    loss_test = 0 
    mae_test=0
    pearson_test=0
    count=0
    total_target=torch.zeros(1, 1).to(device)
    total_pred=torch.zeros(1, 1).to(device)
    for data in test_loader:
        pred = model(data)
        target = [data.y.float().unsqueeze(-1) for data in data]
        target = torch.cat((target)).to(pred.device)
        loss = loss_function(pred, target)
        loss_test += loss.item()
        mae=mean_squared_error(pred, target)
        mae_test += mae.item()
        count=count+1
        total_pred=torch.cat((total_pred,pred.unsqueeze(-1)),dim=0)
        total_target=torch.cat((total_target,target.unsqueeze(-1)),dim=0)
    pred_target=torch.cat((total_pred,total_target),dim=1)
    pred_target=pred_target.cpu().numpy()[1:,:]
    df_pred_target=pd.DataFrame(pred_target,index=id,columns=['pred','target'])
    df_pred_target.to_csv("NK_agePrediction.csv")
    print( df_pred_target.head(10),len(pred_target))
    print("Test PCC",pearsonr(df_pred_target['pred'],df_pred_target['target']))

    loss_test /= count
    mae_test/= count
    pearson_test/= count

    print(('Test  Loss: {:.4f}'.format(loss_test), 'Test MAE{:.4f}'.format(mae_test),'Test PCC',pearsonr(df_pred_target['pred'],df_pred_target['target'])))
    
    return loss_test, mae_test
if __name__ == '__main__':
   loss_test, mae_test=run()
   print(('RMSE_test: {:.4f} , mae_test {:.4f}'.format( np.sqrt(loss_test), mae_test)))