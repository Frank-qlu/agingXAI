import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.nn as pyg_nn
import os
import pandas as pd
from scipy.stats import pearsonr
from torchmetrics import MeanAbsoluteError,PearsonCorrCoef
import logging
import shutil

data_dir='/data/NK/'

## setting logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logger.addHandler(logging.FileHandler('NK.log', 'a'))

print = logger.info

data_load = torch.load('./NK_pyg.pt')
print((len(data_load)))

data_adj=pd.read_csv(os.path.join(data_dir, f'ppi_of_NK.csv'),sep=",")
in_dim=len(data_adj)

train_dataset = data_load[0:3487] # train data
test_dataset = data_load[3487:] # test data
print((len(train_dataset),len(test_dataset)))
class DeepRNAGenConv(nn.Module):
    
    def __init__(self,in_dim, node_features_dim=None, node_embedding_dim=None, num_layers=None, node_output_features_dim=None, convolution_dropout=0.1, dense_dropout=0.0):
        super(DeepRNAGenConv, self).__init__()
        
        self.node_encoder = torch.nn.Linear(node_features_dim, node_embedding_dim)
        
        self.gcn_layers = torch.nn.ModuleList()
        self.in_dim=in_dim
        self.hidden1 = nn.Linear(in_features=in_dim, out_features=1000, bias=True)
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
        xs= F.leaky_relu(self.hidden1(xs))
        xs = F.leaky_relu(self.hidden2(xs))
        x = self.node_encoder(x)
        for layer in self.gcn_layers:
            x = layer(x, edge_index)
        x = self.dropout(x)
        x = pyg_nn.global_mean_pool(x, data.batch)
        x=0.5*xs+0.5*x
        x = self.decoder(x)
        return x[:, 0]



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
epochs = 20  
lr = 0.001  
num_node_features = 1 
train_loader = torch_geometric.loader.DataListLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch_geometric.loader.DataListLoader(test_dataset, batch_size=64, shuffle=False)
model = pyg_nn.DataParallel( DeepRNAGenConv(
    in_dim=in_dim,
    node_features_dim=1,
    node_embedding_dim=100,
    num_layers=2,
    node_output_features_dim=1,
    convolution_dropout=0.2,
    dense_dropout=0.2)).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=0.7, patience=3,
                                                       min_lr=0.0001)
loss_function =  torch.nn.MSELoss(reduction='mean')

mean_absolute_error = MeanAbsoluteError().to(device)
pearson = PearsonCorrCoef().to(device)

folder_path="./models"

if os.path.exists(folder_path):
    shutil.rmtree(folder_path)
    os.mkdir(folder_path)
else:
    os.mkdir(folder_path)
def run():
  for epoch in range(epochs):
    lr = scheduler.optimizer.param_groups[0]['lr']
    model.train()
    loss_train = 0  
    mae_total=0
    pearson_total=0
    count=0
    for data in train_loader:
        optimizer.zero_grad()
        pred = model(data)
        target = [data.y.float().unsqueeze(-1) for data in data]
        target = torch.cat((target)).to(pred.device)
        loss = loss_function(pred, target)
        loss_train += loss.item()
        mae=mean_absolute_error(pred, target)
        result=pearson(pred, target )
        mae_total+=mae.item()
        pearson_total+=result.item()

        loss.backward()
        optimizer.step()
        count=count+1
    loss_train /= count
    mae_total /= count
    pearson_total /= count
    print(("【EPOCH: 】%s" % str(epoch)))
    print(('Train  Loss',loss_train, 'Train MAE: ',mae_total,'Train PCC: ',pearson_total))
    test(epoch)

  print(('【Finished Training!】'))

best_test_loss=0
def test(i):
  global best_test_loss
  model.eval()
  with torch.no_grad():
    loss_train = 0
    mae_train=0
    pearson_train=0
    count=0
    for data in train_loader:
        pred = model(data)
        target = [data.y.float().unsqueeze(-1) for data in data]
        target = torch.cat((target)).to(pred.device)
        loss = loss_function(pred, target)
        loss_train += loss.item()
        mae=mean_absolute_error(pred, target)
        mae_train += mae.item()
        result=pearson(pred, target )
        pearson_train += result.item()
        count=count+1

    mae_train/= count
    pearson_train/= count
    loss_train /= count

    print(('Val Loss: {:.4f}'.format(loss_train),'Val MAE: {:.4f}'.format(mae_train),' Val PCC: ',pearson_train))


    loss_test = 0  
    mae_test=0
    count=0
    total_target=torch.zeros(1, 1).to(device)
    total_pred=torch.zeros(1, 1).to(device)
    for data in test_loader:
        pred = model(data)
        target = [data.y.float().unsqueeze(-1) for data in data]
        target = torch.cat((target)).to(pred.device)

        loss = loss_function(pred, target)
        loss_test += loss.item()
        mae=mean_absolute_error(pred, target)
        mae_test += mae.item()
        count=count+1
        total_pred=torch.cat((total_pred,pred.unsqueeze(-1)),dim=0)
        total_target=torch.cat((total_target,target.unsqueeze(-1)),dim=0)
    pred_target=torch.cat((total_pred,total_target),dim=1)
    pred_target=pred_target.cpu().numpy()[1:,:]
    df_pred_target=pd.DataFrame(pred_target,columns=['pred','target'])
    loss_test /= count
    mae_test/= count
    scheduler.step(mae_test)
    print(('Test  Loss: {:.4f}'.format(loss_test), 'Test MAE:{:.4f}'.format(mae_test),'Test pcc',pearsonr(df_pred_target['pred'],df_pred_target['target'])))
    if i==0:
        best_test_loss=loss_test 
    if best_test_loss>loss_test:
        best_test_loss=loss_test 
        torch.save(model.state_dict(), folder_path+'/'+'NK_best_model.pth')
if __name__ == '__main__':
   avg_mse = []
   avg_mae=[]
   run()