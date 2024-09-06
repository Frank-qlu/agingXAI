import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.nn as pyg_nn
import tqdm
import os
import pandas as pd
from torchmetrics import MeanAbsoluteError,PearsonCorrCoef
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.loader import DataLoader
import torch_geometric.nn as pyg_nn
from explain import Explainer, PGExplainer
import torch.nn.functional as F
from tqdm import tqdm
import os

data_dir='/data/NK/'
data_adj=pd.read_csv(os.path.join(data_dir, f'ppi_of_NK.csv'),sep=",")
in_dim=len(data_adj)

data_load = torch.load('./NK_pyg.pt')

test_dataset = data_load[0:3487] 
print((len(test_dataset)))

class DeepRNAGenConv(nn.Module):
    
    def __init__(self,in_dim, node_features_dim=None, node_embedding_dim=None, num_layers=None, node_output_features_dim=None, convolution_dropout=0.1, dense_dropout=0.0):
        super(DeepRNAGenConv, self).__init__()
        
        self.node_encoder = torch.nn.Linear(node_features_dim, node_embedding_dim)
        self.in_dim=in_dim
        self.gcn_layers = torch.nn.ModuleList()
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
    
    def forward(self, x, edge_index, batch):
        xs=x.view(-1,self.in_dim)
        xs= F.leaky_relu(self.hidden1(xs))
        xs = F.leaky_relu(self.hidden2(xs))
        x = self.node_encoder(x)
        for layer in self.gcn_layers:
            x = layer(x, edge_index)
        x = self.dropout(x)
        x = pyg_nn.global_mean_pool(x,batch)
        x=0.5*xs+0.5*x
        x = self.decoder(x)
        return x[:, 0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 50
num_node_features = 1 
test_loader = torch_geometric.loader.DataLoader(test_dataset, batch_size=64, shuffle=False)
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
loss_function =  torch.nn.MSELoss(reduction='mean')# loss function
model.load_state_dict(torch.load('models/NK_best_model.pth', map_location=device))
model.to(device)
if isinstance(model, pyg_nn.DataParallel):
    model = model.module


mean_squared_error = MeanAbsoluteError().to(device)
pearson = PearsonCorrCoef().to(device)

def run():
 lr = scheduler.optimizer.param_groups[0]['lr']
 model.eval()
 with torch.no_grad():

    loss_test = 0
    mae_test=0
    pearson_test=0
    count=0
    print(model)
    for data in tqdm(test_loader):
        pred = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device))
        target=data.y.float().to(pred.device)
        loss = loss_function(pred, target)
        loss_test += loss.item()
        mae=mean_squared_error(pred, target )
        mae_test += mae.item()
        result=pearson(pred, target )
        pearson_test += result.item()
        count=count+1
    
    loss_test /= count
    mae_test/= count
    scheduler.step(mae_test)
    pearson_test/= count
    print(('Test  Loss: {:.4f}'.format(loss_test), 'Test MAE:{:.4f}'.format(mae_test),'Test PCC:{:.4f}'.format(pearson_test)))

explainer = Explainer(
    model=model,
    algorithm=PGExplainer(epochs=5, lr=0.003).to(device),
    explanation_type='phenomenon',
    edge_mask_type='object',
    model_config=dict(
        mode='regression',
        task_level='graph',
        return_type='raw',
    ),
    threshold_config=dict(threshold_type='topk', value=100),
)
total_list=[]
total_edges=[]
def age_explain():
    data_loader = DataLoader(test_dataset,batch_size=1)
    data_loader1 = DataLoader(test_dataset,batch_size=1,shuffle=True)

    for epoch in range(5):
            t_loss=0
            count=0
            for data in data_loader:  # Indices to train against.
                data=data.to(device)
                loss = explainer.algorithm.train(epoch, model, data.x, data.edge_index, batch=data.batch,target=data.y.float())
                t_loss += loss
                count=count+1
            t_loss/= count
            print(('PGExlianer  Loss: {:.4f}'.format(t_loss)))
    f=open('NK_explain_nodes_index.txt','w',encoding='utf-8')
    f1=open('NK_explain_edges_index.txt','w',encoding='utf-8')
    for data in tqdm(data_loader):
        data=data.to(device)
        explanation = explainer(data.x, data.edge_index,batch=data.batch,target=data.y.float())
        path = 'subgraph_node.png'
        node_list,edge_list=explanation.visualize_graph(path,'networkx')
        f.writelines(str(node_list).strip("[]")) #nodes
        f.write('\n')
        f1.writelines(str(edge_list).strip("[]")) #edges
        f1.write('\n')
    f.close()
    f1.close()
    print("explain end......")

if __name__ == '__main__':
    run()
    age_explain()
