import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df=pd.read_csv("dataset.csv")

X=df.drop("label",axis=1).values
y=df["label"].values

scaler=StandardScaler()
X=scaler.fit_transform(X)

X_train,X_test,y_train,y_test=train_test_split(
X,y,test_size=0.2,random_state=42)

X_train=torch.tensor(X_train).float()
y_train=torch.tensor(y_train).long()

X_test=torch.tensor(X_test).float()
y_test=torch.tensor(y_test).long()

class Model(nn.Module):

    def __init__(self,dim):

        super().__init__()

        self.net=nn.Sequential(
            nn.Linear(dim,128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,4)
        )

    def forward(self,x):
        return self.net(x)

model=Model(X.shape[1])

opt=torch.optim.Adam(model.parameters(),lr=0.001)

loss_fn=nn.CrossEntropyLoss()

for epoch in range(30):

    pred=model(X_train)

    loss=loss_fn(pred,y_train)

    opt.zero_grad()
    loss.backward()
    opt.step()

    if epoch%5==0:

        acc=(model(X_test).argmax(1)==y_test).float().mean()

        print(epoch,loss.item(),acc.item())

torch.save(model.state_dict(),"model.pt")