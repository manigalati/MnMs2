import torch.nn as nn
import torch
import os
import numpy as np
from utils import device
from utils import MSELoss, GDLoss
from utils import DC, HD

class Reconstructor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.init_layers()
        self.apply(self.weight_init)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=1e-5)

    def init_layers(self):
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=self.args.in_channels, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.Conv2d(
                in_channels=32,
                out_channels=self.args.latent_size,
                kernel_size=self.args.last_layer[0],
                stride=self.args.last_layer[1],
                padding=self.args.last_layer[2]
            )
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.args.latent_size,
                out_channels=32,
                kernel_size=self.args.last_layer[0],
                stride=self.args.last_layer[1],
                padding=self.args.last_layer[2]
            ),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(.2),
            nn.Dropout(0.5),

            nn.ConvTranspose2d(in_channels=32, out_channels=self.args.in_channels, kernel_size=4, stride=2, padding=1),
            nn.Softmax(dim=1)
        )

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_uniform_(m.weight)

    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction

    def Loss(self, prediction, target, epoch=None, validation=False):
        contributes = {}
        contributes["MSELoss"] = MSELoss(prediction,target)
        contributes["GDLoss"] = GDLoss(prediction,target)
        contributes["Total"] = sum(contributes.values())
        if validation:
            return {k:v.item() for k,v in contributes.items()}
        return contributes["Total"]

    def Metrics(self,prediction,target):    
        metrics = {}
        for c, key in enumerate(["LV_", "MYO_", "RV_"], start=1):
            ref = np.copy(target)
            pred = np.copy(prediction)
            ref = np.where(ref != c, 0, 1)
            pred = np.where(pred != c, 0, 1)  
            metrics[key + "dc"] = DC(pred, ref)
            metrics[key + "hd"] = HD(pred, ref)  
        return metrics

    def training_routine(self, epochs, train_loader, val_loader, ckpt_folder):
        if not os.path.isdir(ckpt_folder):
            os.makedirs(ckpt_folder)
        history = []
        best_acc = np.inf
        for epoch in epochs:
            self.train()
            for batch in train_loader:
                batch = batch["gt"].to(device)
                self.optimizer.zero_grad()
                reconstruction = self.forward(batch)
                loss = self.Loss(reconstruction, batch, epoch)
                loss.backward()
                self.optimizer.step()

            self.eval()
            with torch.no_grad():
                result = self.evaluation_routine(val_loader)
            if result["Total"] < best_acc or epoch%10 == 0:
                ckpt = os.path.join(ckpt_folder, "{:03d}.pth".format(epoch))
                if result["Total"] < best_acc:
                    best_acc = result["Total"]
                    ckpt = ckpt.split(".pth")[0] + "_best.pth"
                torch.save({"R": self.state_dict(), "R_optim": self.optimizer.state_dict()}, ckpt)
            
            self.epoch_end(epoch, result)
            history.append(result["Total"])
        return history

    def evaluation_routine(self, val_loader):
        epoch_summary={}
        for patient in val_loader:
            gt, reconstruction = [], []
            for batch in patient:
                batch = {"gt": batch["gt"].to(device)}
                batch["reconstruction"] = self.forward(batch["gt"])
                gt = torch.cat([gt,batch["gt"]], dim=0) if len(gt)>0 else batch["gt"]
                reconstruction = torch.cat([reconstruction, batch["reconstruction"]], dim=0) if len(reconstruction)>0 else batch["reconstruction"]
                for k,v in self.Loss(batch["reconstruction"], batch["gt"], validation=True).items():
                    if k not in epoch_summary.keys():
                        epoch_summary[k] = []
                    epoch_summary[k].append(v)
            gt = np.argmax(gt.cpu().numpy(), axis=1)
            gt = {"ED": gt[:len(gt)//2], "ES": gt[len(gt)//2:]}
            reconstruction = np.argmax(reconstruction.cpu().numpy(), axis=1)
            reconstruction = {"ED": reconstruction[:len(reconstruction)//2], "ES": reconstruction[len(reconstruction)//2:]}
            for phase in ["ED","ES"]:
                for k,v in self.Metrics(reconstruction[phase],gt[phase]).items():
                  if k not in epoch_summary.keys(): epoch_summary[k]=[]
                  epoch_summary[k].append(v)
        epoch_summary = {k:np.mean(v) for k,v in epoch_summary.items()}
        return epoch_summary

    def epoch_end(self,epoch,result):
        print("\033[1mEpoch [{}]\033[0m".format(epoch))
        header, row = "", ""
        for k,v in result.items():
            header += "{:.6}\t".format(k)
            row += "{:.6}\t".format("{:.4f}".format(v))
        print(header);print(row)