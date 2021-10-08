from utils import Baseline

class Baseline_1(Baseline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_layers()
        self.apply(self.weight_init)
        self.deep = True
        self.optimizer = torch.optim.Adam(self.parameters(), lr=5e-4, weight_decay=1e-5)

    def init_layers(self):
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=48, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(.01),
                nn.BatchNorm2d(num_features=48),#contr_1_1

                nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(.01),
                nn.BatchNorm2d(num_features=48)#contr_1_2
            ),

            nn.Sequential(
                nn.MaxPool2d(kernel_size=2),#pool1

                nn.Conv2d(in_channels=48, out_channels=48*2, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(.01),
                nn.BatchNorm2d(num_features=48*2),#contr_2_1

                nn.Conv2d(in_channels=48*2, out_channels=48*2, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(.01),
                nn.BatchNorm2d(num_features=48*2)#contr_2_2
            ),

            nn.Sequential(
                nn.MaxPool2d(kernel_size=2),#pool2

                nn.Dropout(0.3),

                nn.Conv2d(in_channels=48*2, out_channels=48*4, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(.01),
                nn.BatchNorm2d(num_features=48*4),#contr_3_1

                nn.Conv2d(in_channels=48*4, out_channels=48*4, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(.01),
                nn.BatchNorm2d(num_features=48*4)#contr_3_2
            ),

            nn.Sequential(
                nn.MaxPool2d(kernel_size=2),#pool3

                nn.Dropout(0.3),

                nn.Conv2d(in_channels=48*4, out_channels=48*8, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(.01),
                nn.BatchNorm2d(num_features=48*8),#contr_4_1

                nn.Conv2d(in_channels=48*8, out_channels=48*8, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(.01),
                nn.BatchNorm2d(num_features=48*8)#contr_4_2
            ),

            nn.Sequential(
                nn.MaxPool2d(kernel_size=2),#pool_4

                nn.Dropout(0.3),

                nn.Conv2d(in_channels=48*8, out_channels=48*16, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(.01),
                nn.BatchNorm2d(num_features=48*16),#encode_1

                nn.Conv2d(in_channels=48*16, out_channels=48*16, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(.01),
                nn.BatchNorm2d(num_features=48*16),#encode_2

                nn.Upsample(scale_factor=2)#upscale1
            ),

            nn.Sequential(
                nn.Dropout(0.3),

                nn.Conv2d(in_channels=48*16+48*8, out_channels=48*8, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(.01),
                nn.BatchNorm2d(num_features=48*8),#expand_1_1

                nn.Conv2d(in_channels=48*8, out_channels=48*8, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(.01),
                nn.BatchNorm2d(num_features=48*8),#expand_1_2

                nn.Upsample(scale_factor=2)#upscale2
            ),

            nn.Sequential(
                nn.Dropout(0.3),

                nn.Conv2d(in_channels=48*8+48*4, out_channels=48*4, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(.01),
                nn.BatchNorm2d(num_features=48*4),#expand_2_1

                nn.Conv2d(in_channels=48*4, out_channels=48*4, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(.01),
                nn.BatchNorm2d(num_features=48*4),#expand_2_2
            ),

            nn.Upsample(scale_factor=2),#upscale3

            nn.Sequential(
                nn.Dropout(0.3),

                nn.Conv2d(in_channels=48*4+48*2, out_channels=48*2, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(.01),
                nn.BatchNorm2d(num_features=48*2),#expand_3_1

                nn.Conv2d(in_channels=48*2, out_channels=48*2, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(.01),
                nn.BatchNorm2d(num_features=48*2),#expand_3_2
            ),

            nn.Upsample(scale_factor=2),#upscale4

            nn.Sequential(
                nn.Dropout(0.3),

                nn.Conv2d(in_channels=48*2+48, out_channels=48, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(.01),
                nn.BatchNorm2d(num_features=48),#expand_4_1

                nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(.01),
                nn.BatchNorm2d(num_features=48),#expand_4_2

                nn.Conv2d(in_channels=48, out_channels=4, kernel_size=1, stride=1, padding=0),#output_segmentation
            ),

            nn.Sequential(
                nn.Conv2d(in_channels=48*4, out_channels=4, kernel_size=1, stride=1, padding=0),#ds2_1x1_conv
                nn.Upsample(scale_factor=2)#ds1_ds2_sum_upscale
            ),

            nn.Conv2d(in_channels=48*2, out_channels=4, kernel_size=1, stride=1, padding=0),#ds3_1x1_conv

            nn.Upsample(scale_factor=2),#ds1_ds2_sum_upscale_ds3_sum_upscale

            nn.Softmax(dim=1)#output_flattened
        ])

    def weight_init(self,m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight,nonlinearity='relu')

    def forward(self, x):
        contr_1_2 = self.layers[0](x)
        contr_2_2 = self.layers[1](contr_1_2)
        contr_3_2 = self.layers[2](contr_2_2)
        contr_4_2 = self.layers[3](contr_3_2)

        upscale1 = self.layers[4](contr_4_2)
        upscale2 = self.layers[5](torch.cat([upscale1, contr_4_2], dim=1))
        expand_2_2 = self.layers[6](torch.cat([upscale2, contr_3_2], dim=1))
        upscale3 = self.layers[7](expand_2_2)
        expand_3_2 = self.layers[8](torch.cat([upscale3, contr_2_2], dim=1))
        upscale4 = self.layers[9](expand_3_2)
        output_segmentation = self.layers[10](torch.cat([upscale4, contr_1_2], dim=1))
        
        if(self.deep):
            ds1_ds2_sum_upscale = self.layers[11](expand_2_2)
            ds3_1x1_conv = self.layers[12](expand_3_2)
            ds1_ds2_sum_upscale_ds3_sum = ds1_ds2_sum_upscale+ds3_1x1_conv
            ds1_ds2_sum_upscale_ds3_sum_upscale = self.layers[13](ds1_ds2_sum_upscale_ds3_sum)
            
            output_segmentation += ds1_ds2_sum_upscale_ds3_sum_upscale
        return self.layers[14](output_segmentation)
    
    def adjust_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] *= 0.985