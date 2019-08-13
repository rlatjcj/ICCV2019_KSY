import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class RetinaNet(nn.Module):
    num_anchors = 9
    
    def __init__(self, 
                 backbone,
                 classes=599):
                 
        super(RetinaNet, self).__init__()
        self.backbone = self._set_backbone(backbone)
        self.classes = classes

        self.relu = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d( 256, 256, kernel_size=3, stride=2, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)

        # Top-down layers
        self.toplayer1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.toplayer2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        
        self.loc_head = self._make_head(self.num_anchors*4)
        self.cls_head = self._make_head(self.num_anchors*self.classes)

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        p6 = self.conv6(c5)
        p7 = self.conv7(self.relu(p6))

        # Top-down
        p5 = self.latlayer1(c5)
        p4 = self._upsample_add(p5, self.latlayer2(c4))
        p4 = self.toplayer1(p4)
        p3 = self._upsample_add(p4, self.latlayer3(c3))
        p3 = self.toplayer2(p3)

        loc_preds = []
        cls_preds = []
        for fm in [p3, p4, p5, p6, p7]:
            loc_pred = self.loc_head(fm)
            cls_pred = self.cls_head(fm)
            loc_pred = loc_pred.permute(0,2,3,1).contiguous().view(x.size(0),-1,4)                  # [N, 9*4,H,W] -> [N,H,W, 9*4] -> [N,H*W*9, 4]
            cls_pred = cls_pred.permute(0,2,3,1).contiguous().view(x.size(0),-1,self.classes)       # [N,9*20,H,W] -> [N,H,W,9*20] -> [N,H*W*9,20]
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)

        return torch.cat(loc_preds,1), torch.cat(cls_preds,1)

    def _set_backbone(self, backbone):
        if 'res' in backbone:
            # ResNet backbone
            from .resnet import ResNet
            if 'res50' in backbone:
                return ResNet(
                    block='bottleneck', 
                    num_blocks=[3,4,6,3],
                    is_seblock=True if 'seres' in backbone else False
                )

            elif 'res101' in backbone:
                return ResNet(
                    block='bottleneck', 
                    num_blocks=[3,4,23,3],
                    is_seblock=True if 'seres' in backbone else False
                )

        elif 'incep' in backbone:
            # Inception backbone
            pass

        elif 'xcep' in backbone:
            # Xception backbone
            pass

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def _make_head(self, out_planes):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))

        layers.append(nn.Conv2d(256, out_planes, kernel_size=3, stride=1, padding=1))
        
        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()


if __name__ == "__main__":
    net = RetinaNet()
    loc_preds, cls_preds = net(Variable(torch.randn(2,3,224,224)))
    print(loc_preds.size())
    print(cls_preds.size())
    loc_grads = Variable(torch.randn(loc_preds.size()))
    cls_grads = Variable(torch.randn(cls_preds.size()))
    loc_preds.backward(loc_grads)
    cls_preds.backward(cls_grads)

