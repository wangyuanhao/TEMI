import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from collections import OrderedDict


class Aligner(nn.Module):
    def __init__(self, params):
        super(Aligner, self).__init__()
        outputs = params["outputs"]
        self.fc1 = nn.Linear(outputs[2], outputs[2])
        self.fc2 = nn.Linear(outputs[2], outputs[2])

    def forward(self, h_G, h_W):
        h = torch.mul(self.fc1(h_G), self.fc2(h_W))
        return h

class ProjectionHead(nn.ModuleDict):
    def __init__(self, inputs, outputs):
        super(ProjectionHead, self).__init__()
        assert len(inputs) == len(outputs)
        layer_num = len(inputs)
        for i in range(layer_num):
            if i == layer_num - 1:
                self.add_module("linear%d" % i, nn.Linear(inputs[i], outputs[i]))
            else:
                self.add_module("linear%d" % i, nn.Linear(inputs[i], outputs[i]))
                self.add_module("relu%d" % i, nn.ReLU())

    def forward(self, X):
        x_ = X
        for name, layer in self.items():
            x_ = layer(x_)

        return x_


class MTAEncoder(nn.Module):
    def __init__(self, params):
        super(MTAEncoder, self).__init__()
        inputs = params["inputs"]
        outputs = params["outputs"]
        assert len(inputs) == len(outputs)
        self.encoder_layer1 = nn.Linear(inputs[0], outputs[0])
        self.ebn1 = nn.BatchNorm1d(num_features=outputs[0])
        self.encoder_layer2 = nn.Linear(inputs[1], outputs[1])
        self.ebn2 = nn.BatchNorm1d(num_features=outputs[1])
        self.encoder_layer3 = nn.Linear(inputs[2], outputs[2])

        self.projection_head = nn.Sequential(OrderedDict([
            ("lin1", nn.Linear(outputs[2], 64)),
            ("bn1", nn.BatchNorm1d(64)),
            ("relu", nn.Tanh()),
            ("lin2", nn.Linear(64, 128)),
            ("bn2", nn.BatchNorm1d(128))
        ]))
    
    def forward(self, X):
        layer1 = torch.tanh(self.ebn1(self.encoder_layer1(X)))
        layer2 = torch.tanh(self.ebn2(self.encoder_layer2(layer1)))
       
        layer3 = self.encoder_layer3(layer2)


        player3 = self.projection_head(layer3)

        return layer3, player3


class MTADecoder(nn.Module):
    def __init__(self, params):
        super(MTADecoder, self).__init__()
        inputs = params["inputs"]
        outputs = params["outputs"]
        assert len(inputs) == len(outputs)


        self.decoder_layer1 = nn.Linear(inputs[3], outputs[3])
        self.dbn1 = nn.BatchNorm1d(num_features=outputs[3])
        self.decoder_layer2 = nn.Linear(inputs[4], outputs[4])
        self.dbn2 = nn.BatchNorm1d(num_features=outputs[4])
        self.decoder_layer3 = nn.Linear(inputs[5], outputs[5])

     
    def forward(self, hidden):
        layer4 = torch.tanh(self.dbn1(self.decoder_layer1(hidden)))
        layer5 = torch.tanh(self.dbn2(self.decoder_layer2(layer4)))
        layer6 = self.decoder_layer3(layer5)


        return layer6
    

class MaskTranscriptomicAutoEncoder(nn.Module):
    def __init__(self, params):
        super(MaskTranscriptomicAutoEncoder, self).__init__()
        inputs = params["inputs"]
        outputs = params["outputs"]
        assert len(inputs) == len(outputs)
        self.encoder_layer1 = nn.Linear(inputs[0], outputs[0])
        self.ebn1 = nn.BatchNorm1d(num_features=outputs[0])
        self.encoder_layer2 = nn.Linear(inputs[1], outputs[1])
        self.ebn2 = nn.BatchNorm1d(num_features=outputs[1])
        self.encoder_layer3 = nn.Linear(inputs[2], outputs[2])



        self.decoder_layer1 = nn.Linear(inputs[3], outputs[3])
        self.dbn1 = nn.BatchNorm1d(num_features=outputs[3])
        self.decoder_layer2 = nn.Linear(inputs[4], outputs[4])
        self.dbn2 = nn.BatchNorm1d(num_features=outputs[4])
        self.decoder_layer3 = nn.Linear(inputs[5], outputs[5])

     
        self.projection_head = nn.Sequential(OrderedDict([
            ("lin1", nn.Linear(outputs[2], 64)),
            ("bn1", nn.BatchNorm1d(64)),
            ("relu", nn.Tanh()),
            ("lin2", nn.Linear(64, 128)),
            ("bn2", nn.BatchNorm1d(128))
        ]))
    
    def forward(self, X):
        layer1 = torch.tanh(self.ebn1(self.encoder_layer1(X)))
        layer2 = torch.tanh(self.ebn2(self.encoder_layer2(layer1)))
       
        layer3 = self.encoder_layer3(layer2)


        layer4 = torch.tanh(self.dbn1(self.decoder_layer1(layer3)))
        layer5 = torch.tanh(self.dbn2(self.decoder_layer2(layer4)))
        layer6 = self.decoder_layer3(layer5)

        player3 = self.projection_head(layer3)

        return layer3, player3, layer6
    
    
class OneLayerClassifier(nn.Module):
    def __init__(self, D, class_num):
        super(OneLayerClassifier, self).__init__()
        self.linear = nn.Linear(D, class_num)

    def forward(self, x):
        x_ = self.linear(x)
        return x_


class TwoLayerClassifier(nn.Module):
    def __init__(self, D, class_num, dropout):
        super(TwoLayerClassifier, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(D,  int(D/2)),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(int(D/2), class_num)
        )

    def forward(self, x):
        x_ = self.encoder(x)
        return x_


class FittingBlocks(nn.Module):
    def __init__(self, inputs, outputs, dropout):
        super(FittingBlocks, self).__init__()
        self.FitBlocks = nn.Sequential(
            nn.Linear(inputs, outputs),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        z = self.FitBlocks(x)
        return z


class PatchScoreNet(nn.ModuleDict):
    def __init__(self, fc_inputs, fc_outputs, ydim, dropout):
        super(PatchScoreNet, self).__init__()
        for i in range(len(fc_inputs)):
            if i == (len(fc_inputs) - 1):
                dropout = 0.0
            self.add_module("fitblock%d"% (i+1), FittingBlocks(fc_inputs[i], fc_outputs[i], dropout))

        self.add_module("regressor", nn.Linear(fc_outputs[i], ydim))

    def forward(self, x):
        z = None
        x_ = x
        for name, layer in self.items():
            if name == "regressor":
                z = layer(x_)
            else:
                x_ = layer(x_)

        return x_, z


class SimpleAttention(nn.Module):
    def __init__(self, D, H):
        super(SimpleAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(D, H),
            nn.Tanh()
        )

    def forward(self, X):
        atten = self.attention(X)
        return atten



class DotProductAttention(nn.Module):
    def __init__(self, D, H):
        super(DotProductAttention, self).__init__()
        self.D = D 
        self.H = H 

        self.scaling_weight = nn.ParameterList([
            nn.Parameter(torch.randn(D))
            for _ in range(H)
        ])

        self.in_place_scaling_weight = nn.Parameter(
            torch.full((D,), 1.0 / D)
        )

        in_place_scaling_weight = nn.Parameter(torch.Tensor(D))
        torch.nn.init.constant_(in_place_scaling_weight, 1/D)
        self.in_place_scaling_weight = in_place_scaling_weight
        

    def forward(self, X):
        atten_weight = []
        for i in range(self.H):
            scaling_X_ = torch.mul(X, self.scaling_weight[i].to(X.device))
            weight_ = torch.bmm(scaling_X_, torch.transpose(scaling_X_, 1, 2)) / torch.tensor(X.shape[2])
            weight = weight_ - torch.diag_embed(torch.diagonal(weight_, dim1=1, dim2=2))
            average_weight = torch.sum(weight, dim=2, keepdim=True)
            atten_weight.append(average_weight)
        
        atten_weight_ = torch.cat(atten_weight, dim=2)
        X_ = torch.mul(X, self.in_place_scaling_weight.to(X.device))

        return atten_weight_, X_
        
        

class ConvBlock(nn.Module):
    def __init__(self, in_chan, D, out_chan, dropout):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Linear(D, out_chan, bias=False),
            nn.BatchNorm1d(in_chan),
            nn.Tanh(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_ = self.conv(x)

        return x_


class ConvAttenBlock(nn.Module):
    def __init__(self, in_chan, D, out_chan, heads, dropout):
        super(ConvAttenBlock, self).__init__()
        self.conv = ConvBlock(in_chan, D, out_chan, dropout)
        self.attention = DotProductAttention(in_chan, heads)


    def forward(self, X):
        z = self.conv(X)
        z_ = z.transpose(1, 2)
        atten, z__ = self.attention(z_)
        atten_ = torch.softmax(atten, dim=1)

        z___ = torch.bmm(atten_.transpose(1, 2), z__)


        return z___

class ConvAttenFusion_(nn.ModuleDict):
    """[100, 512, 16, 8]"""
    """[8, 100, 8, 4]"""
    def __init__(self, params, class_num):
        in_chan = params["in_chans"]
        D = params["dims"]
        out_chan = params["out_chans"]
        heads = params["heads"]
        dropout = params["dropout"]
        super(ConvAttenFusion_, self).__init__()
        layers = len(in_chan)
        for i in range(layers):
            self.add_module("ConvAtten%d" % i, ConvAttenBlock(in_chan[i], D[i],
                                                              out_chan[i], heads[i], dropout[i]))

        flatten_dim = heads[-1] * heads[-2]
        self.add_module("Classifier", OneLayerClassifier(flatten_dim, class_num))


    def forward(self, X):
        z = None
        x__ = None
        x_ = deepcopy(X)
        atten_weight = None
        for name, layer in self.items():
            if name == "Classifier":
                x__ = x_.view(x_.shape[0], -1)
                z = layer(x__)
  
            else:
                x_ = layer(x_)
            if name == "ConvAtten0":
                atten_weight = x_

        return x__, z, atten_weight
    
    
class ConvAttenFusion(nn.ModuleDict):
    """[100, 512, 16, 8]"""
    """[8, 100, 8, 4]"""
    def __init__(self, params, class_num):
        in_chan = params["in_chans"]
        D = params["dims"]
        out_chan = params["out_chans"]
        heads = params["heads"]
        dropout = params["dropout"]
        super(ConvAttenFusion, self).__init__()
        layers = len(in_chan)
        for i in range(layers):
            self.add_module("ConvAtten%d" % i, ConvAttenBlock(in_chan[i], D[i],
                                                              out_chan[i], heads[i], dropout[i]))

        flatten_dim = heads[-1] * heads[-2]
        self.add_module("Classifier", OneLayerClassifier(flatten_dim, class_num))


    def forward(self, X):
        z = None
        x__ = None
        x_ = deepcopy(X)
        for name, layer in self.items():
            if name == "Classifier":
                x__ = x_.view(x_.shape[0], -1)
                z = layer(x__)
            else:
                x_ = layer(x_)

        return x__, z



class MILAttenBlock(nn.Module):
    def __init__(self, D, L, dropout):
        super(MILAttenBlock, self).__init__()
        self.D, self.L = D, L
        self.attention = nn.Sequential(
            nn.Linear(self.D, self.L),
            nn.Dropout(dropout),
            nn.Tanh(),
            nn.Linear(self.L, 1)
        )

    def forward(self, X):
        A = self.attention(X)
        W = F.softmax(A, dim=1)
        z = torch.bmm(X.transpose(1, 2), W)
        z_ = z.squeeze()
        return z_


class GatedAttenBlock(nn.Module):
    def __init__(self, D, L, dropout):
        super(GatedAttenBlock, self).__init__()
        self.D, self.L = D, L

        self.attention_U = nn.Sequential(
            nn.Linear(self.D, self.L),
            nn.Sigmoid(),
            nn.Dropout(dropout)
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.D, self.L),
            nn.Tanh(),
            nn.Dropout(dropout)
        )

        self.weight = nn.Linear(self.L, 1)


    def forward(self, X):
        A_U = self.attention_U(X)
        A_V = self.attention_V(X)
        W_ = self.weight(A_V * A_U)
        W = F.softmax(W_, dim=1)
        z = torch.bmm(X.transpose(1, 2), W)
        z_ = z.squeeze()
        return z_


class MILAttenFusion(nn.ModuleDict):

    def __init__(self, D, L, CD, pinputs, poutputs, dropout):
        super(MILAttenFusion, self).__init__()
        self.add_module("Attention", MILAttenBlock(D, L, dropout[0]))
        self.add_module("Classifier", OneLayerClassifier(D))
        self.add_module("Projection", ProjectionHead(pinputs, poutputs))

    def forward(self, X):
        z = None
        projection = None
        x_ = deepcopy(X)
        for name, layer in self.items():
            if name == "Classifier":
                z = layer(x_)
            elif name == "Projection":
                projection = layer(x_)
            else:
                x_ = layer(x_)

        return projection, x_, z


class GMILAttenFusion(nn.ModuleDict):

    def __init__(self, D, L, CD,pinputs, poutputs, dropout):
        super(GMILAttenFusion, self).__init__()
        self.add_module("GAttention", GatedAttenBlock(D, L, dropout[0]))
        self.add_module("Classifier", TwoLayerClassifier(D, CD, dropout[1]))
        self.add_module("Projection", ProjectionHead(pinputs, poutputs))

    def forward(self, X):
        z = None
        projection = None
        x_ = deepcopy(X)
        for name, layer in self.items():
            if name == "Classifier":
                z = layer(x_)
            elif name == "Projection":
                projection = layer(x_)
            else:
                x_ = layer(x_)

        return projection, x_, z


class ConvFusion(nn.ModuleDict):
    def __init__(self, in_chan, D, out_chan, CD, GD, dropouts):
        super(ConvFusion, self).__init__()
        self.add_module("Conv1D", ConvBlock(in_chan, D, out_chan, dropouts[0]))
        self.add_module("Classifier", TwoLayerClassifier(in_chan, CD, dropouts[1]))
        self.add_module("Projection", ProjectionHead([in_chan], [GD]))

    def forward(self, X):
        z = None
        projection = None
        x_ = deepcopy(X)
        for name, layer in self.items():
            if name == "Classifier":
                x_ = x_.squeeze()
                z = layer(x_)
            elif name == "Projection":
                x_ = x_.squeeze()
                projection = layer(x_)
            else:
                x_ = layer(x_)

        return projection, x_, z




