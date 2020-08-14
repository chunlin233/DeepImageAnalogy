import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import copy
import time
import numpy as np
from torch.autograd import Variable

class FeatureExtractor(nn.Sequential):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

    def add_layer(self, name, layer):
        self.add_module(name, layer)

    def forward(self, x):
        feature_maps = []
        for module in self._modules:
            x = self._modules[module](x)
            feature_maps.append(x)
        return feature_maps


class VGG19:
    def __init__(self):

        vgg19_model = models.vgg19(pretrained=False)
        # pretrained_weights = "https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg19-d01eb7cb.pth"
        # vgg19_model.load_state_dict(model_zoo.load_url(pretrained_weights), strict=False)
        pretrained_weights = 'vgg19-dcbb9e9d.pth'
        vgg19_model.load_state_dict(torch.load(pretrained_weights), strict=False)
        self.vgg19_features = vgg19_model.features
        self.model = FeatureExtractor()  # the new Feature extractor module network
        conv_counter = 1
        relu_counter = 1
        block_counter = 1
        
        # build feature extractor, 相当于重命名了, layer并没有改变
        for i, layer in enumerate(list(self.vgg19_features)):

            if isinstance(layer, nn.Conv2d):
                name = "conv_" + str(block_counter) + "_" + str(conv_counter) + "__" + str(i)
                conv_counter += 1
                self.model.add_layer(name, layer)

            if isinstance(layer, nn.ReLU):
                name = "relu_" + str(block_counter) + "_" + str(relu_counter) + "__" + str(i)
                relu_counter += 1
                self.model.add_layer(name, nn.ReLU(inplace=False))

            if isinstance(layer, nn.MaxPool2d):
                name = "pool_" + str(block_counter) + "__" + str(i)
                relu_counter,conv_counter = 1,1
                block_counter += 1
                self.model.add_layer(name, nn.MaxPool2d((2, 2), ceil_mode=True))

        self.model.cuda()
        self._mean = (103.939, 116.779, 123.68)

    def forward_subnet(self, input_var, start_index, end_index):
        """
        将input_var输入model，从start_index层执行到end_index层，输出最后的feature map
        """
        for i, layer in enumerate(list(self.model)):
            if i >= start_index and i <= end_index:
                input_var = layer(input_var)
        return input_var

    def get_features(self, img_tensor, layers):
        """
        把选中层layers的feature map都提取出来, [F5,F4,F3,F2,F1,INPUT]
        """
        img_tensor = img_tensor.cuda()
        for channel in range(3):
            img_tensor[:, channel, :, :] -= self._mean[channel]

        img_var = Variable(img_tensor)
        feature_maps = self.model(img_var)
        features = []  # feature maps actually used
        for i, f in enumerate(feature_maps):
            if i in layers:
                features.append(f.data)
        features.reverse()
        features.append(img_var.data)    # Now the feature maps are [F5,F4,F3,F2,F1,INPUT]
        sizes = [f.size() for f in features]
        return features, sizes

    def get_deconvoluted_feat(self, writer, feat, curr_layer, init=None, feat_name='', lr=10, blob_layers=[29, 20, 11, 6, 1]):
        """
        -----------------input-----------------: 
        (1) feat: target feature->F_BP(nnf_AB) of end_layer (blob_layers[curr_layer])
        (2) curr_layer: curr_laye in range(5);
        (3) init(noise): input feature->the feature map of the layer that two layers ahead the curr_layer, F_BP(nnf_AB) of mid_layer (blob_layers[curr_layer+2])
        -----------------output-----------------:
        (4) out: the feature which is the output of inputing the optimized noise feature into model from start_layer to mid_layer;
        """
        blob_layers = blob_layers+[-1]
        end_layer = blob_layers[curr_layer]
        mid_layer = blob_layers[curr_layer + 1]
        start_layer = blob_layers[curr_layer + 2] + 1
        t_begin = time.time()
        print("="*20+"Deconvolution Start"+"="*20)
        print("Start:{},Mid:{},End:{}".format(start_layer,mid_layer,end_layer))
        layers = []
        for i, layer in enumerate(list(self.model)):
            if i >= start_layer and i <= end_layer:
                l = copy.deepcopy(layer)
                for p in l.parameters():
                    p.data = p.data.type(torch.DoubleTensor).cuda()
                layers.append(l)
        net = nn.Sequential(*layers).cuda()
        noise = init.type(torch.cuda.DoubleTensor).clone()
        target = Variable(feat.type(torch.cuda.DoubleTensor),requires_grad=False)
        noise_size = noise.size()        
        noise = Variable(noise.cuda(), requires_grad=True)
        optimizer = torch.optim.LBFGS([noise], lr=lr,max_iter=20,history_size=4,tolerance_grad=1e-4)
        def closure():
            optimizer.zero_grad()
            output = net(noise)
            loss = torch.mean((target - output)**2)
            loss.backward()
            return loss
        for i in range(25):
            loss = optimizer.step(closure)
            print("LBFGS iter:{} Loss:{}".format((i+1)*20, loss.item()))
            self.visualize_deconvolute_loss(
                writer, loss, i, curr_layer, feat_name)

        self.visualize_deconvolute_noise(
            writer, target, noise, i, curr_layer, feat_name)

        noise = noise.type(torch.cuda.FloatTensor)
        out = self.forward_subnet(input_var=noise, start_index=start_layer, end_index=mid_layer)
        elapse_time = time.time() - t_begin
        print("Deconvolution Finished, Elapsed: {:.2f}s".format(elapse_time))
        return out.data

    def visualize_deconvolute_loss(self, writer, loss, iter_num, curr_layer, feat_name):
        writer.add_scalar("deconv loss on layer{} for {}".format(
            curr_layer, feat_name), loss, iter_num)

    def visualize_deconvolute_noise(self, writer, target, noise, iter_num, curr_layer, feat_name):
        from torchvision.utils import make_grid
        img_target = make_grid(
            target[0, :10].unsqueeze(1), nrow=5, normalize=True)
        img_noise = make_grid(
            noise[0, :10].unsqueeze(1), nrow=5, normalize=True)
        writer.add_image("target for {}".format(feat_name), img_target, curr_layer)
        writer.add_image("noise for {}".format(feat_name), img_noise, curr_layer)
