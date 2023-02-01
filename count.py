
from operator import imod
from torchvision.models import resnet50
from ptflops import get_model_complexity_info
from thop import profile
from fvcore.nn import FlopCountAnalysis, parameter_count_table
import torch
from torchprofile import profile_macs

from torchstat import stat

from net.RMMLP.model import RMMLP
from torchsummary import summary
def model_structure(model):
    blank = ' '
    print('-'*90)
    print('|'+' '*11+'weight name'+' '*10+'|' \
            +' '*15+'weight shape'+' '*15+'|' \
            +' '*3+'number'+' '*3+'|')
    print('-'*90)
    num_para = 0
    type_size = 1  ##如果是浮点数就是4
    
    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30: 
            key = key + (30-len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40-len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10-len(str_num)) * blank
    
        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-'*90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-'*90)



# model_path = '/home/amax/repository/jc/MySCI/models/batmlp1816/model.pth'
# model = torch.load(model_path)
model = RMMLP() 

# input = torch.randn(1,3,256,256)
# flops, params = profile(model, (input,),verbose=False)
# print('flops: ', flops, 'params: ', params)
# print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
model_structure(model)
#stat(model, (3, 256, 256))

# flops,params = get_model_complexity_info(model,(3,256,256),as_strings=True,print_per_layer_stat=True)
# print('Flops:{}'.format(flops))
# print('Params:'+params)

# inputs = torch.randn(1,3, 256, 256).cuda(0)
# # out = model(inputs)

# macs = profile_macs(model, inputs) / 1e9
# print(f'GFLOPs {macs}.')

# inputdata = torch.randn(1, 3, 256, 256)  
# if torch.cuda.is_available():
#     model1 = model.cuda()
#     inputdata = inputdata.cuda()
# flops, params = profile(model, inputs=(inputdata, ))
# print("GFLOPs :{:.2f}, Params : {:.2f}".format(flops/1e9,params/1e6)) #flops单位G，para单位M