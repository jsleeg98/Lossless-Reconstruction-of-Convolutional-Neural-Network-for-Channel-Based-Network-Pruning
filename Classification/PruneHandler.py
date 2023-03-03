import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torchvision.models.resnet as resnet

class PruneHandler():
    def __init__(self, model):
        self.model = model
        self.remain_index = []
        self.union_index = []

    def get_remain_index(self):
        for name, module in self.model.named_children():
            if isinstance(module, torch.nn.Conv2d):
                tmp_remain_index = torch.where(torch.norm(module.weight, p=1, dim=(1, 2, 3)) != 0)[0].tolist()
                self.remain_index.append([tmp_remain_index])
            elif isinstance(module, torch.nn.Sequential):
                li_li_remain_index = []
                for name_, module_ in module.named_children():
                    if isinstance(module_, resnet.BasicBlock):
                        li_remain_index = []
                        for name__, module__ in module_.named_children():
                            if isinstance(module__, torch.nn.Conv2d):
                                tmp_remain_index = torch.where(torch.norm(module__.weight, p=1, dim=(1, 2, 3)) != 0)[0].tolist()
                                li_remain_index.append(tmp_remain_index)
                            elif isinstance(module__, torch.nn.Sequential):
                                for name___, module___ in module__.named_children():
                                    if isinstance(module___, torch.nn.Conv2d):
                                        tmp_remain_index = torch.where(torch.norm(module___.weight, p=1, dim=(1, 2, 3)) != 0)[0].tolist()
                                        li_remain_index.append(tmp_remain_index)
                        li_li_remain_index.append(li_remain_index)
                    elif isinstance(module_, resnet.Bottleneck):
                        li_remain_index = []
                        for name__, module__ in module_.named_children():
                            if isinstance(module__, torch.nn.Conv2d):
                                tmp_remain_index = torch.where(torch.norm(module__.weight, p=1, dim=(1, 2, 3)) != 0)[
                                    0].tolist()
                                li_remain_index.append(tmp_remain_index)
                            elif isinstance(module__, torch.nn.Sequential):
                                for name___, module___ in module__.named_children():
                                    if isinstance(module___, torch.nn.Conv2d):
                                        tmp_remain_index = \
                                        torch.where(torch.norm(module___.weight, p=1, dim=(1, 2, 3)) != 0)[0].tolist()
                                        li_remain_index.append(tmp_remain_index)
                        li_li_remain_index.append(li_remain_index)
                self.remain_index.append(li_li_remain_index)

    def union_remain_index_basic(self):
        first = self.remain_index[0][0]  # resnet34 first conv1 insert to layer1
        self.remain_index[1].insert(0, [first])
        del self.remain_index[0]

        for li_li_remain_index in self.remain_index:
            self.set = set()
            set_union_index = self.set
            for li_remain_index in li_li_remain_index:
                if len(li_remain_index) != 3:
                    set_remain_index = set(li_remain_index[-1])
                    set_union_index = set_union_index.union(set_remain_index)
                elif len(li_remain_index) == 3:
                    set_remain_index = set(li_remain_index[1])
                    set_union_index = set_union_index.union(set_remain_index)
                    set_remain_index = set(li_remain_index[2])
                    set_union_index = set_union_index.union(set_remain_index)
            self.union_index.append(list(set_union_index))

        # update remain index
        for li_li_remain_index, li_union_index in zip(self.remain_index, self.union_index):
            for li_remain_index in li_li_remain_index:
                if len(li_remain_index) != 3:
                    li_remain_index[-1] = li_union_index
                elif len(li_remain_index) == 3:
                    li_remain_index[1] = li_union_index
                    li_remain_index[2] = li_union_index

        # print remain index len
        # for li_li_remain_index in self.remain_index:
        #     for i, li_remain_index in enumerate(li_li_remain_index):
        #         print(i)
        #         for j, remain_index in enumerate(li_remain_index):
        #
        #             print(len(remain_index))
        #         print('-' * 20)

    def union_remain_index_bottle(self):
        # union index except first conv1
        for li_li_remain_index in self.remain_index[1:]:
            self.set = set()
            set_union_index = self.set
            for li_remain_index in li_li_remain_index:
                if len(li_remain_index) == 4:
                    set_remain_index = set(li_remain_index[-2])
                    set_union_index = set_union_index.union(set_remain_index)
                    set_remain_index = set(li_remain_index[-1])
                    set_union_index = set_union_index.union(set_remain_index)
                elif len(li_remain_index) == 3:
                    set_remain_index = set(li_remain_index[-1])
                    set_union_index = set_union_index.union(set_remain_index)
            self.union_index.append(list(set_union_index))

        # update remain index
        for li_li_remain_index, li_union_index in zip(self.remain_index[1:], self.union_index):
            for li_remain_index in li_li_remain_index:
                if len(li_remain_index) == 4:
                    li_remain_index[-1] = li_union_index
                    li_remain_index[-2] = li_union_index
                elif len(li_remain_index) == 3:
                    li_remain_index[-1] = li_union_index

    def and_remain_index_bottle(self):
        # union index except first conv1
        for li_li_remain_index in self.remain_index[1:]:
            self.set = set()
            set_union_index = self.set
            for li_remain_index in li_li_remain_index:
                if len(li_remain_index) == 4:
                    set_remain_index = set(li_remain_index[-2])
                    set_union_index = set_union_index.union(set_remain_index)
                    set_remain_index = set(li_remain_index[-1])
                    set_union_index = set_union_index.intersection(set_remain_index)
                elif len(li_remain_index) == 3:
                    set_remain_index = set(li_remain_index[-1])
                    set_union_index = set_union_index.intersection(set_remain_index)
            self.union_index.append(list(set_union_index))

        # update remain index
        for li_li_remain_index, li_union_index in zip(self.remain_index[1:], self.union_index):
            for li_remain_index in li_li_remain_index:
                if len(li_remain_index) == 4:
                    li_remain_index[-1] = li_union_index
                    li_remain_index[-2] = li_union_index
                elif len(li_remain_index) == 3:
                    li_remain_index[-1] = li_union_index

    def reconstruction_basic(self):
        flatten_remain_index = []
        for li_li_remain_index in self.remain_index:
            for li_remain_index in li_li_remain_index:
                for remain_index in li_remain_index:
                    flatten_remain_index.append(remain_index)

        idx = 0

        for name, module in self.model.named_children():
            if isinstance(module, torch.nn.Conv2d):
                module.weight = torch.nn.parameter.Parameter(torch.index_select(module.weight, 0, torch.tensor(flatten_remain_index[idx])))
                module.out_channels = len(flatten_remain_index[idx])
                tmp_in_channels = module.out_channels
            elif isinstance(module, torch.nn.BatchNorm2d):
                module.weight = torch.nn.parameter.Parameter(torch.index_select(module.weight, 0, torch.tensor(flatten_remain_index[idx])))
                module.bias = torch.nn.parameter.Parameter(torch.index_select(module.bias, 0, torch.tensor(flatten_remain_index[idx])))
                module.running_mean = torch.index_select(module.running_mean, 0, torch.tensor(flatten_remain_index[idx]))
                module.running_var = torch.index_select(module.running_var, 0, torch.tensor(flatten_remain_index[idx]))
                module.num_features = len(flatten_remain_index[idx])

            elif isinstance(module, torch.nn.Sequential):
                for name_, module_ in module.named_children():
                    if isinstance(module_, resnet.BasicBlock):
                        for name__, module__ in module_.named_children():
                            if isinstance(module__, torch.nn.Conv2d):
                                module__.in_channels = tmp_in_channels
                                module__.weight = torch.nn.parameter.Parameter(
                                    torch.index_select(module__.weight, 1, torch.tensor(flatten_remain_index[idx])))
                                idx += 1
                                module__.weight = torch.nn.parameter.Parameter(
                                    torch.index_select(module__.weight, 0, torch.tensor(flatten_remain_index[idx])))
                                module__.out_channels = len(flatten_remain_index[idx])
                                tmp_in_channels = module__.out_channels
                            elif isinstance(module__, torch.nn.BatchNorm2d):
                                module__.weight = torch.nn.parameter.Parameter(
                                   torch.index_select(module__.weight, 0, torch.tensor(flatten_remain_index[idx])))
                                module__.bias = torch.nn.parameter.Parameter(
                                    torch.index_select(module__.bias, 0, torch.tensor(flatten_remain_index[idx])))
                                module__.running_mean = torch.index_select(module__.running_mean, 0, torch.tensor(flatten_remain_index[idx]))
                                module__.running_var = torch.index_select(module__.running_var, 0, torch.tensor(flatten_remain_index[idx]))
                                module__.num_features = len(flatten_remain_index[idx])
                            elif isinstance(module__, torch.nn.Sequential):  # downsample
                                for name___, module___ in module__.named_children():
                                    if isinstance(module___, torch.nn.Conv2d):
                                        module___.in_channels = len(flatten_remain_index[idx-2])
                                        module___.weight = torch.nn.parameter.Parameter(
                                            torch.index_select(module___.weight, 1,
                                                               torch.tensor(flatten_remain_index[idx-2])))
                                        idx += 1
                                        module___.weight = torch.nn.parameter.Parameter(
                                            torch.index_select(module___.weight, 0,
                                                               torch.tensor(flatten_remain_index[idx])))
                                        module___.out_channels = len(flatten_remain_index[idx])
                                        tmp_in_channels = module___.out_channels
                                    elif isinstance(module___, torch.nn.BatchNorm2d):
                                        module___.weight = torch.nn.parameter.Parameter(
                                            torch.index_select(module___.weight, 0,
                                                               torch.tensor(flatten_remain_index[idx])))
                                        module___.bias = torch.nn.parameter.Parameter(
                                            torch.index_select(module___.bias, 0,
                                                               torch.tensor(flatten_remain_index[idx])))
                                        module___.running_mean = torch.index_select(module___.running_mean, 0,
                                                               torch.tensor(flatten_remain_index[idx]))
                                        module___.running_var = torch.index_select(module___.running_var, 0,
                                                               torch.tensor(flatten_remain_index[idx]))
                                        module___.num_features = len(flatten_remain_index[idx])
            elif isinstance(module, torch.nn.Linear):
                module.in_features = tmp_in_channels
                module.weight = torch.nn.parameter.Parameter(torch.index_select(module.weight, 1, torch.tensor(flatten_remain_index[idx])))

    def reconstruction_bottle(self):
        flatten_remain_index = []
        for li_li_remain_index in self.remain_index:
            if len(li_li_remain_index) == 1:
                flatten_remain_index.append(li_li_remain_index[0])
            else:
                for li_remain_index in li_li_remain_index:
                    for remain_index in li_remain_index:
                        flatten_remain_index.append(remain_index)

        idx = 0

        for name, module in self.model.named_children():
            if isinstance(module, torch.nn.Conv2d):
                module.weight = torch.nn.parameter.Parameter(torch.index_select(module.weight, 0, torch.tensor(flatten_remain_index[idx])))
                module.out_channels = len(flatten_remain_index[idx])
                tmp_in_channels = module.out_channels
            elif isinstance(module, torch.nn.BatchNorm2d):
                module.weight = torch.nn.parameter.Parameter(torch.index_select(module.weight, 0, torch.tensor(flatten_remain_index[idx])))
                module.bias = torch.nn.parameter.Parameter(torch.index_select(module.bias, 0, torch.tensor(flatten_remain_index[idx])))
                module.running_mean = torch.index_select(module.running_mean, 0, torch.tensor(flatten_remain_index[idx]))
                module.running_var = torch.index_select(module.running_var, 0, torch.tensor(flatten_remain_index[idx]))
                module.num_features = len(flatten_remain_index[idx])

            elif isinstance(module, torch.nn.Sequential):
                for name_, module_ in module.named_children():
                    if isinstance(module_, resnet.Bottleneck):
                        for name__, module__ in module_.named_children():
                            if isinstance(module__, torch.nn.Conv2d):
                                module__.in_channels = tmp_in_channels
                                module__.weight = torch.nn.parameter.Parameter(
                                    torch.index_select(module__.weight, 1, torch.tensor(flatten_remain_index[idx])))
                                idx += 1
                                module__.weight = torch.nn.parameter.Parameter(
                                    torch.index_select(module__.weight, 0, torch.tensor(flatten_remain_index[idx])))
                                module__.out_channels = len(flatten_remain_index[idx])
                                tmp_in_channels = module__.out_channels
                            elif isinstance(module__, torch.nn.BatchNorm2d):
                                module__.weight = torch.nn.parameter.Parameter(
                                   torch.index_select(module__.weight, 0, torch.tensor(flatten_remain_index[idx])))
                                module__.bias = torch.nn.parameter.Parameter(
                                    torch.index_select(module__.bias, 0, torch.tensor(flatten_remain_index[idx])))
                                module__.running_mean = torch.index_select(module__.running_mean, 0, torch.tensor(flatten_remain_index[idx]))
                                module__.running_var = torch.index_select(module__.running_var, 0, torch.tensor(flatten_remain_index[idx]))
                                module__.num_features = len(flatten_remain_index[idx])
                            elif isinstance(module__, torch.nn.Sequential):  # downsample
                                for name___, module___ in module__.named_children():
                                    if isinstance(module___, torch.nn.Conv2d):
                                        module___.in_channels = len(flatten_remain_index[idx-3])
                                        module___.weight = torch.nn.parameter.Parameter(
                                            torch.index_select(module___.weight, 1,
                                                               torch.tensor(flatten_remain_index[idx-3])))
                                        idx += 1
                                        module___.weight = torch.nn.parameter.Parameter(
                                            torch.index_select(module___.weight, 0,
                                                               torch.tensor(flatten_remain_index[idx])))
                                        module___.out_channels = len(flatten_remain_index[idx])
                                        tmp_in_channels = module___.out_channels
                                    elif isinstance(module___, torch.nn.BatchNorm2d):
                                        module___.weight = torch.nn.parameter.Parameter(
                                            torch.index_select(module___.weight, 0,
                                                               torch.tensor(flatten_remain_index[idx])))
                                        module___.bias = torch.nn.parameter.Parameter(
                                            torch.index_select(module___.bias, 0,
                                                               torch.tensor(flatten_remain_index[idx])))
                                        module___.running_mean = torch.index_select(module___.running_mean, 0,
                                                               torch.tensor(flatten_remain_index[idx]))
                                        module___.running_var = torch.index_select(module___.running_var, 0,
                                                               torch.tensor(flatten_remain_index[idx]))
                                        module___.num_features = len(flatten_remain_index[idx])
            elif isinstance(module, torch.nn.Linear):
                module.in_features = tmp_in_channels
                module.weight = torch.nn.parameter.Parameter(torch.index_select(module.weight, 1, torch.tensor(flatten_remain_index[idx])))
        # import pdb; pdb.set_trace()
    def reconstruction_model(self, block, operation='OR'):
        assert block in ['basic', 'bottle']
        self.get_remain_index()
        if block == 'basic':
            self.union_remain_index_basic()
            self.reconstruction_basic()
        elif block == 'bottle' and operation == 'OR':
            print('OR')
            self.union_remain_index_bottle()
            self.reconstruction_bottle()
        elif block == 'bottle' and operation == 'AND':
            print('AND')
            self.and_remain_index_bottle()
            self.reconstruction_bottle()

        return self.model