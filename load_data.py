import numpy as np
import matplotlib.pyplot as plt
import torch

from sklearn.model_selection import train_test_split



class ProcessData():
    def __init__(self, file_path, num_of_file):
        self.file_path = file_path
        self.name_list = ['press_0', 'channel_t_0', 'substrate_t_0', 'rho', 'rhof', 'vel_0', 'vel_1']
        self.num_of_file = num_of_file

    def load_data(self, file_path):
        dic_data_power_fct = {}
        for i in range(self.num_of_file):
            dic_data_power_fct[f"{i}th power map data"] = f'formatted_data{i}/'
        dic_power_map = {}
        dic_of_file = {}
        for key, value in dic_data_power_fct.items():
            for name in self.name_list:
                for i in range(10):
                    main_name = f'results-{i}.h5_'
                    dic_of_file[key + '_' + name + '_' + f'{i}'] = self.file_path + value + main_name + name + '.out'
            dic_power_map[key] = self.file_path + value + 'powermap' + '.out'
        
        return dic_of_file, dic_power_map
                
    def import_data(self, dic_file:dict, per_file:bool):
        """Function to import the data assuming the file as numpy dictionnaries"""
        dic_data ={}
        for key, file in dic_file.items():
            dic_data[key] = np.loadtxt(file)
            
        if per_file: # if you want each file stored separately
            return dic_data
        
        dic_concat = {}
        count = 0
        last_key = 'to_del'
        data = 0
        
        for key, value in dic_data.items():
            if count % 9 == 0:
                try:
                    dic_concat[last_key[:-2]] = data.reshape(-1,100,100)
                except:
                    dic_concat[last_key] = data
                data = value
            data = np.concatenate((data, value), axis=0)
            count += 1
            last_key = key
            
        del dic_concat['to_del']
        
        return dic_concat


    def data_to_tensor(self, 
                    dic_file:dict,
                    dic_pwr_map:dict, 
                    train_test_split : bool = True, 
                    gaussian_normalization: bool = True,
                    grid_boundaries: np.array = [[0, 1], [0, 1]],
                    batch_size: int = 8,
                    num_workers : int = 8,
                    pin_memory : bool = False,
                    persistent_workers :bool = False,
                    #list_of_features : list
                    ):
        
        data_dic = {}
        for j in range(10):
            for i in range(self.num_of_file):
                data_dic[f'{i}'] = dic_pwr_map[f"{i}th power map data"]
                for key in dic_file.keys():
                    if key[0] == f'{i}':
                        try :
                            data_dic[f'{i}'] = np.stack([data_dic[f'{i}'], dic_file[key][j]])
                        except:
                            data_dic[f'{i}'] = np.concatenate([data_dic[f'{i}'], dic_file[key][j].reshape(-1,100,100)])
            data_dic[f'data_point_{j}_power_map_{i}'] = data_dic[f'{i}']#.to(device)
        print(data_dic['data_point_9_power_map_1'].shape)
        channel_t = np.concatenate((data_dic['data_point_9_power_map_1'][1], data_dic['data_point_8_power_map_1'][1], data_dic['data_point_7_power_map_1'][1])).reshape(-1,100,100)
        power_map = np.concatenate((data_dic['data_point_9_power_map_1'][0], data_dic['data_point_8_power_map_1'][0],data_dic['data_point_7_power_map_1'][0])).reshape(-1,100,100)
        print(channel_t.shape)
        print(power_map.shape)
        db = torch.utils.data.TensorDataset(power_map, channel_t)
        
        loader = torch.utils.data.DataLoader(db)
        return loader
        #for key, data in dic_file.items():
            #print(key)
                
            
            #dic_tensor[key] = torch.Tensor(data).to(device)
        
        #to_load = [dic_tensor[key] for key in list_of_features]
        #db = torch.utils.data.TensorDataset()
        #loader = torch.utils.data.DataLoader(db,
        #                                    batch_size=batch_size, shuffle=True, drop_last=True,
        #                                    num_workers=num_workers, pin_memory=pin_memory, persistent_workers=persistent_workers)
            
        return data_dic
    def load_method():
        file_path = "/mnt/c/Users/bonvi/Documents/simulation_hack/simulation_hackaton_eth-rafael/simulation_hackaton_eth-rafael/"
        num_of_files = 2
        process_data = ProcessData(file_path, num_of_files)
        dic, dic_power_map = process_data.load_data(file_path)
        dic_data = process_data.import_data(dic_file = dic, per_file = False)
        dic_power_map_data = process_data.import_data(dic_file = dic_power_map, per_file = True)
        print(dic_power_map_data.keys())
        
        #process_data.plot_data(dic_data)
        tensor_data = process_data.data_to_tensor(dic_data, dic_power_map_data)
        
        # keys are of the form '{i}th power map data_{feature}', 
        print(dic_data['1th power map data_rho'].shape)
        #load_data(dic_file=dic_data)
        
    def plot_data(self, dic_data):
        for data in dic_data.values():
            for i in range(data.shape[0]):
                plt.imshow(data[i], interpolation='none')
                plt.show()


def main():
    file_path = "/mnt/c/Users/bonvi/Documents/simulation_hack/simulation_hackaton_eth-rafael/simulation_hackaton_eth-rafael/"
    num_of_files = 2
    process_data = ProcessData(file_path, num_of_files)
    dic, dic_power_map = process_data.load_data(file_path)
    dic_data = process_data.import_data(dic_file = dic, per_file = False)
    dic_power_map_data = process_data.import_data(dic_file = dic_power_map, per_file = True)
    print(dic_power_map_data.keys())
    
    #process_data.plot_data(dic_data)
    tensor_data = process_data.data_to_tensor(dic_data, dic_power_map_data)
    
    # keys are of the form '{i}th power map data_{feature}', 
    print(dic_data['1th power map data_rho'].shape)
    #load_data(dic_file=dic_data)

if __name__ == "__main__":
    main()
