from copy import deepcopy
import numpy as np
from scipy.stats import truncnorm
import pickle
# if 'google.colab' in str(get_ipython()):
#     from MPEAsAlloyPublic.model_paths import models
# else:
#     from model_paths import models
import joblib
import pickle
import warnings
warnings.filterwarnings('ignore')

model_dir = "/content/drive/MyDrive/MPEAs"

models = {"Current density": joblib.load(f"{model_dir}/RF_icorr"),
          "Corrosion potential": joblib.load(f"{model_dir}/RF_Ecorr"),
          "Pitting potential": joblib.load(f"{model_dir}/RF_Pcorr")
          }



class MPEAsDatapoint:
    def __init__(self, settings):
        self.categorical_inputs = settings.categorical_inputs
        self.categorical_inputs_info = settings.categorical_inputs_info
        self.range_based_inputs = settings.range_based_inputs
        self.concentration_inputs = settings.concentration_inputs
        self.al_balance = True

    def formatForInput(self):
        #ht = [1 if [i+1] in [*self.categorical_inputs.values()] else 0 for i in range(6)]
        for key in self.categorical_inputs:
            elec = self.categorical_inputs[key]
        if sum([*self.range_based_inputs.values()]) != 100.0:
            self.al_balance = False
            

        my_input = [sum(*self.concentration_inputs.values())] + elec + [100 - sum([*self.range_based_inputs.values()][1:])] + [*self.range_based_inputs.values()][1:] 
                   # [*self.range_based_inputs.values()] 
        
        
        return np.reshape(my_input, (1, -1))

    def print(self):
        for key, value in self.categorical_inputs.items():
            print(f"{key}: {self.categorical_inputs_info[key]['tag'][self.categorical_inputs_info[key]['span'].index(value)]}")
        #print(f"Mg%: {round(self.getMg(), 2)}")
        for key, value in self.range_based_inputs.items():
            if value:
                print(f"{key}: {value}")
        print("Electrolyte concentration: " + self.concentration)

    #def getMg(self):
        #return 100 - sum(sum(row) for row in list(self.range_based_inputs.values())[1:])
                


class scanSettings:
    def __init__(self, mode):
        self.mode = mode

        if self.mode == 'DoS':
            self.loss_type = 'Linear'
            self.max_steps = 1
            self.targets = {
                'DoS': 10
            }
            self.categorical_inputs = {
                'Electrolyte': [1]
            }
            self.categorical_inputs_info = {
                'Electrolyte': {'span': [1, 2, 3, 4, 5, 6,7], 'tag': ['H2SO4', 'HCl','HNO3', 'KOH', 'NaCl','NaOH', 'Seawater']}}
            
            self.range_based_inputs = dict.fromkeys(
                ['Al', 'B', 'Be', 'Co', 'Cr',
       'Cu', 'Fe', 'Ga', 'Hf', 'La', 'Mg', 'Mn', 'Mo', 'Nb', 'Ni', 'Si', 'Sn',
       'Ta', 'Ti', 'V', 'W', 'Y', 'Zn', 'Zr'], [0])

            self.concentration_inputs = dict.fromkeys(['Concentration in M'], 0.6)


        if self.mode == 'Corrosion':
            self.loss_type = 'Percentage'
            self.max_steps = 1
            self.targets = {
                'Current density': 100,
                'Corrosion potential': -350,
                'Pitting potential': 250
            }
            self.categorical_inputs = {
                'Electrolyte': [1]
            }
            self.categorical_inputs_info = {
                'Electrolyte': {'span': [1, 2, 3, 4, 5, 6,7], 'tag': ['H2SO4', 'HCl','HNO3', 'KOH', 'NaCl','NaOH', 'Seawater']}}
            
#             self.range_based_inputs = dict(zip(
#                 ['Mg', 'Nd', 'Ce', 'La', 'Zn', 'Sn', 'Al', 'Ca', 'Zr', 'Ag', 'Ho', 'Mn',
#                  'Y', 'Gd', 'Cu', 'Si', 'Li', 'Yb', 'Th', 'Sb', 'Pr', 'Ga', 'Be', 'Fe',
#                  'Ni', 'Sc', 'Tb', 'Dy', 'Er', 'Sr', 'Bi'],
#                 [[0.827], [0.0026], [0], [0], [0], [0], [0.065], [0.0945], [0],
#                  [0], [0], [0], [0], [0], [0], [0.0076],
#                  [0], [0], [0], [0], [0], [0], [0], 
#                  [0], [0], [0], [0], [0], [0.0032], [0], [0]]))
            
            self.range_based_inputs = dict(zip(
                ['Al', 'B', 'Be', 'Co', 'Cr',
       'Cu', 'Fe', 'Ga', 'Hf', 'La', 'Mg', 'Mn', 'Mo', 'Nb', 'Ni', 'Si', 'Sn',
       'Ta', 'Ti', 'V', 'W', 'Y', 'Zn', 'Zr'],
                [[100], [0] , [0] , [0] ,
                    [0] , [0] , [0] , [0] , [0],
                    [0] , [0] , [0] , [0] , [0], 
                    [0] , [0] , [0] , [0] , [0], 
                    [0] , [0] , [0] , [0] , [0]]))
        
            self.range_based_inputs['Al'] = [100 - sum(sum(row) for row in list(self.range_based_inputs.values())[1:])]
            self.concentration_inputs = dict.fromkeys(['Concentration in M'], [0.6])
            
        
        
            


class optimiser:
    def __init__(self, settings):
        self.step_batch_size = 100
        self.step_final_std = 0.01
        self.finetune_max_rounds = 3
        self.finetune_batch_size = 10
        self.mode = settings.mode
        self.loss_type = settings.loss_type
        self.targets = settings.targets
        self.max_steps = settings.max_steps
        self.categorical_inputs = settings.categorical_inputs
        self.range_based_inputs = settings.range_based_inputs
        self.concentration_inputs = settings.concentration_inputs
        self.settings = settings
        self.models = models

        self.run()

    def calculateLoss(self, datapoint):
        if self.mode == 'DoS':
            return self.models['Current density'].predict(datapoint.formatForInput())[0]
        elif self.mode == 'Corrosion':
            return self.models['Current density'].predict(datapoint.formatForInput())[0]

    def printResults(self, best_datapoint):
        if self.mode == 'DoS':
            print('data point:',best_datapoint.formatForInput()) 
            #print('predicted %f Elongation' % (1.25*self.models['elongation'].predict(best_datapoint.formatForInput())[0]))
            #print('predicted %f Yield Strength' % (1.25*self.models['yield'].predict(best_datapoint.formatForInput())[0]))
            #print('predicted %f Tensile Strength' % (1.25*self.models['tensile'].predict(best_datapoint.formatForInput())[0]))
        elif self.mode == 'Corrosion':
            final_alloy  = dict(zip(
                ['Concentration in M','H2SO4', 'HCl','HNO3', 'KOH', 'NaCl','NaOH', 'Seawater','Al', 'B', 'Be', 'Co', 'Cr',
       'Cu', 'Fe', 'Ga', 'Hf', 'La', 'Mg', 'Mn', 'Mo', 'Nb', 'Ni', 'Si', 'Sn',
       'Ta', 'Ti', 'V', 'W', 'Y', 'Zn', 'Zr'],
                best_datapoint.formatForInput().reshape(-1,)))
            
            if not best_datapoint.al_balance:
                print()
                print('\033[1m'+'\033[91m'+ "Al content has been balanced to "+ str(final_alloy['Al']) + " %" +'\033[0m')
            
            print()
            print('Chemical composition: ')
            for index, key in enumerate(final_alloy):
                print(key+ ":" + str(final_alloy[key]), end="  ")
                if (index+1)%10 ==0:
                    print("")
                  
                
            print('\n')
            print('Predicted %f Current density' % (self.models['RF_icorr'].predict(best_datapoint.formatForInput())[0]))
            print('Predicted %f Corrosion potential' % (self.models['RF_Ecorr'].predict(best_datapoint.formatForInput())[0]))
            print('Predicted %f Pitting potential' % (self.models['RF_Pcorr'].predict(best_datapoint.formatForInput())[0]))
            print()
            print('=============================================')
            print()

    def run(self):
        best_loss = None
        best_datapoint = MPEAsDatapoint(self.settings)
        for key in self.range_based_inputs.keys():
            best_datapoint.range_based_inputs[key] = min(self.range_based_inputs[key])
        
        #for i in range(self.max_steps):
          #  loss, datapoint = self.calculateStep(best_datapoint, i, 'all')
           # if best_loss is None or loss < best_loss:
            #    best_datapoint = datapoint
             #   best_loss = loss

       # for i in range(self.finetune_max_rounds):
          #  for key in [*self.categorical_inputs.keys(), *self.range_based_inputs.keys()]:
             #   loss, datapoint = self.calculateStep(best_datapoint, i, key)
              #  if loss < best_loss:
               #     best_datapoint = datapoint
                #    best_loss = loss
          #  else:
           #     break
        print('=============== Scan Finished ===============')
        self.printResults(best_datapoint)

#     def calculateStep(self, best_datapoint, step_number, target_var):
#         if target_var == 'all':
#             batch_size = self.step_batch_size
#         else:
#             batch_size = self.finetune_batch_size
#         loss = [0] * batch_size
#         datapoints = []
#         std = self.step_final_std * (self.max_steps / float(step_number + 1))
#         for i in range(batch_size):
#             datapoints.append(deepcopy(best_datapoint))
#             for key in self.categorical_inputs.keys():
#                 if target_var == key or target_var == 'all':
#                     datapoints[i].categorical_inputs[key] = np.random.choice(self.categorical_inputs[key])
#             for key in self.range_based_inputs.keys():
#                 if target_var == key or target_var == 'all':
#                     if max(self.range_based_inputs[key]) != min(self.range_based_inputs[key]):
#                         a = (min(self.range_based_inputs[key]) - np.mean(best_datapoint.range_based_inputs[key])) / std
#                         b = (max(self.range_based_inputs[key]) - np.mean(best_datapoint.range_based_inputs[key])) / std
#                         datapoints[i].range_based_inputs[key] = round(
#                             float(truncnorm.rvs(a, b, loc=np.mean(best_datapoint.range_based_inputs[key]), scale=std)),
#                             2)
#                     else:
#                         datapoints[i].range_based_inputs[key] = min(self.range_based_inputs[key])
#             loss[i] = self.calculateLoss(datapoints[i])
#         return min(loss), datapoints[loss.index(min(loss))]
