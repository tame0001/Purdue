import random
import operator
import numpy as np
import matplotlib.pyplot as plt

seed = 0           
random.seed(seed)
np.random.seed(seed)

from ComputationalGraphPrimer import *

class DataLoader:
    '''
    The same data loader class. Just take out of run_training function
    to make that function cleaner. 
    '''
    def __init__(self, training_data, batch_size):
        self.training_data = training_data
        self.batch_size = batch_size
        self.class_0_samples = [(item, 0) for item in self.training_data[0]]   
        self.class_1_samples = [(item, 1) for item in self.training_data[1]]   

    def __len__(self):
        return len(self.training_data[0]) + len(self.training_data[1])

    def _getitem(self):    
        cointoss = random.choice([0,1])                           
                                                                    
        if cointoss == 0:
            return random.choice(self.class_0_samples)
        else:
            return random.choice(self.class_1_samples)            

    def getbatch(self):
        batch_data,batch_labels = [],[]                            
        maxval = 0.0                                               
        for _ in range(self.batch_size):
            item = self._getitem()
            if np.max(item[0]) > maxval: 
                maxval = np.max(item[0])
            batch_data.append(item[0])
            batch_labels.append(item[1])
        batch_data = [item/maxval for item in batch_data]          
        batch = [batch_data, batch_labels]
        return batch     

class OneNeuron(ComputationalGraphPrimer):
    '''
    A modified class for single neuron network. There are options for
    normal SGD, SGD with momentum and Adam.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def initialize_params(self):
        '''
        To make all options start with the same initization.
        '''
        self.learnable_params_original = {
            param: random.uniform(0,1) for param in self.learnable_params
        }
        self.bias_original = random.uniform(0,1)  

    def run_training(self, data_loader, mu=None, beta=None):
        '''
        For normal SGD, no additional parameters need.
        For SGD+, pass mu in range [0, 1] as an interger.
        For Adam, pass a list of [beta1, beta2].
        '''
        # Re-assign the learnable parameters
        # Therefore, all options start with the same random params
        self.vals_for_learnable_params = self.learnable_params_original.copy()
        self.bias = self.bias_original

        if mu is not None: # For SGD+, create placeholder for previous step
            self.step_prev = {param: 0 for param in self.learnable_params}
            self.step_prev.update({'bias': 0})

        if beta is not None: # For Adam, create placeholder for momentum
            self.moment_prev = \
                {param: [0, 0] for param in self.learnable_params}
            self.moment_prev.update({'bias': [0, 0]})
            self.correction_factor = beta.copy() # For beta power to iteration

        # Copy from original class. Just rewritten to fit 80 characters per line
        loss_running_record = []
        avg_loss_over_iterations = 0.0       
        for i in range(self.training_iterations): 
            data = data_loader.getbatch()
            data_tuples = data[0]
            class_labels = data[1]
            y_preds, deriv = self.forward_prop_one_neuron_model(data_tuples)
            loss = sum([
                (abs(class_labels[i] - y_preds[i]))**2 
                for i in range(len(class_labels))
            ])
            loss_avg = loss / float(len(class_labels))
            avg_loss_over_iterations += loss_avg     
            if i%(self.display_loss_how_often) == 0: 
                avg_loss_over_iterations /= self.display_loss_how_often
                loss_running_record.append(avg_loss_over_iterations)
                print(f"[iter={i+1:>5}]  loss = {avg_loss_over_iterations:.4f}")
                avg_loss_over_iterations = 0.0 
            y_errors = list(map(operator.sub, class_labels, y_preds))
            y_error_avg = sum(y_errors) / float(len(class_labels))
            deriv_sigmoid_avg = sum(deriv) / float(len(class_labels))
            data_tuple_avg = [sum(x) for x in zip(*data_tuples)]
            data_tuple_avg = list(map(
                operator.truediv, 
                data_tuple_avg, 
                [float(len(class_labels))] * len(class_labels) 
            ))
            # Modified backprop to fit 3 options
            if mu is not None: # SGD+
                self.backprop_sgd(y_error_avg, data_tuple_avg, 
                                  deriv_sigmoid_avg, mu)
            elif beta is not None: # Adam
                self.backprop_adam(y_error_avg, data_tuple_avg,
                                   deriv_sigmoid_avg, beta)
            else: # SGD
                self.backprop_sgd(y_error_avg, data_tuple_avg, 
                                  deriv_sigmoid_avg)
            
        return loss_running_record
    
    def backprop_sgd(self, y_error, vals_for_input_vars, 
                     deriv_sigmoid, mu=None):
        '''
        Copy from the original class. Only modified for SGD and SGD+.
        And refactor to fit 80 characters per line. 
        '''
        input_vars = self.independent_vars
        vals_for_input_vars_dict =  dict(zip(
            input_vars, list(vals_for_input_vars)
        ))
        for i, param in enumerate(self.vals_for_learnable_params):
            gradient = (y_error # Calculate gradient
                        * vals_for_input_vars_dict[input_vars[i]]
                        * deriv_sigmoid)   
            if mu is not None: # For SGD+
                gradient += mu * self.step_prev[param]
                self.step_prev[param] = gradient
            # Update param value
            step = self.learning_rate * gradient
            self.vals_for_learnable_params[param] += step 
        # Also update bias
        gradient_bias =  y_error * deriv_sigmoid 
        if mu:
            gradient_bias += self.step_prev['bias'] * mu
            self.step_prev['bias'] = gradient_bias
        self.bias += self.learning_rate * gradient_bias

    def backprop_adam(self, y_error, vals_for_input_vars, deriv_sigmoid, beta):
        '''
        Copy from the original class. Only modified for Adam.
        And refactor to fit 80 characters per line. 
        '''
        input_vars = self.independent_vars
        vals_for_input_vars_dict =  dict(zip(
            input_vars, list(vals_for_input_vars)
        ))
        for i, param in enumerate(self.vals_for_learnable_params):
            gradient = (y_error # Calculate gradient
                        * vals_for_input_vars_dict[input_vars[i]]
                        * deriv_sigmoid)    
            momentum_first = (beta[0] * self.moment_prev[param][0]
                              + (1-beta[0]) * gradient)
            momentum_second = (beta[1] * self.moment_prev[param][1]
                              + (1-beta[1]) * gradient**2)
            self.moment_prev[param] = [momentum_first, momentum_second]
            momentum_first /= 1 - self.correction_factor[0]
            momentum_second /= 1 - self.correction_factor[1]
            # Update param value
            self.vals_for_learnable_params[param] += (
                self.learning_rate 
                * momentum_first 
                / (np.sqrt(momentum_second) + 1e-8))            
        # Also update bias
        gradient_bias = y_error * deriv_sigmoid    
        momentum_first = (beta[0] * self.moment_prev['bias'][0]
                          + (1-beta[0]) * gradient_bias)
        momentum_second = (beta[1] * self.moment_prev['bias'][1]
                          + (1-beta[1]) * (gradient_bias**2))
        self.moment_prev['bias'] = [momentum_first, momentum_second]
        momentum_first /= 1 - self.correction_factor[0]
        momentum_second /= 1 - self.correction_factor[1]
        self.bias += (self.learning_rate
                      * momentum_first
                      / (np.sqrt(momentum_second)+1e-8))

        # Update correction factors
        self.correction_factor = [
            self.correction_factor[0]*beta[0], 
            self.correction_factor[1]*beta[1]
        ]   
                                                        

cgp = OneNeuron( # Create instance for single-neuron model
    one_neuron_model = True,
    expressions = ['xw=ab*xa+bc*xb+cd*xc+ac*xd'],
    output_vars = ['xw'],
    dataset_size = 5000,
    learning_rate = 1e-4,
    training_iterations = 40000,
    batch_size = 8,
    display_loss_how_often = 100,
    debug = True,
)

cgp.parse_expressions()      
cgp.initialize_params() # Initial params for all options
training_data = cgp.gen_training_data() # Shared training set for all options
# Pass data loader instead of training set
data_loader = DataLoader(training_data, batch_size=cgp.batch_size)

fig, ax = plt.subplots(figsize=(8,4))

sgd_loss = cgp.run_training(data_loader) # Normal SGD
ax.plot(sgd_loss, label='SGD')

for mu in [0.25, 0.75]: # SGD+
    sgd_plus_loss = cgp.run_training(data_loader, mu)
    ax.plot(sgd_plus_loss, label=f'SGD+ \u03BC={mu}')

sdg_adam_loss = cgp.run_training(data_loader, beta=[0.9, 0.999]) # Adam
ax.plot(sdg_adam_loss, label=f'Adam')

ax.set_xlabel('Iteration (in hundreds)')
ax.set_ylabel('Average Traning Loss')
ax.legend(loc='best')
ax.set_title(f'Training Loss vs Iteration ' 
    f'when Learning Rate = {cgp.learning_rate:.0e}')
plt.show()