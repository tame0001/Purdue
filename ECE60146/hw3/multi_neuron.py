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

class MultiNeuron(ComputationalGraphPrimer):
    '''
    A modified class for mutiple neuron network. There are options for
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
        self.bias_original = [
            random.uniform(0,1) for _ in range(self.num_layers-1)
        ] 

    def run_training(self, data_loader, mu=None, beta=None):
        '''
        For normal SGD, no additional parameters need.
        For SGD+, pass mu in range [0, 1] as an interger.
        For Adam, pass a list of [beta1, beta2].
        '''
        # Re-assign the learnable parameters
        # Therefore, all options start with the same random params
        self.vals_for_learnable_params = self.learnable_params_original.copy()
        self.bias = self.bias_original.copy() 

        if mu is not None: # For SGD+, create placeholder for previous step
            self.step_prev = {param: 0 for param in self.learnable_params}
            self.bias_prev = [0 for _ in range(self.num_layers-1)]    

        if beta is not None: # For Adam, create placeholder for momentum
            self.moment_prev = {
                param: [0, 0] for param in self.learnable_params
            }
            self.bias_prev = [[0, 0] for _ in range(self.num_layers-1)] 
            self.correction_factor = beta.copy() # For beta power to iteration

        # Copy from original class. Just rewritten to fit 80 characters per line                                                                     
        loss_running_record = []
        avg_loss_over_iterations = 0.0                                                                                                                    
        for i in range(self.training_iterations):
            data = data_loader.getbatch()
            data_tuples = data[0]
            class_labels = data[1]
            self.forward_prop_multi_neuron_model(data_tuples)                                  
            pred_labels = self.forw_prop_vals_at_layers[self.num_layers-1]     
            y_preds =  [item for sublist in  pred_labels  for item in sublist]  
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

            # Modified backprop to fit 3 options
            if mu is not None: # SGD+
                self.backprop(y_error_avg, class_labels, mu)
            elif beta is not None: # Adam
                self.backprop(y_error_avg, class_labels, mu=None, beta=beta)
            else: # SGD
                self.backprop(y_error_avg, class_labels)     

        return loss_running_record   

    def backprop(self, y_error, class_labels, mu=None, beta=None):
        '''
        Copy from the original class. Only modified the very end of this method.
        And refactor to fit 80 characters per line. 
        '''
        pred_err_backproped = {i : [] for i in range(1, self.num_layers-1)}  
        pred_err_backproped[self.num_layers-1] = [y_error]
        for layer_index in reversed(range(1, self.num_layers)):
            input_vals = self.forw_prop_vals_at_layers[layer_index -1]
            input_vals_avg = [sum(x) for x in zip(*input_vals)]
            input_vals_avg = list(map(
                operator.truediv, 
                input_vals_avg, 
                [float(len(class_labels))] * len(class_labels)
            ))
            deriv_sigmoid = self.gradient_vals_for_layers[layer_index]
            deriv_sigmoid_avg = [sum(x) for x in zip(*deriv_sigmoid)]
            deriv_sigmoid_avg = list(map(
                operator.truediv, 
                deriv_sigmoid_avg, 
                [float(len(class_labels))] * len(class_labels)
            ))
            vars_in_layer = self.layer_vars[layer_index]                 
            vars_in_next_layer_back = self.layer_vars[layer_index - 1] 
            layer_params = self.layer_params[layer_index]         
            layer_params_t = list(zip(*layer_params))         
            backproped_error = [None] * len(vars_in_next_layer_back)
            for k, _ in enumerate(vars_in_next_layer_back):
                for j, _ in enumerate(vars_in_layer):
                    backproped_error[k] = sum([
                        self.vals_for_learnable_params[layer_params_t[k][i]] 
                        * pred_err_backproped[layer_index][i] 
                        for i in range(len(vars_in_layer))
                    ])                                           
            pred_err_backproped[layer_index-1]  =  backproped_error
            for j, _ in enumerate(vars_in_layer):
                layer_params = self.layer_params[layer_index][j]
                for i, param in enumerate(layer_params):
                    # Modified update params part
                    gradient_param = ( # calculate gradient for current param
                        input_vals_avg[i] 
                        * pred_err_backproped[layer_index][j]
                        * deriv_sigmoid_avg[j]
                    ) 
                    if mu is not None: # For SGD+
                        # Add effect of momentum
                        gradient_param += mu * self.step_prev[param]
                        # Keep step for the next iteration
                        self.step_prev[param] = gradient_param

                    if beta is not None: # For Adam
                        # Calculate momentum
                        momentum_first = (beta[0] * self.moment_prev[param][0] 
                                          + (1 - beta[0]) * gradient_param)
                        momentum_second = (beta[1] * self.moment_prev[param][1]
                                           + (1 - beta[1]) * gradient_param**2)
                        # Keep momentum for the iteration
                        self.moment_prev[param] = [ 
                            momentum_first, 
                            momentum_second
                        ]
                        # Correct momentum values
                        momentum_first /= 1 - self.correction_factor[0]
                        momentum_second /= 1 - self.correction_factor[1]
                        gradient_param += (momentum_first
                                          / (np.sqrt(momentum_second) + 1e-8))
                    
                    # Apply learning rate and update param
                    step = self.learning_rate * gradient_param 
                    self.vals_for_learnable_params[param] += step

            gradient_bias = ( # Calculate gradient for bias
                sum(pred_err_backproped[layer_index])
                * sum(deriv_sigmoid_avg)
                / len(deriv_sigmoid_avg)
            )
            if mu is not None: # For SGD+
                gradient_bias += mu * self.bias_prev[layer_index-1]
                self.bias_prev[layer_index-1] = gradient_bias

            if beta is not None: # For Adam
                # Calculate momentum
                momentum_first = (beta[0] * self.moment_prev[param][0] 
                                 + (1 - beta[0]) * gradient_bias)
                momentum_second = (beta[1] * self.moment_prev[param][1]
                                  + (1 - beta[1]) *gradient_bias**2)
                # Keep momentum for the iteration
                self.bias_prev[layer_index-1] = [
                    momentum_first, 
                    momentum_second
                ]
                # Correct momentum values
                momentum_first /= 1 - self.correction_factor[0]
                momentum_second /= 1 - self.correction_factor[1]
                gradient_bias += (momentum_first
                                  / (np.sqrt(momentum_second) + 1e-8))
            # Apply learning rate and update param                      
            self.bias[layer_index-1] += self.learning_rate * gradient_bias

        if beta is not None: # Update correction factors
            self.correction_factor = [
                self.correction_factor[0]*beta[0], 
                self.correction_factor[1]*beta[1]
            ]   



cgp = MultiNeuron( # Create instance for multi-neuron model
    num_layers = 3,
    layers_config = [4,2,1],                         
    expressions = [
        'xw=ap*xp+aq*xq+ar*xr+as*xs',
        'xz=bp*xp+bq*xq+br*xr+bs*xs',
        'xo=cp*xw+cq*xz'
    ],
    output_vars = ['xo'],
    dataset_size = 5000,
    learning_rate = 1e-4,
    training_iterations = 40000,
    batch_size = 8,
    display_loss_how_often = 100,
    debug = True,
)

cgp.parse_multi_layer_expressions()  
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
ax.set_title(
    f'Multi-Neuron Training Loss vs Iteration ' 
    f'when Learning Rate = {cgp.learning_rate:.0e}'
)
plt.show()