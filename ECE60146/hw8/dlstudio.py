'''
Modified DLstudio. Only focus on TextClassificationWithEmbeddings class.
Change log (reference to DL Studio 2.2.5)
Line 5231:
- Original: review_tensor = torch.FloatTensor(list_of_embeddings)
- New: review_tensor = torch.FloatTensor(np.array(list_of_embeddings))
- Note: Convert to numpy array before converting to torch tensor
'''

import sys,os,os.path
import torch
import torch.nn as nn
import torchvision                  
import torchvision.transforms as tvt
import torch.optim as optim
import numpy as np
import re
import random
import copy
import matplotlib.pyplot as plt
import gzip
import pickle
import time
import logging

#______________________________  DLStudio Class Definition  ________________________________

class DLStudio(object):

    def __init__(self, *args, **kwargs ):
        if args:
            raise ValueError(  
                   '''DLStudio constructor can only be called with keyword arguments for 
                      the following keywords: epochs, learning_rate, batch_size, momentum,
                      convo_layers_config, image_size, dataroot, path_saved_model, classes, 
                      image_size, convo_layers_config, fc_layers_config, debug_train, use_gpu, and 
                      debug_test''')
        learning_rate = epochs = batch_size = convo_layers_config = momentum = None
        image_size = fc_layers_config = dataroot =  path_saved_model = classes = use_gpu = None
        debug_train  = debug_test = None
        if 'dataroot' in kwargs                      :   dataroot = kwargs.pop('dataroot')
        if 'learning_rate' in kwargs                 :   learning_rate = kwargs.pop('learning_rate')
        if 'momentum' in kwargs                      :   momentum = kwargs.pop('momentum')
        if 'epochs' in kwargs                        :   epochs = kwargs.pop('epochs')
        if 'batch_size' in kwargs                    :   batch_size = kwargs.pop('batch_size')
        if 'convo_layers_config' in kwargs           :   convo_layers_config = kwargs.pop('convo_layers_config')
        if 'image_size' in kwargs                    :   image_size = kwargs.pop('image_size')
        if 'fc_layers_config' in kwargs              :   fc_layers_config = kwargs.pop('fc_layers_config')
        if 'path_saved_model' in kwargs              :   path_saved_model = kwargs.pop('path_saved_model')
        if 'classes' in kwargs                       :   classes = kwargs.pop('classes') 
        if 'use_gpu' in kwargs                       :   use_gpu = kwargs.pop('use_gpu') 
        if 'debug_train' in kwargs                   :   debug_train = kwargs.pop('debug_train') 
        if 'debug_test' in kwargs                    :   debug_test = kwargs.pop('debug_test') 
        if len(kwargs) != 0: raise ValueError('''You have provided unrecognizable keyword args''')
        if dataroot:
            self.dataroot = dataroot
        if convo_layers_config:
            self.convo_layers_config = convo_layers_config
        if image_size:
            self.image_size = image_size
        if fc_layers_config:
            self.fc_layers_config = fc_layers_config
#            if fc_layers_config[0] is not -1:
            if fc_layers_config[0] != -1:
                raise Exception("""\n\n\nYour 'fc_layers_config' construction option is not correct. """
                                """The first element of the list of nodes in the fc layer must be -1 """
                                """because the input to fc will be set automatically to the size of """
                                """the final activation volume of the convolutional part of the network""")
        if  path_saved_model:
            self.path_saved_model = path_saved_model
        if classes:
            self.class_labels = classes
        if learning_rate:
            self.learning_rate = learning_rate
        else:
            self.learning_rate = 1e-6
        if momentum:
            self.momentum = momentum
        if epochs:
            self.epochs = epochs
        if batch_size:
            self.batch_size = batch_size
        if use_gpu is not None:
            self.use_gpu = use_gpu
            if use_gpu is True:
                if torch.cuda.is_available():
                    self.device = torch.device("cuda:0")
                else:
                    raise Exception("You requested GPU support, but there's no GPU on this machine")
        else:
            self.device = torch.device("cpu")
        if debug_train:                             
            self.debug_train = debug_train
        else:
            self.debug_train = 0
        if debug_test:                             
            self.debug_test = debug_test
        else:
            self.debug_test = 0
        self.debug_config = 0
#        self.device = torch.device("cuda:0" if torch.cuda.is_available() and self.use_gpu is False else "cpu")

    class Net(nn.Module):
        def __init__(self, convo_layers, fc_layers):
            super(DLStudio.Net, self).__init__()
            self.my_modules_convo = convo_layers
            self.my_modules_fc = fc_layers
        def forward(self, x):
            for m in self.my_modules_convo:
                x = m(x)
            x = x.view(x.shape[0], -1)
            for m in self.my_modules_fc:
                x = m(x)
            return x


    def run_code_for_training(self, net, display_images=False):        
        filename_for_out = "performance_numbers_" + str(self.epochs) + ".txt"
        FILE = open(filename_for_out, 'w')
        net = copy.deepcopy(net)
        net = net.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=self.learning_rate, momentum=self.momentum)
        print("\n\nStarting training loop...")
        start_time = time.perf_counter()
        loss_tally = []
        elapsed_time = 0.0
        for epoch in range(self.epochs):  
            print("")
            running_loss = 0.0
            for i, data in enumerate(self.train_data_loader):
                inputs, labels = data
                if i % 1000 == 999:
                    current_time = time.perf_counter()
                    elapsed_time = current_time - start_time 
                    print("\n\n[epoch:%d/%d  iter=%4d  elapsed_time=%5d secs]   Ground Truth:     " % 
                          (epoch+1, self.epochs, i+1, elapsed_time) + 
                          ' '.join('%10s' % self.class_labels[labels[j]] for j in range(self.batch_size)))
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                ##  Since PyTorch likes to construct dynamic computational graphs, we need to
                ##  zero out the previously calculated gradients for the learnable parameters:
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                if i % 1000 == 999:
                    _, predicted = torch.max(outputs.data, 1)
                    print("[epoch:%d/%d  iter=%4d  elapsed_time=%5d secs]   Predicted Labels: " % 
                     (epoch+1, self.epochs, i+1, elapsed_time ) +
                     ' '.join('%10s' % self.class_labels[predicted[j]] for j in range(self.batch_size)))
                    avg_loss = running_loss / float(2000)
                    loss_tally.append(avg_loss)
                    print("[epoch:%d/%d  iter=%4d  elapsed_time=%5d secs]   Loss: %.3f" % 
                                                                   (epoch+1, self.epochs, i+1, elapsed_time, avg_loss))    
                    FILE.write("%.3f\n" % avg_loss)
                    FILE.flush()
                    running_loss = 0.0
                    if display_images:
                        logger = logging.getLogger()
                        old_level = logger.level
                        logger.setLevel(100)
                        plt.figure(figsize=[6,3])
                        plt.imshow(np.transpose(torchvision.utils.make_grid(inputs, 
                                                            normalize=False, padding=3, pad_value=255).cpu(), (1,2,0)))
                        plt.show()
                        logger.setLevel(old_level)
                loss.backward()
                optimizer.step()
        print("\nFinished Training\n")
        self.save_model(net)
        plt.figure(figsize=(10,5))
        plt.title("Labeling Loss vs. Iterations")
        plt.plot(loss_tally)
        plt.xlabel("iterations")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig("playing_with_skips_loss.png")
        plt.show()

    def save_model(self, model):
        '''
        Save the trained model to a disk file
        '''
        torch.save(model.state_dict(), self.path_saved_model)


    def run_code_for_testing(self, net, display_images=False):
        net.load_state_dict(torch.load(self.path_saved_model))
        net = net.eval()
        net = net.to(self.device)
        ##  In what follows, in addition to determining the predicted label for each test
        ##  image, we will also compute some stats to measure the overall performance of
        ##  the trained network.  This we will do in two different ways: For each class,
        ##  we will measure how frequently the network predicts the correct labels.  In
        ##  addition, we will compute the confusion matrix for the predictions.
        filename_for_results = "classification_results_" + str(self.epochs) + ".txt"
        FILE = open(filename_for_results, 'w')
        correct = 0
        total = 0
        confusion_matrix = torch.zeros(len(self.class_labels), len(self.class_labels))
        class_correct = [0] * len(self.class_labels)
        class_total = [0] * len(self.class_labels)
        with torch.no_grad():
            for i,data in enumerate(self.test_data_loader):
                ##  data is set to the images and the labels for one batch at a time:
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                if i % 1000 == 999:
                    print("\n\n[i=%d:] Ground Truth:     " % (i+1) + ' '.join('%5s' % self.class_labels[labels[j]] 
                                                                   for j in range(self.batch_size)))
                outputs = net(images)
                ##  max() returns two things: the max value and its index in the 10 element
                ##  output vector.  We are only interested in the index --- since that is 
                ##  essentially the predicted class label:
                _, predicted = torch.max(outputs.data, 1)#
#                if display_images and i % 1000 == 999:
                if i % 1000 == 999:
                    print("[i=%d:] Predicted Labels: " % (i+1) + ' '.join('%5s' % self.class_labels[predicted[j]]
                                                              for j in range(self.batch_size)))
                    logger = logging.getLogger()
                    old_level = logger.level
                    if display_images:
                        logger.setLevel(100)
                        plt.figure(figsize=[6,3])
                        plt.imshow(np.transpose(torchvision.utils.make_grid(images,
                                                      normalize=False, padding=3, pad_value=255).cpu(), (1,2,0)))
                        plt.show()
                        logger.setLevel(old_level)
                for label,prediction in zip(labels,predicted):
                        confusion_matrix[label][prediction] += 1
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                ##  comp is a list of size batch_size of "True" and "False" vals
                comp = predicted == labels       
                for j in range(self.batch_size):
                    label = labels[j]
                    ##  The following works because, in a numeric context, the boolean value
                    ##  "False" is the same as number 0 and the boolean value True is the 
                    ##  same as number 1. For that reason "4 + True" will evaluate to 5 and
                    ##  "4 + False" will evaluate to 4.  Also, "1 == True" evaluates to "True"
                    ##  "1 == False" evaluates to "False".  However, note that "1 is True" 
                    ##  evaluates to "False" because the operator "is" does not provide a 
                    ##  numeric context for "True". And so on.  In the statement that follows,
                    ##  while  c[j].item() will either return "False" or "True", for the 
                    ##  addition operator, Python will use the values 0 and 1 instead.
                    class_correct[label] += comp[j].item()
                    class_total[label] += 1
        for j in range(len(self.class_labels)):
            print('Prediction accuracy for %5s : %2d %%' % (self.class_labels[j], 100 * class_correct[j] / class_total[j]))
            FILE.write('\n\nPrediction accuracy for %5s : %2d %%\n' % (self.class_labels[j], 100 * class_correct[j] / class_total[j]))
        print("\n\n\nOverall accuracy of the network on the 10000 test images: %d %%" % (100 * correct / float(total)))
        FILE.write("\n\n\nOverall accuracy of the network on the 10000 test images: %d %%\n" % (100 * correct / float(total)))
        print("\n\nDisplaying the confusion matrix:\n")
        FILE.write("\n\nDisplaying the confusion matrix:\n\n")
        out_str = "         "
        for j in range(len(self.class_labels)):  out_str +=  "%7s" % self.class_labels[j]   
        print(out_str + "\n")
        FILE.write(out_str + "\n\n")
        for i,label in enumerate(self.class_labels):
            out_percents = [100 * confusion_matrix[i,j] / float(class_total[i]) 
                                                      for j in range(len(self.class_labels))]
            out_percents = ["%.2f" % item.item() for item in out_percents]
            out_str = "%6s:  " % self.class_labels[i]
            for j in range(len(self.class_labels)): out_str +=  "%7s" % out_percents[j]
            print(out_str)
            FILE.write(out_str + "\n")
        FILE.close()        

    ###%%%
    ########################################################################################
    ########  Start Definition of Inner Class TextClassificationWithEmbeddings  ############

    class TextClassificationWithEmbeddings(nn.Module):             
        """
        The text processing class described previously, TextClassification, was based on
        using one-hot vectors for representing the words.  The main challenge we faced
        with one-hot vectors was that the larger the size of the training dataset, the
        larger the size of the vocabulary, and, therefore, the larger the size of the
        one-hot vectors.  The increase in the size of the one-hot vectors led to a
        model with a significantly larger number of learnable parameters --- and, that,
        in turn, created a need for a still larger training dataset.  Sounds like a classic
        example of a vicious circle.  In this section, I use the idea of word embeddings
        to break out of this vicious circle.

        Word embeddings are fixed-sized numerical representations for words that are
        learned on the basis of the similarity of word contexts.  The original and still
        the most famous of these representations are known as the word2vec
        embeddings. The embeddings that I use in this section consist of pre-trained
        300-element word vectors for 3 million words and phrases as learned from Google
        News reports.  I access these embeddings through the popular Gensim library.
 
        Class Path:  DLStudio -> TextClassificationWithEmbeddings
        """
        def __init__(self, dl_studio,dataserver_train=None,dataserver_test=None,dataset_file_train=None,dataset_file_test=None):
            super(DLStudio.TextClassificationWithEmbeddings, self).__init__()
            self.dl_studio = dl_studio
            self.dataserver_train = dataserver_train
            self.dataserver_test = dataserver_test

        class SentimentAnalysisDataset(torch.utils.data.Dataset):
            """
            In relation to the SentimentAnalysisDataset defined for the TextClassification section of 
            DLStudio, the __getitem__() method of the dataloader must now fetch the embeddings from
            the word2vec word vectors.

            Class Path:  DLStudio -> TextClassificationWithEmbeddings -> SentimentAnalysisDataset
            """
            def __init__(self, dl_studio, train_or_test, dataset_file, path_to_saved_embeddings=None):
                super(DLStudio.TextClassificationWithEmbeddings.SentimentAnalysisDataset, self).__init__()
                import gensim.downloader as gen_api
#                self.word_vectors = gen_api.load("word2vec-google-news-300")
                self.path_to_saved_embeddings = path_to_saved_embeddings
                self.train_or_test = train_or_test
                root_dir = dl_studio.dataroot
                f = gzip.open(root_dir + dataset_file, 'rb')
                dataset = f.read()
                if path_to_saved_embeddings is not None:
                    import gensim.downloader as genapi
                    from gensim.models import KeyedVectors 
                    if os.path.exists(path_to_saved_embeddings + 'vectors.kv'):
                        self.word_vectors = KeyedVectors.load(path_to_saved_embeddings + 'vectors.kv')
                    else:
                        print("""\n\nSince this is your first time to install the word2vec embeddings, it may take"""
                              """\na couple of minutes. The embeddings occupy around 3.6GB of your disk space.\n\n""")
                        self.word_vectors = genapi.load("word2vec-google-news-300")               
                        ##  'kv' stands for  "KeyedVectors", a special datatype used by gensim because it 
                        ##  has a smaller footprint than dict
                        self.word_vectors.save(path_to_saved_embeddings + 'vectors.kv')    
                if train_or_test == 'train':
                    if sys.version_info[0] == 3:
                        self.positive_reviews_train, self.negative_reviews_train, self.vocab = pickle.loads(dataset, encoding='latin1')
                    else:
                        self.positive_reviews_train, self.negative_reviews_train, self.vocab = pickle.loads(dataset)
                    self.categories = sorted(list(self.positive_reviews_train.keys()))
                    self.category_sizes_train_pos = {category : len(self.positive_reviews_train[category]) for category in self.categories}
                    self.category_sizes_train_neg = {category : len(self.negative_reviews_train[category]) for category in self.categories}
                    self.indexed_dataset_train = []
                    for category in self.positive_reviews_train:
                        for review in self.positive_reviews_train[category]:
                            self.indexed_dataset_train.append([review, category, 1])
                    for category in self.negative_reviews_train:
                        for review in self.negative_reviews_train[category]:
                            self.indexed_dataset_train.append([review, category, 0])
                    random.shuffle(self.indexed_dataset_train)
                elif train_or_test == 'test':
                    if sys.version_info[0] == 3:
                        self.positive_reviews_test, self.negative_reviews_test, self.vocab = pickle.loads(dataset, encoding='latin1')
                    else:
                        self.positive_reviews_test, self.negative_reviews_test, self.vocab = pickle.loads(dataset)
                    self.vocab = sorted(self.vocab)
                    self.categories = sorted(list(self.positive_reviews_test.keys()))
                    self.category_sizes_test_pos = {category : len(self.positive_reviews_test[category]) for category in self.categories}
                    self.category_sizes_test_neg = {category : len(self.negative_reviews_test[category]) for category in self.categories}
                    self.indexed_dataset_test = []
                    for category in self.positive_reviews_test:
                        for review in self.positive_reviews_test[category]:
                            self.indexed_dataset_test.append([review, category, 1])
                    for category in self.negative_reviews_test:
                        for review in self.negative_reviews_test[category]:
                            self.indexed_dataset_test.append([review, category, 0])
                    random.shuffle(self.indexed_dataset_test)

            def review_to_tensor(self, review):
                list_of_embeddings = []
                for i,word in enumerate(review):
                    if word in self.word_vectors.key_to_index:
                        embedding = self.word_vectors[word]
                        list_of_embeddings.append(np.array(embedding))
                    else:
                        next
                review_tensor = torch.FloatTensor(np.array(list_of_embeddings))
                return review_tensor

            def sentiment_to_tensor(self, sentiment):
                """
                Sentiment is ordinarily just a binary valued thing.  It is 0 for negative
                sentiment and 1 for positive sentiment.  We need to pack this value in a
                two-element tensor.
                """        
                sentiment_tensor = torch.zeros(2)
                if sentiment == 1:
                    sentiment_tensor[1] = 1
                elif sentiment == 0: 
                    sentiment_tensor[0] = 1
                sentiment_tensor = sentiment_tensor.type(torch.long)
                return sentiment_tensor

            def __len__(self):
                if self.train_or_test == 'train':
                    return len(self.indexed_dataset_train)
                elif self.train_or_test == 'test':
                    return len(self.indexed_dataset_test)

            def __getitem__(self, idx):
                sample = self.indexed_dataset_train[idx] if self.train_or_test == 'train' else self.indexed_dataset_test[idx]
                review = sample[0]
                review_category = sample[1]
                review_sentiment = sample[2]
                review_sentiment = self.sentiment_to_tensor(review_sentiment)
                review_tensor = self.review_to_tensor(review)
                category_index = self.categories.index(review_category)
                sample = {'review'       : review_tensor, 
                          'category'     : category_index, # should be converted to tensor, but not yet used
                          'sentiment'    : review_sentiment }
                return sample

        def load_SentimentAnalysisDataset(self, dataserver_train, dataserver_test ):   
            self.train_dataloader = torch.utils.data.DataLoader(dataserver_train,
                        batch_size=self.dl_studio.batch_size,shuffle=True, num_workers=2)
            self.test_dataloader = torch.utils.data.DataLoader(dataserver_test,
                               batch_size=self.dl_studio.batch_size,shuffle=False, num_workers=2)

        class TEXTnetWithEmbeddings(nn.Module):
            """
            This is embeddings version of the class TEXTnet class shown previously.  Since we
            are using the word2vec embeddings, we know that the input size for each word vector 
            will be a constant value of 300.  Overall, though, this network is meant for semantic 
            classification of variable-length sentiment data.  Based on my limited testing, the 
            performance of this network is very poor because it has no protection against 
            vanishing gradients when used in an RNN.  

            Class Path:  DLStudio -> TextClassificationWithEmbeddings -> TEXTnetWithEmbeddings
            """
            def __init__(self, input_size, hidden_size, output_size):
                super(DLStudio.TextClassificationWithEmbeddings.TEXTnetWithEmbeddings, self).__init__()
                self.input_size = input_size
                self.hidden_size = hidden_size
                self.output_size = output_size
                self.combined_to_hidden = nn.Linear(input_size + hidden_size, hidden_size)
                self.combined_to_middle = nn.Linear(input_size + hidden_size, 100)
                self.middle_to_out = nn.Linear(100, output_size)     
                self.logsoftmax = nn.LogSoftmax(dim=1)

            def forward(self, input, hidden):
                combined = torch.cat((input, hidden), 1)
                hidden = self.combined_to_hidden(combined)
                hidden = torch.tanh(hidden)                     
                out = self.combined_to_middle(combined)
                out = nn.functional.relu(out)
                out = self.middle_to_out(out)
                out = self.logsoftmax(out)
                return out,hidden         

            def init_hidden(self):
                hidden = torch.zeros(1, self.hidden_size)
                return hidden


        class TEXTnetOrder2WithEmbeddings(nn.Module):
            """
            This is an embeddings version of the TEXTnetOrder2 class shown previously.
            With the embeddings, we know that the size the tensor for word will be 300.
            As to how TEXTnetOrder2 differs from TEXTnet, the value of hidden as used at
            each time step also includes its value at the previous time step.  This 
            fact, not directly apparent by the definition of the class shown below, 
            is made possible by the last parameter, cell, in the header of forward().  
            All you can see here, at the end of forward(), is that the value of cell 
            goes through a linear layer and through a sigmoid nonlinearity. By the way, 
            since the sigmoid saturates at 0 and 1, it can act like a switch. Later 
            when I use this class in the training function, you will see the cell
            values being used in such a manner that the hidden state at each time
            step is mixed with the hidden state at the previous time step.

            Class Path:  DLStudio -> TextClassificationWithEmbeddings -> TEXTnetOrder2WithEmbeddings
            """
            def __init__(self, hidden_size, output_size, input_size=300):
                super(DLStudio.TextClassificationWithEmbeddings.TEXTnetOrder2WithEmbeddings, self).__init__()
                self.input_size = input_size
                self.hidden_size = hidden_size
                self.output_size = output_size
                self.combined_to_hidden = nn.Linear(input_size + 2*hidden_size, hidden_size)
                self.combined_to_middle = nn.Linear(input_size + 2*hidden_size, 100)
                self.middle_to_out = nn.Linear(100, output_size)     
                self.logsoftmax = nn.LogSoftmax(dim=1)
                self.dropout = nn.Dropout(p=0.1)
                # for the cell
                self.linear_for_cell = nn.Linear(hidden_size, hidden_size)

            def forward(self, input, hidden, cell):
                combined = torch.cat((input, hidden, cell), 1)
                hidden = self.combined_to_hidden(combined)
                hidden = torch.tanh(hidden)                     
                out = self.combined_to_middle(combined)
                out = nn.functional.relu(out)
                out = self.dropout(out)
                out = self.middle_to_out(out)
                out = self.logsoftmax(out)
                hidden_clone = hidden.clone()
#                cell = torch.tanh(self.linear_for_cell(hidden_clone))
                cell = torch.sigmoid(self.linear_for_cell(hidden_clone))
                return out,hidden,cell         

            def initialize_cell(self):
                weight = next(self.linear_for_cell.parameters()).data
                cell = weight.new(1, self.hidden_size).zero_()
                return cell

            def init_hidden(self):
                hidden = torch.zeros(1, self.hidden_size)
                return hidden


        class GRUnetWithEmbeddings(nn.Module):
            """
            For this embeddings adapted version of the GRUnet shown earlier, we can assume that
            the 'input_size' for a tensor representing a word is always 300.
            Source: https://blog.floydhub.com/gru-with-pytorch/
            with the only modification that the final output of forward() is now
            routed through LogSoftmax activation. 

            Class Path:  DLStudio -> TextClassificationWithEmbeddings -> GRUnetWithEmbeddings 
            """
            def __init__(self, input_size, hidden_size, output_size, num_layers=1): 
                """
                -- input_size is the size of the tensor for each word in a sequence of words.  If you word2vec
                       embedding, the value of this variable will always be equal to 300.
                -- hidden_size is the size of the hidden state in the RNN
                -- output_size is the size of output of the RNN.  For binary classification of 
                       input text, output_size is 2.
                -- num_layers creates a stack of GRUs
                """
                super(DLStudio.TextClassificationWithEmbeddings.GRUnetWithEmbeddings, self).__init__()
                self.input_size = input_size
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.gru = nn.GRU(input_size, hidden_size, num_layers)
                self.fc = nn.Linear(hidden_size, output_size)
                self.relu = nn.ReLU()
                self.logsoftmax = nn.LogSoftmax(dim=1)
                
            def forward(self, x, h):
                out, h = self.gru(x, h)
                out = self.fc(self.relu(out[:,-1]))
                out = self.logsoftmax(out)
                return out, h

            def init_hidden(self):
                weight = next(self.parameters()).data
                #                  num_layers  batch_size    hidden_size
                hidden = weight.new(  2,          1,         self.hidden_size    ).zero_()
                return hidden

        def save_model(self, model):
            "Save the trained model to a disk file"
            torch.save(model.state_dict(), self.dl_studio.path_saved_model)


        def run_code_for_training_with_TEXTnet_word2vec(self, net, display_train_loss=False):        
            filename_for_out = "performance_numbers_" + str(self.dl_studio.epochs) + ".txt"
            FILE = open(filename_for_out, 'w')
            net = copy.deepcopy(net)
            net = net.to(self.dl_studio.device)
            ## Note that the TEXTnet and TEXTnetOrder2 both produce LogSoftmax output. So we
            ## use nn.NLLLoss. The combined effect of LogSoftMax and NLLLoss is the same as 
            ## for the CrossEntropyLoss
            criterion = nn.NLLLoss()
            accum_times = []
            optimizer = optim.SGD(net.parameters(), 
                         lr=self.dl_studio.learning_rate, momentum=self.dl_studio.momentum)
            start_time = time.perf_counter()
            training_loss_tally = []
            for epoch in range(self.dl_studio.epochs):  
                print("")
                running_loss = 0.0
                for i, data in enumerate(self.train_dataloader):    
                    hidden = net.init_hidden().to(self.dl_studio.device)              
                    review_tensor,category,sentiment = data['review'], data['category'], data['sentiment']
                    review_tensor = review_tensor.to(self.dl_studio.device)
                    sentiment = sentiment.to(self.dl_studio.device)
                    optimizer.zero_grad()
                    input = torch.zeros(1,review_tensor.shape[2]).to(self.dl_studio.device)
                    for k in range(review_tensor.shape[1]):
                        input[0,:] = review_tensor[0,k]
                        output, hidden = net(input, hidden)
                    loss = criterion(output, torch.argmax(sentiment,1))
                    running_loss += loss.item()
                    loss.backward(retain_graph=True)        
                    optimizer.step()
                    if i % 200 == 199:    
                        avg_loss = running_loss / float(200)
                        training_loss_tally.append(avg_loss)
                        running_loss = 0.0
                        current_time = time.perf_counter()
                        time_elapsed = current_time-start_time
                        print("[epoch:%d  iter:%4d  elapsed_time: %4d secs]     loss: %.5f" % (epoch+1,i+1, time_elapsed,avg_loss))
                        accum_times.append(current_time-start_time)
                        FILE.write("%.3f\n" % avg_loss)
                        FILE.flush()
            print("\nFinished Training\n\n")
            self.save_model(net)
            if display_train_loss:
                plt.figure(figsize=(10,5))
                plt.title("Training Loss vs. Iterations")
                plt.plot(training_loss_tally)
                plt.xlabel("iterations")
                plt.ylabel("training loss")
                plt.legend()
                plt.savefig("training_loss.png")
                plt.show()


        def run_code_for_training_with_TEXTnetOrder2_word2vec(self, net, display_train_loss=False):        
            filename_for_out = "performance_numbers_" + str(self.dl_studio.epochs) + ".txt"
            FILE = open(filename_for_out, 'w')
            net = copy.deepcopy(net)
            net.to(self.dl_studio.device)
            ## Note that the TEXTnet and TEXTnetOrder2 both produce LogSoftmax output:
            criterion = nn.NLLLoss()
            accum_times = []
            optimizer = optim.SGD(net.parameters(), 
                                       lr=self.dl_studio.learning_rate, momentum=self.dl_studio.momentum)
            start_time = time.perf_counter()
            training_loss_tally = []
            for epoch in range(self.dl_studio.epochs):  
                print("")
                running_loss = 0.0
                for i, data in enumerate(self.train_dataloader):    
                    cell_prev = net.initialize_cell().to(self.dl_studio.device)
                    cell_prev_2_prev = net.initialize_cell().to(self.dl_studio.device)
                    hidden = net.init_hidden().to(self.dl_studio.device)              
                    review_tensor,category,sentiment = data['review'], data['category'], data['sentiment']
                    review_tensor = review_tensor.to(self.dl_studio.device)
                    sentiment = sentiment.to(self.dl_studio.device)
                    optimizer.zero_grad()
                    input = torch.zeros(1,review_tensor.shape[2])
                    input = input.to(self.dl_studio.device)
                    for k in range(review_tensor.shape[1]):
                        input[0,:] = review_tensor[0,k]
                        output, hidden, cell = net(input, hidden, cell_prev_2_prev)
                        if k == 0:
                            cell_prev = cell
                        else:
                            cell_prev_2_prev = cell_prev
                            cell_prev = cell
                    loss = criterion(output, torch.argmax(sentiment,1))
                    running_loss += loss.item()
                    loss.backward()        
                    optimizer.step()
                    if i % 200 == 199:    
                        avg_loss = running_loss / float(200)
                        training_loss_tally.append(avg_loss)
                        current_time = time.perf_counter()
                        time_elapsed = current_time-start_time
                        print("[epoch:%d  iter:%4d  elapsed_time: %4d secs]     loss: %.5f" % (epoch+1,i+1, time_elapsed,avg_loss))
                        accum_times.append(current_time-start_time)
                        FILE.write("%.3f\n" % avg_loss)
                        FILE.flush()
                        running_loss = 0.0
            print("\nFinished Training\n")
            self.save_model(net)
            if display_train_loss:
                plt.figure(figsize=(10,5))
                plt.title("Training Loss vs. Iterations")
                plt.plot(training_loss_tally)
                plt.xlabel("iterations")
                plt.ylabel("training loss")
                plt.legend()
                plt.savefig("training_loss.png")
                plt.show()


        def run_code_for_training_for_text_classification_with_GRU_word2vec(self, net, display_train_loss=False): 
            filename_for_out = "performance_numbers_" + str(self.dl_studio.epochs) + ".txt"
            FILE = open(filename_for_out, 'w')
            net = copy.deepcopy(net)
            net = net.to(self.dl_studio.device)
            ##  Note that the GRUnet now produces the LogSoftmax output:
            criterion = nn.NLLLoss()
            accum_times = []
            optimizer = optim.SGD(net.parameters(), 
                         lr=self.dl_studio.learning_rate, momentum=self.dl_studio.momentum)
            training_loss_tally = []
            start_time = time.perf_counter()
            for epoch in range(self.dl_studio.epochs):  
                print("")
                running_loss = 0.0
                for i, data in enumerate(self.train_dataloader):    
                    review_tensor,category,sentiment = data['review'], data['category'], data['sentiment']
                    review_tensor = review_tensor.to(self.dl_studio.device)
                    sentiment = sentiment.to(self.dl_studio.device)
                    ## The following type conversion needed for MSELoss:
                    ##sentiment = sentiment.float()
                    optimizer.zero_grad()
                    hidden = net.init_hidden().to(self.dl_studio.device)
                    for k in range(review_tensor.shape[1]):
                        output, hidden = net(torch.unsqueeze(torch.unsqueeze(review_tensor[0,k],0),0), hidden)
                    loss = criterion(output, torch.argmax(sentiment, 1))
                    running_loss += loss.item()
                    loss.backward()
                    optimizer.step()
                    if i % 200 == 199:    
                        avg_loss = running_loss / float(200)
                        training_loss_tally.append(avg_loss)
                        current_time = time.perf_counter()
                        time_elapsed = current_time-start_time
                        print("[epoch:%d  iter:%4d  elapsed_time:%4d secs]     loss: %.5f" % (epoch+1,i+1, time_elapsed,avg_loss))
                        accum_times.append(current_time-start_time)
                        FILE.write("%.5f\n" % avg_loss)
                        FILE.flush()
                        running_loss = 0.0
            self.save_model(net)
            print("Total Training Time: {}".format(str(sum(accum_times))))
            print("\nFinished Training\n\n")
            if display_train_loss:
                plt.figure(figsize=(10,5))
                plt.title("Training Loss vs. Iterations")
                plt.plot(training_loss_tally)
                plt.xlabel("iterations")
                plt.ylabel("training loss")
                plt.legend()
                plt.savefig("training_loss.png")
                plt.show()


        def run_code_for_testing_with_TEXTnet_word2vec(self, net):
            net.load_state_dict(torch.load(self.dl_studio.path_saved_model))
            net.to(self.dl_studio.device)
            classification_accuracy = 0.0
            negative_total = 0
            positive_total = 0
            confusion_matrix = torch.zeros(2,2)
            with torch.no_grad():
                for i, data in enumerate(self.test_dataloader):
                    review_tensor,category,sentiment = data['review'], data['category'], data['sentiment']
                    review_tensor = review_tensor.to(self.dl_studio.device)
                    category      = category.to(self.dl_studio.device)
                    sentiment     = sentiment.to(self.dl_studio.device)
                    input = torch.zeros(1,review_tensor.shape[2]).to(self.dl_studio.device)
                    hidden = net.init_hidden().to(self.dl_studio.device)
                    for k in range(review_tensor.shape[1]):
                        input[0,:] = review_tensor[0,k]
                        output, hidden = net(input, hidden)
                    predicted_idx = torch.argmax(output).item()
                    gt_idx = torch.argmax(sentiment).item()
                    if i % 100 == 99:
                        print("   [i=%4d]    predicted_label=%d       gt_label=%d" % (i+1, predicted_idx,gt_idx))
                    if predicted_idx == gt_idx:
                        classification_accuracy += 1
                    if gt_idx == 0: 
                        negative_total += 1
                    elif gt_idx == 1:
                        positive_total += 1
                    confusion_matrix[gt_idx,predicted_idx] += 1
            print("\nOverall classification accuracy: %0.2f%%" %  (float(classification_accuracy) * 100 /float(i)))
            out_percent = np.zeros((2,2), dtype='float')
            out_percent[0,0] = "%.3f" % (100 * confusion_matrix[0,0] / float(negative_total))
            out_percent[0,1] = "%.3f" % (100 * confusion_matrix[0,1] / float(negative_total))
            out_percent[1,0] = "%.3f" % (100 * confusion_matrix[1,0] / float(positive_total))
            out_percent[1,1] = "%.3f" % (100 * confusion_matrix[1,1] / float(positive_total))
            print("\n\nNumber of positive reviews tested: %d" % positive_total)
            print("\n\nNumber of negative reviews tested: %d" % negative_total)
            print("\n\nDisplaying the confusion matrix:\n")
            out_str = "                      "
            out_str +=  "%18s    %18s" % ('predicted negative', 'predicted positive')
            print(out_str + "\n")
            for i,label in enumerate(['true negative', 'true positive']):
                out_str = "%12s%%:  " % label
                for j in range(2):
                    out_str +=  "%18s%%" % out_percent[i,j]
                print(out_str)


        def run_code_for_testing_with_TEXTnetOrder2_word2vec(self, net):
            net.load_state_dict(torch.load(self.dl_studio.path_saved_model))
            net.to(self.dl_studio.device)
            classification_accuracy = 0.0
            negative_total = 0
            positive_total = 0
            confusion_matrix = torch.zeros(2,2)
            with torch.no_grad():
                for i, data in enumerate(self.test_dataloader):
                    cell_prev = net.initialize_cell()
                    cell_prev_2_prev = net.initialize_cell()
                    review_tensor,category,sentiment = data['review'], data['category'], data['sentiment']
                    input = torch.zeros(1,review_tensor.shape[2]).to(self.dl_studio.device)
                    hidden = net.init_hidden().to(self.dl_studio.device)
                    for k in range(review_tensor.shape[1]):
                        input[0,:] = review_tensor[0,k]
                        output, hidden, cell = net(input, hidden, cell_prev_2_prev)
                        if k == 0:
                            cell_prev = cell
                        else:
                            cell_prev_2_prev = cell_prev
                            cell_prev = cell
                    predicted_idx = torch.argmax(output).item()
                    gt_idx = torch.argmax(sentiment).item()
                    if i % 100 == 99:
                        print("   [i=%4d]    predicted_label=%d       gt_label=%d" % (i+1, predicted_idx,gt_idx))
                    if predicted_idx == gt_idx:
                        classification_accuracy += 1
                    if gt_idx == 0: 
                        negative_total += 1
                    elif gt_idx == 1:
                        positive_total += 1
                    confusion_matrix[gt_idx,predicted_idx] += 1
            print("\nOverall classification accuracy: %0.2f%%" %  (float(classification_accuracy) * 100 /float(i)))
            out_percent = np.zeros((2,2), dtype='float')
            out_percent[0,0] = "%.3f" % (100 * confusion_matrix[0,0] / float(negative_total))
            out_percent[0,1] = "%.3f" % (100 * confusion_matrix[0,1] / float(negative_total))
            out_percent[1,0] = "%.3f" % (100 * confusion_matrix[1,0] / float(positive_total))
            out_percent[1,1] = "%.3f" % (100 * confusion_matrix[1,1] / float(positive_total))
            print("\n\nNumber of positive reviews tested: %d" % positive_total)
            print("\n\nNumber of negative reviews tested: %d" % negative_total)
            print("\n\nDisplaying the confusion matrix:\n")
            out_str = "                      "
            out_str +=  "%18s    %18s" % ('predicted negative', 'predicted positive')
            print(out_str + "\n")
            for i,label in enumerate(['true negative', 'true positive']):
                out_str = "%12s:  " % label
                for j in range(2):
                    out_str +=  "%18s" % out_percent[i,j]
                print(out_str)


        def run_code_for_testing_text_classification_with_GRU_word2vec(self, net):
            net.load_state_dict(torch.load(self.dl_studio.path_saved_model))
            classification_accuracy = 0.0
            negative_total = 0
            positive_total = 0
            confusion_matrix = torch.zeros(2,2)
            with torch.no_grad():
                for i, data in enumerate(self.test_dataloader):
                    review_tensor,category,sentiment = data['review'], data['category'], data['sentiment']
                    hidden = net.init_hidden()
                    for k in range(review_tensor.shape[1]):
                        output, hidden = net(torch.unsqueeze(torch.unsqueeze(review_tensor[0,k],0),0), hidden)
                    predicted_idx = torch.argmax(output).item()
                    gt_idx = torch.argmax(sentiment).item()
                    if i % 100 == 99:
                        print("   [i=%d]    predicted_label=%d       gt_label=%d" % (i+1, predicted_idx,gt_idx))
                    if predicted_idx == gt_idx:
                        classification_accuracy += 1
                    if gt_idx == 0: 
                        negative_total += 1
                    elif gt_idx == 1:
                        positive_total += 1
                    confusion_matrix[gt_idx,predicted_idx] += 1
            print("\nOverall classification accuracy: %0.2f%%" %  (float(classification_accuracy) * 100 /float(i)))
            out_percent = np.zeros((2,2), dtype='float')
            out_percent[0,0] = "%.3f" % (100 * confusion_matrix[0,0] / float(negative_total))
            out_percent[0,1] = "%.3f" % (100 * confusion_matrix[0,1] / float(negative_total))
            out_percent[1,0] = "%.3f" % (100 * confusion_matrix[1,0] / float(positive_total))
            out_percent[1,1] = "%.3f" % (100 * confusion_matrix[1,1] / float(positive_total))
            print("\n\nNumber of positive reviews tested: %d" % positive_total)
            print("\n\nNumber of negative reviews tested: %d" % negative_total)
            print("\n\nDisplaying the confusion matrix:\n")
            out_str = "                      "
            out_str +=  "%18s    %18s" % ('predicted negative', 'predicted positive')
            print(out_str + "\n")
            for i,label in enumerate(['true negative', 'true positive']):
                out_str = "%12s:  " % label
                for j in range(2):
                    out_str +=  "%18s%%" % out_percent[i,j]
                print(out_str)


#_________________________  End of DLStudio Class Definition ___________________________

#______________________________    Test code follows    _________________________________

if __name__ == '__main__': 
    from torchinfo import summary

    dataroot = "/home/tam/git/ece60146/data/hw8_dataset/"
    dataset_archive_train = "sentiment_dataset_train_400.tar.gz"
    dataset_archive_test =  "sentiment_dataset_test_400.tar.gz"

    dls = DLStudio(
        dataroot=dataroot,
        path_saved_model="./saved_model",
        momentum=0.9,
        learning_rate= 1e-5,
        epochs=1,
        batch_size=1,
        classes=('negative','positive'),
        use_gpu=True,
    )
    text_cl = DLStudio.TextClassificationWithEmbeddings(dl_studio=dls)
    dataserver_train = DLStudio.TextClassificationWithEmbeddings.SentimentAnalysisDataset(
        train_or_test='train',
        dl_studio=dls,
        dataset_file=dataset_archive_train,
        path_to_saved_embeddings=dataroot,
    )
    dataserver_test = DLStudio.TextClassificationWithEmbeddings.SentimentAnalysisDataset(
        train_or_test='test',
        dl_studio=dls,
        dataset_file=dataset_archive_test,
        path_to_saved_embeddings=dataroot,
    )
    text_cl.dataserver_train = dataserver_train
    text_cl.dataserver_test = dataserver_test
    text_cl.load_SentimentAnalysisDataset(dataserver_train, dataserver_test)
    model = text_cl.GRUnetWithEmbeddings(input_size=300, hidden_size=100, output_size=2, num_layers=2)
    number_of_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_layers = len(list(model.parameters()))
    print(f"\nThe number of layers in the model: {num_layers}")
    print(f"\nThe number of learnable parameters in the model: "
          f"{number_of_learnable_params}")
    summary(model)

    print("\nStarting training\n")
    text_cl.run_code_for_training_for_text_classification_with_GRU_word2vec(model, display_train_loss=True)

    ## TESTING:
    text_cl.run_code_for_testing_text_classification_with_GRU_word2vec(model)