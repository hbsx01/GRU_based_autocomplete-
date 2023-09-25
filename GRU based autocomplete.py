import numpy as np
import matplotlib.pylab as plt
import torch
import torch.nn as nn
import copy
from tqdm import tqdm
import re
from termcolor import colored  
import os
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'



# Dataset downloading 
class Origin(torch.utils.data.Dataset):
    # Obtained externally from the internet 
    '''
     ds = Origin(seq_length=10, start=0, stop=-1)
     
     Creates a PyTorch Dataset object, holding a simplified version
     of the text from Charles Darwin's "On the Origin of Species".
     
     The class contains utility functions to convert between the
     string-based form of a sequence, and its vector encoding, in which
     each character is represented by a one-hot 28-vector corresponding
     to the 28 characters in the string
       ' .abcdefghijklmnopqrstuvwxyz'  (the first character is a space)
     
     The target sequences are the same as the inputs, but advanced by
     one character.
     
     Inputs:
      seq_length  the number of characters in each sequence
      start       the index of the character to start taking sequences from
      stop        the index of the character to stop taking sequences from
      
     Usage:
      ds = Origin(seq_length=5, start=7, stop=100)
      x,t = ds.__getitem__(0)
      print(ds.read_seq(x))   # Produces 'origi'
      print(ds.read_seq(t))   # Produces 'rigin'
    '''
    
    def __init__(self, seq_length=10, start=0, stop=-1):
        self.seq_length = seq_length
        os.chdir(r"C:\Users\User\Desktop\Projects\GRU based Auto-Complete")
        orig_text = open('origin_of_species.txt').read().lower()
        chars = sorted(list(set(orig_text)))
        chars.insert(0, "\0") # add newline character
        vocab_size = len(chars)

        char_indices = dict((c, i) for i, c in enumerate(chars))
        indices_char = dict((i, c) for i, c in enumerate(chars))
        idx = [char_indices[c] for c in orig_text]

        # simplifying it by keeping only letters, spaces, and periods.
        filt_idx = []
        for i in idx:
            if i<=24 and i!=10:
                filt_idx.append(2)
            elif i>24 or i==10:
                filt_idx.append(i)
        blah = ''.join([indices_char[f] for f in filt_idx])
        self.text = re.sub(' +', ' ', blah)  # collapse multiple spaces using regular expressions
        self.text = self.text[start:stop]
        #chars = sorted(list(set(self.text)))
        chars = sorted(list(set(' .abcdefghijklmnopqrstuvwxyz')))
        self.vocab_size = len(chars)
        print('Character set: '+''.join(chars)+' (first char is a space)')

        self.char_indices = dict((c, i) for i, c in enumerate(chars))
        self.indices_char = dict((i, c) for i, c in enumerate(chars))
        self.idx = [self.char_indices[c] for c in self.text]

        print('There are '+str(self.vocab_size)+' characters in our character set')

    def __len__(self):
        return len(self.text) - 1 - self.seq_length
    
    def __getitem__(self, k):
        x = self.idx[k:k+self.seq_length]
        t = self.idx[k+1:k+1+self.seq_length]
        return self.seq_i2v(x), torch.tensor(t, dtype=torch.long)
      
    def seq_i2v(self, seq):
        x = torch.zeros((len(seq), self.vocab_size))
        for k,i in enumerate(seq):
            x[k,i] = 1.
        return x
    
    def read_seq(self, x):
        idx = [torch.argmax(v).item() for v in x]        
        return ''.join(self.indices_char[i] for i in idx)
    
    def encode_seq(self, c):
        idx = [self.char_indices[cc] for cc in c]
        return self.seq_i2v(idx)
    

# Initializing the dataset
oos = Origin(start=11000, stop=21000, seq_length=10)
x, t = oos.__getitem__(0)

# Checking if sampling the prose works
print('Here is how you can view one of the samples:')
print(f'Sample input: "{oos.read_seq(x)}"')


# Creating a PyTorch DataLoader
dl = torch.utils.data.DataLoader(oos, batch_size=128, shuffle=True)

# Defining the GRU class
class GRU(nn.Module):
    '''
     net = GRU(dims)
     Input:
       dims is [I, H], where the input/output layers have I neurons, and the
       hidden layer has H neurons.

    '''
    def __init__(self, dims):
        super().__init__()
        self.losses = []
        
        self.input_dim, self.hidden_dim = dims
        
        self.U_r = nn.Linear(self.input_dim, self.hidden_dim)
        self.W_r = nn.Linear(self.hidden_dim, self.hidden_dim , bias=True)
        self.act_r = nn.Sigmoid()
        self.r_t = 0
        
        self.U_z = nn.Linear(self.input_dim, self.hidden_dim)
        self.W_z = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        #self.b_z = nn.Linear(1,self.hidden_dim)
        self.act_z = nn.Sigmoid()
        self.z_t = 0
        
        self.U_h = nn.Linear(self.input_dim, self.hidden_dim)
        self.W_h = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        #self.b_h = nn.Linear(1,self.hidden_dim)
        self.act_h_bar = nn.Tanh()
        self.h_bar_t = 0
        
        self.V = nn.Linear(self.hidden_dim, self.input_dim, bias=True)
        #self.b_y = nn.Linear(1, self.input_dim)
        self.act_y = torch.nn.LogSoftmax()

        
        
    def step(self, x, h):
        '''
         hnext = net.step(x, h)
         
         Takes a time step, with input x and current hidden state h.
         Returns the new h.
         
         Inputs:
          x      a DxI tensor holding a batch of inputs, where
                    D is the batch size, and
                    I is the dimension of the inputs
          h      a DxH tensor holding a batch of hidden states, where
                    H is the number of hidden nodes
                
         Output:
          hnext  a DxH tensor holding the hidden states for the next
                 timestep
        '''
       
        self.r_t = self.act_r(self.U_r(x) + self.W_r(h))
        self.z_t = self.act_z(self.U_z(x) + self.W_z(h))
        self.h_bar_t = self.act_h_bar(self.U_h(x) + self.W_h(torch.mul(self.r_t,h)))
        h = torch.mul(1-self.z_t,h) + torch.mul(self.z_t,self.h_bar_t)
        return h 
    
    
    def output(self, h):
        '''
         y = net.output(h)
         
         Given the hidden state, returns the *log* of the output.
         Example: for categorical cross-entropy, it should return LogSoftmax.
         
         Input:
          h  a DxH tensor holding a batch of hidden states, where
                D is the batch size, and
                H is the dimension of the hidden state (number of hidden nodes)
                
         Output:
          y  a DxI tensor holding a batch of outputs, where
                I is the dimension of the output
        '''
        
        return self.act_y(self.V(h))
    
    
    def forward(self, x):
        '''
         y = net.forward(x)
         
         Takes a batch of squences, and returns the batch of output
         sequences.
         
         Inputs:
          x      a DxTxI tensor, where
                    D is the batch size (number of sequences in the batch)
                    T is the sequence length, and
                    I is the dimension of the input to the network
                 
         Output:
          y      a DxTxI tensor, as above
        '''
        
        x = x.to(device)  # me trying to use GPU instead of the CPU to train the GRU


        # To reorder the batch from (D, T, I) to (T, D, I) so that the batch can run through the network, one timestep at a time.
        seq_of_batches = torch.einsum('ijk->jik', x)

        output_seq = []
        T, samples, input_dim = seq_of_batches.shape
        h = torch.zeros((samples, self.hidden_dim)).to(device)
        for xt in seq_of_batches:
            h = self.step(xt, h)
            output_seq.append(self.output(h))
        y = torch.stack(output_seq, dim=0).to(device)  # (T, batch_size, output_dim)
        return torch.einsum('jik->ijk', y)  # (batch_size, T, output_dim)

    # Backprop through time class to train the network
    def bptt(self, dl, epochs=10, loss_fcn=nn.NLLLoss(), lr=0.001):
        '''
         net.bptt(dl, epochs=10, loss_fcn=nn.NLLLoss(), lr=0.001)
         
         Trains the recurrent network using Backprop Through Time.
         
         Inputs:
          dl        PyTorch DataLoader object
                    Each batch shoud be shaped DxTxI where
                      D is the number of sequences in the batch,
                      T is the length of each sequence, and
                      I is dim of each input to the network
          epochs    number of epochs to train for
          loss_fcn  PyTorch loss function
          lr        learning rate
        '''
        optim = torch.optim.Adam(self.parameters(), lr=lr)  # optimizer
        for epoch in tqdm(range(epochs)):
            total_loss = 0.
            for x,t in (dl):
                y = self(x)   # process the batch of sequences
                
                # go through output sequences, and compute loss
                loss = torch.tensor(0., device=device, requires_grad=True)
                for ys,ts in zip(y,t.to(device)):
                    loss = loss + loss_fcn(ys, ts)
                    
                # Auto Differentiation
                optim.zero_grad()
                loss.backward()
                optim.step()
                total_loss += loss.detach().cpu().item()
                
            self.losses.append(total_loss/len(dl))
        plt.plot(self.losses)
        
        
    def predict(self, x, n=10):
        '''
         y = net.predict(x, n=10)
         
         Run the network on sequence x, and then continue to predict
         the next n outputs.
         
         Inputs:
          x  a TxI tensor for a single input sequence
          n  how many output timesteps to predict
          
         Output:
          y  an nxI tensor, holding the sequence of n outputs
             predicted after the input sequence
        '''
        assert len(x.shape)==2
        with torch.no_grad():
            h = torch.zeros((1, self.hidden_dim)).to(device)
            for xx in x:  # step through the given sequence
                h = self.step(xx, h)
            y = self.output(h)
            pred = [y]   # for storing the output sequence
            
            # Take n more steps, and add the network's output
            for t in range(n-1):
                # Make a one-hot input out of the last output
                c = torch.argmax(y)
                x = torch.zeros_like(y)
                x[0,c] = 1.
                # Take a timestep
                h = self.step(x, h)
                y = self.output(h) # output from prev step becomes input to next step
                pred.append(y)
                
        return torch.stack(pred, dim=0)
    

# Creating and training the network
net = GRU([oos.vocab_size, 400])
net.bptt(dl, epochs=40, loss_fcn=nn.NLLLoss(reduction='mean'), lr=0.001)

# Saving and loading my trained network as a .pt file for later use 
torch.save(net.cpu(), 'mygru.pt')
net = torch.load('mygru.pt').to(device)



# Run experiment on a bunch of random seed sequences.
# - choose random seed seq
# - predict next 100 characters
# - find out how many characters match the text

import random
import re
msl = [0] * 101

def compare_strings(str1, str2):
    count = 0
    loop = min(len(str1), len(str2))
    for i in range(loop):
        ##print(f"Comparing {str1[i]} and {str2[i]}")
        if str1[i] == str2[i]:
            #print("This worked")
            count = count + 1
        else:
            break
    return count



for i in range(200):
    j = 0
    pos = 0
    max_chars = 0
    temp_chars = []
    seq = random.randint(1,9889)
    word = oos.text[seq:seq+10]
    v = oos.encode_seq(word)  
    y = net.predict(v, n = 100)  
    pred = f'{oos.read_seq(v)}'+ f'{oos.read_seq(y)}'
    occur = [m.start() for m in re.finditer(word, oos.text)]
    
    for i in range(len(occur)):
        #print(oos.text[occur[i]:occur[i]+30])
        temp_chars = compare_strings(str(oos.text[occur[i]:occur[i]+110]),pred)
        temp_chars -= len(word)
        if temp_chars > max_chars:
            max_chars = temp_chars
    #print(max_chars)
    msl[max_chars] += 1
    
            
    
        
# Plot matching-length vs trials
x = []
for i in range(101):
    x.append(i)
    
plt.plot(x,msl)
plt.xlabel("Matching String Length")
plt.ylabel("Trials")
print(msl)
