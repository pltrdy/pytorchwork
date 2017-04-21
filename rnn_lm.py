
# coding: utf-8

# In[1]:

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[2]:

import sys
class Params(object):
    """A parameters class which attributes are in fact a dict       
    """
    def __init__(self, params={}):
        self.params = params
        
    def __getattribute__(self,name):
        try:
            ret = object.__getattribute__(self, name)
            return ret
        except AttributeError:
            t, v, tb = sys.exc_info()
            if name in self.params:
                return self.params[name]
            raise v.with_traceback(tb)
            
    def __setattr__(self, name, value):
        if name == "params":
            super().__setattr__(name, value)
        else:
            self.params[name] = value

class RnnLm(nn.Module):
    def __init__(self, params):
        """
            params: a 'Params' object with at least:
                - vocab_size
                - embed_dim
                - hidden_size
        """
        super().__init__()
        self.params = params
        
        self.embedding = nn.Embedding(num_embeddings=params.vocab_size, 
                                      embedding_dim=params.embed_dim)
        
        self.cell = nn.LSTM(input_size=params.embed_dim, 
                            hidden_size=params.hidden_size,
                            batch_first=True)
        
        self.out_w = nn.Parameter(torch.randn(params.hidden_size, params.vocab_size))
        self.out_b = nn.Parameter(torch.randn(params.vocab_size))
    
    def _embed_data(self, src):
        """Embeds a list of words 
        """
        src_var = autograd.Variable(src)
        embedded = self.embedding(src_var)
        return embedded
        
    def forward(self, inputs):
        # inputs: nested list [batch_size x time_steps]
        # emb_inputs: [bs x ts x emb_size]
        emb_inputs = self._embed_data(inputs) 
        log("Input: %s ; Embedded: %s "% (str(inputs.size()), str(emb_inputs.size())))
        

        # Running the RNN
        # o: [bs x ts x h_size]
        # h: [n_layer x ts x h_size]
        # c: [n_layer x ts x h_size]
        o, (h, c) = self.cell(emb_inputs)
        o = o.contiguous()
        self.o = o
        log("Outputs: %s" % str(o.size()))
        log("h %s" % str(h.size()))
        log("c %s" % str(c.size()))
        
        
        # Output projection
        # oo: [bs*ts x h_size]
        # logits: [bs*ts x vocab_size]
        oo = o.view(-1, params.hidden_size)
        
        log("type: oo: %s; out_w: %s" % (str(type(oo)), str(type(self.out_w))))
        log("data type: oo: %s; out_w: %s" % (str(type(oo.data)), str(type(self.out_w.data))))
        
        log("oo: %s" % str(oo.size()))
        log("w: %s" % str(self.out_w.size()))
        logits = oo @ self.out_w
        logits = logits + self.out_b.expand_as(logits)
        log("Logits: %s" % str(logits.size()))
        
        # Softmax
        prediction = F.log_softmax(logits)
        
        return prediction
        
def log(*args, **kwargs):
    #print(*args, **kwargs)
    pass
    


# In[3]:

import reader

class Trainer:
    def __init__(self, params):
        """
            params:
                - data_path
                - batch_size
                - num_steps
                - cuda: bool
        """
        self.params = params
        
        print("Loading data...")
        self.train_data, self.valid_data, self.test_data, self.w2i = reader.raw_data(params.data_path)
        print("Loaded\t%d training words\n\t%d validation words\n\t%d test words" % (
                    len(self.train_data), len(self.valid_data), len(self.test_data)))
        print("Vocabulary: %d" % len(self.w2i))
        self.eos = self.w2i['<eos>']
        
        print("Creating model...")
        
        self.model = RnnLm(params)
        if self.params.cuda:
            print("Using CUDA")
            self.model.cuda()
        print("Done.")
        
    def batch_iterator(self, data):
        return reader.iterator(data, self.params.batch_size, self.params.num_steps)
    
    def run_epoch(self):
        import time
        stime = time.time()
        s = 0
        num_iter = (len(self.train_data)/self.params.num_steps) / self.params.batch_size
        print(num_iter)
        log_step = int(num_iter / 20)
        print(log_step)
        for step, (xx, yy) in enumerate(self.batch_iterator(self.train_data)):
            self.model.zero_grad()
            
            self.x = x = torch.LongTensor(xx.tolist())#.cuda()
            y = torch.LongTensor(yy.tolist())
            print_size(y, "y")
            flat_y = y.view(-1, 1)
            print_size(flat_y, "flat_y")
            y_onehot = torch.FloatTensor(params.batch_size*params.num_steps, params.vocab_size)
            y_onehot.zero_()
            y_onehot.scatter_(1, flat_y, 1)
            print_size(y_onehot, "y_onehot")
            
            if self.params.cuda:
                self.x = x = x.cuda()
                y_onehot = y_onehot.cuda()
            
            y_true = autograd.Variable(y_onehot)
            
            # Good discussions here
            # http://stackoverflow.com/questions/34240703/difference-between-tensorflow-tf-nn-softmax-and-tf-nn-softmax-cross-entropy-with
            # 
            # total_loss_1 = tf.reduce_mean(
            #                    -tf.reduce_sum(y_true * tf.log(y_hat_softmax), reduction_indices=[1]))
            #
            # is ~equivalent to
            # tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_hat, y_true))
            # (difference comes from numerical instability)
            #
            # here: y_hat = pred = self.model(x)
            # thus  y_hat_softmax = F.softmax(y_hat)
            
            y_hat = self.model(x)
            y_hat_softmax = F.softmax(y_hat)
            print_size(y_hat, "y_hat")
            print_size(y_hat_softmax, "y_hat_softmax")
            
            loss_per_instance = -(y_true * y_hat_softmax.log()).sum(dim=1)
            loss = loss_per_instance.mean()
            
            print_size(loss_per_instance, "loss_per_instance")
            print_size(loss, "loss")
            #print(loss)
            #print(step)
            s += float(loss.data[0])
            
            print("Right before backward")
            loss.backward()
            print("Right after backward")
            optimizer = optim.SGD(self.model.parameters(), lr=0.75)
            optimizer.step()
            
            if step % log_step == (log_step-1):
                _etime = time.time() - stime
                _err = float(s)/float(step+1)
                _words = (self.params.batch_size*self.params.num_steps*step)
                _wps = _words / _etime
                print("[%d] Err: %f\twps: %.3f" % (step, _err, _wps))
        return float(s)/float(step+1)

            
def print_size(tensor, name):
    pass
    # print("%s: %s" % (name, str(tensor.size())))


# In[4]:

params = Params({
    "data_path": "./ptb",
    "cuda": True,    
    "vocab_size": 10000,
    "embed_dim": 200,
    "hidden_size": 200,
    "batch_size": 64,
    "num_steps": 20
})


# In[5]:

tr = Trainer(params)


# In[6]:

for i in range(1, 10):
    print("Epoch %d" % i)
    print("Avg Err: %f" % tr.run_epoch())


# In[ ]:




# In[ ]:



