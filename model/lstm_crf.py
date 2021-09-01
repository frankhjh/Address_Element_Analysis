import torch
import torch.nn as nn
import torch.optim as optim
from utils.help import argmax,log_sum_exp


class BiLSTM_CRF(nn.Module):
    def __init__(self,vocab_size,labels_dict,embedding_dim,hidden_dim,batch_size=32):
        super(BiLSTM_CRF,self).__init__()
        self.embedding_dim=embedding_dim
        self.hidden_dim=hidden_dim
        self.vocab_size=vocab_size
        self.labels_dict=labels_dict
        self.labelset_size=len(labels_dict)

        self.word_embeds=nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embedding_dim,
                                      padding_idx=0)
        self.lstm=nn.LSTM(input_size=embedding_dim,
                          hidden_size=hidden_dim//2,
                          num_layers=1,
                          bidirectional=True)
        
        # connect the output of lstm layer to dim of label space
        self.hidden2label=nn.Linear(hidden_dim,self.labelset_size)

        # transition matrix (part of the model parameter)
        self.transition_mat=nn.Parameter(torch.randn(self.labelset_size,self.labelset_size))

        # set contraints for transition matrix

        # NEVER TO THE START LABEL
        self.transition_mat.data[labels_dict['<START>'],:]=-10000
        # NEVER FROM THR STOP LABEL
        self.transition_mat.data[:,labels_dict['<STOP>']]=-10000
        # FROM PAD CAN ONLY TO PAD OR STOP,NO OTHERS
        self.transition_mat.data[labels_dict['<START>'],labels_dict['PAD']]=-10000
        self.transition_mat.data[labels_dict['<STOP>']+1:,labels_dict['PAD']]=-10000

        self.batch_size=batch_size # generally, not a good idea to put it in the model

    def init_hidden(self,predict=False):
        # the first dim = num of direction * num of layers
        if predict:
            return (torch.randn(2, 1, self.hidden_dim // 2),
                    torch.randn(2, 1, self.hidden_dim // 2))
        
        return (torch.randn(2*1,self.batch_size,self.hidden_dim//2),
                torch.randn(2*1,self.batch_size,self.hidden_dim//2))
    
    def _forward_alg(self,feats): # feats are the output of lstm (emit matrix)
        init_alphas=torch.full((1,self.labelset_size),-10000.)
        init_alphas[0][self.labels_dict['<START>']]=0. # START label has max score

        forward_var=init_alphas

        for feat in feats: 
            alphas_t=[] # at step t
            for next_label in range(self.labelset_size):
                # broadcasting the emit-score for this label
                emit_score=feat[next_label].view(1,-1).expand(1,self.labelset_size)
                # transition score for all possible label i to this label
                trans_score=self.transition_mat[next_label].view(1,-1)
                #sum them up
                next_label_var=forward_var+emit_score+trans_score
                #cal the log_sum_exp score for the path to this label and store it io alphas_t
                alphas_t.append(log_sum_exp(next_label_var).view(1))
            forward_var=torch.cat(alphas_t).view(1,-1)
        
        # last step
        terminal_var=forward_var+self.transition_mat[self.labels_dict['<STOP>']]
        alpha=log_sum_exp(terminal_var)
        return alpha
    
    # reference:https://github.com/mali19064/LSTM-CRF-pytorch-faster/blob/master/LSTM_CRF_faster.py
    def _forward_alg_faster(self,feats):
        init_alphas=torch.full([self.labelset_size],-10000.)
        init_alphas[self.labels_dict['<START>']]=0.

        forward_var_li=[init_alphas]
        for feat_idx in range(feats.size(0)):
            forward_var=torch.stack([forward_var_li[feat_idx]]*feats.size(1))
            emit_score=torch.unsqueeze(feats[feat_idx],0).transpose(0,1)
            aa=forward_var+emit_score+self.transition_mat
            forward_var_li.append(torch.logsumexp(aa,dim=1))
        # last step
        terminal_var=forward_var_li[-1]+self.transition_mat[self.labels_dict['<STOP>']]
        terminal_var=torch.unsqueeze(terminal_var,0)
        alpha=torch.logsumexp(terminal_var,dim=1)[0]
        return alpha
    
    def _score_sequence(self,feats,labels): # calulate score for gold sequence
        score=torch.zeros(1)
        labels=torch.cat([torch.tensor([self.labels_dict['<START>']],dtype=torch.long),labels])
        for i,feat in enumerate(feats):
            score=score+\
                self.transition_mat[labels[i+1],labels[i]] + feat[labels[i+1]]
        score+=self.transition_mat[self.labels_dict['<STOP>'],labels[-1]]
        return score
    
    def _get_lstm_features(self,x,predict=False): # x:[batch_size,seq_len]
        if predict:
            self.hidden=self.init_hidden(predict=predict)
            embeddings=self.word_embeds(x).view(x.size(1),1,-1)
            lstm_out,self.hidden=self.lstm(embeddings,self.hidden)
            lstm_out=lstm_out.view(len(x),self.hidden_dim)
            lstm_feats=self.hidden2label(lstm_out)
            return lstm_feats

        embeddings=self.word_embeds(x).view(x.size(1),self.batch_size,-1)
        lstm_out,self.hidden=self.lstm(embeddings,self.hidden) # lstm_out:[seq_len,batch_size,hidden_dim]
        lstm_out=lstm_out.premute(1,0,2) # [batch_size,seq_len,hidden_dim]

        lstm_feats=self.hidden2label(lstm_out)
        return lstm_feats
    
    def _viterbi_decode(self,feats):
        backpointers=[]

        init_vvars=torch.full((1,self.labelset_size),-10000.)
        init_vvars[0][self.labels_dict['<START>']]=0.

        forward_var=init_vvars
        for feat in feats:
            bptrs_t=[] # store the backpointers for this step
            viterbivars_t=[] # store the viterbi variables for this step

            for next_label in range(self.labelset_size):
                next_label_var=forward_var+self.transition_mat[next_label]
                best_label_id=argmax(next_label_var)
                bptrs_t.append(best_label_id) 
                viterbivars_t.append(next_label_var[0][best_label_id].view(1))
            
            # add the emission score -- feat
            forward_var=(torch.cat(viterbivars_t)+feat).view(1,-1)
            # store the optimal path until step t to each possible label
            backpointers.append(bptrs_t)
        
        terminal_var=forward_var+self.transition_mat[self.labels_dict['<STOP>']]
        # final best label id
        best_label_id=argmax(terminal_var)
        best_score=terminal_var[0][best_label_id]

        # decode the best path
        best_path=[best_label_id]
        for bptrs in reversed(backpointers):
            best_label_id=bptrs[best_label_id]
            best_path.append(best_label_id)
        
        start=best_path.pop()
        best_path.reverse()
        return best_score,best_path
    
    # reference:https://github.com/mali19064/LSTM-CRF-pytorch-faster/blob/master/LSTM_CRF_faster.py
    def _viterbi_decode_faster(self,feats):
        backpointers=[]

        init_vvars=torch.full((1,self.labelset_size),-10000.)
        init_vvars[0][self.labels_dict['<START>']]=0.

        forward_var_li=[init_vvars]
        for feat_idx in range(feats.size(0)):
            forward_var=torch.stack([forward_var_li[feat_idx]]*feats.size(1))
            forward_var=torch.squeeze(forward_var)
            next_label_var=forward_var+self.transition_mat
            viterbivars_t,bptrs_t=torch.max(next_label_var,dim=1)

            feat=torch.unsqueeze(feats[feat_idx],0)
            forward_var_new=torch.unsqueeze(viterbivars_t,0)+feat

            forward_var_li.append(forward_var_new)
            backpointers.append(bptrs_t.tolist())
        
        terminal_var=forward_var_li[-1]+self.transition_mat[self.labels_dict['<STOP>']]
        best_label_id=torch.argmax(terminal_var).tolist()
        best_score=terminal_var[0][best_label_id]

        best_path=[best_label_id]
        for bptrs in reversed(backpointers):
            best_label_id=bptrs[best_label_id]
            best_path.append(best_label_id)
        
        start=best_path.pop()
        best_path.reverse()
        return best_score,best_path
    
    # batch loss
    def neg_log_likelihood(self,x,y): # (x,y) -> one batch
        feats=self._get_lstm_features(x)
        scores=[]
        for i in range(feats.size(0)):
            forward_score=self._forward_alg_faster(feats[i])
            gold_score=self._score_sequence(feats[i],y[i])
            scores.append(forward_score-gold_score)
        return torch.sum(torch.tensor(scores))
    
    def forward(self,x): # x -> one sequence

        lstm_feats=self._get_lstm_features(x)
        score,label_seq=self._viterbi_decode_faster(lstm_feats)
        return score,label_seq
    





    
    














