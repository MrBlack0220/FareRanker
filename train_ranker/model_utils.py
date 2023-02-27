
from hmac import new
import token
from turtle import pos
from transformers import RobertaModel, RobertaPreTrainedModel, LongformerModel, RobertaConfig, LongformerConfig, LongformerPreTrainedModel, BartPretrainedModel, BartModel
from transformers.models.bart.modeling_bart import BartEncoder
from transformers import PreTrainedModel
import torch
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.nn import MarginRankingLoss, BCEWithLogitsLoss
from torch.nn.parameter import Parameter
from packaging import version
import numpy as np
import torch.nn.functional as F

class UnixcoderRankerHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 1)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        
    def forward(self,features,mask,batch_size,candidate_num,**kwargs):
        """
        features: (B*C, L , D) 
        """
       
        sentence_embeddings = (features * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1)  # (B*C,D)
        sentence_embed = sentence_embeddings.view(batch_size, candidate_num, -1) # (B, C, D)
        x = self.dropout(sentence_embed)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x) # (B, C, 1)
        logits = x.squeeze(-1)  # (B,C)
        return logits,sentence_embed


class UnixcoderRanker(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels  # 2
        # self.loss_type = config.loss_type  # 'contrastive'
        self.config = config
        self.roberta = RobertaModel(config, add_pooling_layer=True)   
        # self.register_buffer("bias", torch.tril(torch.ones((1024, 1024), dtype=torch.uint8)).view(1,1024, 1024))  #???
        self.classifier = UnixcoderRankerHead(config)
        self.init_weights()  
        

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        step_type = None,  # TODO:
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        input_ids: (B, C, L) B for batch size, C for number of candidates, L for sequence length
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # input_ids[B,C,L]
        batch_size, candidate_num, seq_len = input_ids.size()
        
        input_ids = input_ids.view(-1, input_ids.size(-1)) #(B * C, L)
        attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = outputs[0] # (B*C, L, D)


        logits,sentence_embed = self.classifier(features = sequence_output,mask = attention_mask,batch_size= batch_size,candidate_num = candidate_num)
        if step_type == "step1":
            loss = None
            logit_soft = F.softmax(logits, dim = -1)
            pos_probs = logit_soft[:, 0]
            loss = -torch.log(pos_probs).mean()
        if step_type == "step2":
            
            sentence_embeddings_norm  = torch.nn.functional.normalize(sentence_embed, dim=2) # (B,C,D)  
            
            index1 = torch.arange(1,candidate_num).to(int).to(logits.device)
            contrast_features1 = torch.index_select(sentence_embeddings_norm,dim=1,index=index1) 
            anchor1 = torch.index_select(sentence_embeddings_norm,dim=1,index=torch.tensor([0]).to(logits.device)).permute(0,2,1)
            logits1 = torch.matmul(contrast_features1,anchor1) / self.config.temperature # B,C-1,1
            logits1 = logits1.squeeze(-1) # B,C-1
            logit_soft1 = F.softmax(logits1, dim = -1)
            pos_probs1 = logit_soft1[:, 0]
            loss1 = -torch.log(pos_probs1).mean()
            
            index2 = torch.cat((torch.zeros(1).to(int),torch.arange(2,candidate_num).to(int))).to(logits.device)
            contrast_features2 = torch.index_select(sentence_embeddings_norm,dim=1,index=index2) 
            anchor2 = torch.index_select(sentence_embeddings_norm,dim=1,index=torch.tensor([1]).to(logits.device)).permute(0,2,1) 
            logits2 = torch.matmul(contrast_features2,anchor2) / self.config.temperature # B,C-1,1
            logits2 = logits2.squeeze(-1) # B,C
            logit_soft2 = F.softmax(logits1, dim = -1)
            pos_probs2 = logit_soft2[:, 0]
            loss2 = -torch.log(pos_probs2).mean()

            logits3 = torch.index_select(logits,dim=1,index=index1) 
            logit_soft3 = F.softmax(logits3, dim = -1)  
            pos_probs3 = logit_soft3[:, 0] 
            loss3 = -torch.log(pos_probs3).mean()  
            
            logits4 = torch.index_select(logits,dim=1,index=index2)  
            logit_soft4 = F.softmax(logits4, dim = -1)
            pos_probs4 = logit_soft4[:, 0]
            loss4 = -torch.log(pos_probs4).mean()
            
            loss = (loss2+loss1+loss3+loss4)/4
        
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=None,
            hidden_states=None,
            attentions=None,
        )
        
        
    def evaluate(self,
        input_ids=None,
        attention_mask=None,
        step_type = None,  # TODO:
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None):
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size, candidate_num, seq_len = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1)) #(B * C, L)
        attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask, #TODO
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0] # (B*C, L, D)


        logits,sentence_embed = self.classifier(features = sequence_output,mask = attention_mask,batch_size= batch_size,candidate_num = candidate_num) # (B, C)
        logits = logits.sum(dim=1) # (B,)

        return logits


    def _resize_position_embedding(self, new_max_position_embedding, extend_way = 'normal'):
        '''
            resize the model's position embedding to match the new positional embedding
            should also update the config.max_position_ebmddings here for avoiding error when loading finetuned checkpoints
            should also change the token_type_ids registed in embedding layers
        '''
        # self.roberta.embeddings.position_embeddings
        old_weights = self.roberta.embeddings.position_embeddings.weight # tensor  (Seq_len,d)
        old_num_emb, embedding_dim = old_weights.size() # (1026,768)

        if extend_way == 'normal':
            # initialize new weight in normal
            added_weights = torch.empty((new_max_position_embedding - old_num_emb, embedding_dim), requires_grad=True)  # torch.Size([93, 768])
            nn.init.normal_(added_weights)
            new_weights = torch.cat((old_weights, added_weights), 0)  # ([1119, 768])
        elif extend_way == 'copy':
            # initialize new weight by copying from the old weights
            # to be implemented
            len_to_extend = new_max_position_embedding - old_num_emb
            old_weight_np = old_weights.detach().numpy()

            added_weights = np.array(old_weight_np[2: len_to_extend % (old_num_emb - 2) +2])
            for _ in range(len_to_extend // (old_num_emb - 2) ):
                added_weights = np.concatenate(added_weights, np.array(old_weight_np[2:]))
            
            added_weights = torch.Tensor(added_weights)
            added_weights.requires_grad = True
            new_weights = torch.cat((old_weights, added_weights), 0)

        self.roberta.embeddings.position_embeddings = nn.Embedding(new_max_position_embedding, embedding_dim,
                                                    padding_idx = self.roberta.embeddings.position_embeddings.padding_idx,
                                                    _weight = new_weights)

        self.config.max_position_embeddings = new_max_position_embedding