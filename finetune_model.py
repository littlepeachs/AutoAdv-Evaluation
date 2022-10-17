from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import numpy as np
from datasets import load_from_disk
import torch
from torch import optim
from torch.nn import CrossEntropyLoss
import pandas as pd
import argparse
from tqdm import tqdm
from opendelta import AdapterModel
from openprompt.plms import load_plm
from openprompt.data_utils import InputExample
from openprompt.prompts import ManualTemplate
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptForClassification
from openprompt import PromptDataLoader
import openpyxl

def preprocess_data_json(path): # to set label and delete the illegal data
    df = pd.read_json(path,lines=True)
    df = df.drop(df[df['gold_label']=='-'].index)
    df.loc[df['gold_label']=="entailment",'gold_label']=2
    df.loc[df['gold_label']=="contradiction",'gold_label']=0
    df.loc[df['gold_label']=="neutral",'gold_label']=1
    return df

def preprocess_data_tsv(path):
    df = pd.read_csv(path, sep='\t')
    return df

def preprocess_data_excel(path):
    df = pd.read_excel(path)
    return df

def process_label(df): # because the qqp is 2-classification task,we need label=2 change to label=1
    df.loc[(df['gold_label']==2),'gold_label']=1
    return df

def base_freeze(): #freeze the layer use simple way
    need_frozen_list=["classifier.dense.bias","classifier.dense.weight"]
    for param in model.named_parameters():
        if param[0] in need_frozen_list:
            param[1].requires_grad = False
    
def delta_freeze(backbone_model):
    delta_model = AdapterModel(backbone_model=backbone_model,modified_modules=['classifier.dense','classifier.out_proj'])
    delta_model.freeze_module(exclude=["classifier.dense",'classifier.out_proj'],set_state_dict=True)
    # delta_model.log(delta_ratio=True, trainable_ratio=True, visualization=True)
    return delta_model

def get_batch(datas,attention_masks,labels,i,bsz): 
    max_len = len(labels)
    if(max_len<=i+bsz):
        data = torch.tensor(datas[i:])
        attention_mask = torch.tensor(attention_masks[i:i+bsz])
        label = torch.tensor(labels[i:])
        # label-=1
        return data.to(device) ,attention_mask.to(device),label.to(device)
    else:
        data = torch.tensor(datas[i:i+bsz])
        attention_mask = torch.tensor(attention_masks[i:i+bsz])
        label = torch.tensor(labels[i:i+bsz])
        # label-=1
        return data.to(device) ,attention_mask.to(device),label.to(device)

def accuracy(preds,labels):
    preds = np.argmax(preds,axis=1).flatten()
    labels = labels.flatten()
    acc=np.sum(labels==preds)/len(preds)
    return acc

def read_whole_data(df): 
    premise = np.array(df["sentence1"][:]).tolist() #sentence1
    hypothesis = np.array(df["sentence2"][:]).tolist() #sentence2
    label = np.array(df["gold_label"][:]).tolist() #gold_label
    data_size=len(label)
    return premise,hypothesis,label,data_size

def train_test_split(df,size,num_labels):
    few_shot_ls = []
    if num_labels==3:
        for i in range(num_labels):
            temp_ls=[]
            if i!=1: #specail for the own data
                data = df[df['gold_label'].isin([i])]
                data_sample = data.sample(size,random_state=random_seed)
                temp_ls=data_sample.index.tolist()
                for j in range(len(temp_ls)):
                    df=df.drop(df[(df.index==temp_ls[j])].index)
                few_shot_ls.append(data_sample)
    elif num_labels==2:
        for i in range(num_labels):
            temp_ls=[]
            data = df[df['gold_label'].isin([i])]
            data_sample = data.sample(size,random_state=random_seed)
            temp_ls=data_sample.index.tolist()
            for j in range(len(temp_ls)):
                df=df.drop(df[(df.index==temp_ls[j])].index)
            few_shot_ls.append(data_sample)
    train_dataset = pd.concat(few_shot_ls)
    train_dataset = train_dataset.sample(frac=1)
    return train_dataset,df

def extract_elem(df):
    premise = np.array(df["sentence1"][:]).tolist()
    hypothesis = np.array(df["sentence2"][:]).tolist()
    label = np.array(df["gold_label"][:]).tolist()
    data_size = len(label)
    return premise,hypothesis,label,data_size

def tokenize_data(premise,hypothesis,tokenizer):
    data = tokenizer(premise[:],hypothesis[:],padding="max_length",truncation=True, return_tensors="pt",max_length=512)
    data_id = data['input_ids']
    masks = data["attention_mask"] 
    return data_id,masks


def train(df_train,train_size,num_labels): #in the base learning, do not use dataloader
    if(train_size==0):
        return
    premise,hypothesis,train_label,train_data_size = extract_elem(df_train)
    train_data_id,train_masks = tokenize_data(premise,hypothesis,tokenizer)
    for epoch in range(epochs):
        print("epoch: {}".format(epoch+1))
        if(train_size==0):
            return
        total_loss=0
        count = 0
        for i in range(0,train_data_size,batch_size): #use val_data to get fast test
            data,attention_mask,label = get_batch(train_data_id,train_masks,train_label,i,batch_size)
            model.zero_grad()
            output = model(data,attention_mask=attention_mask)
            output = output["logits"]
            # print(output[:,1])
            
            loss = criterion(output,label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),0.25)
            optimizer.step()
            total_loss+=loss.item()
            count+=1
            if count%1000==0:
                print("the {}th data".format(i))
                print("temp_train_loss: {}".format(total_loss/(i*batch_size)))
            
        total_loss/=train_data_size
        print("train_loss: {}".format(total_loss))

def evaluate(num_labels): 
    eval_batch_size =10
    model.eval()
    total_loss=0.0
    eval_step=0
    eval_acc = 0.0
    premise,hypothesis,labels,data_size =read_whole_data(test_df)
    data_id,masks = tokenize_data(premise,hypothesis,tokenizer)
    with torch.no_grad():
        for i in range(0,data_size,eval_batch_size):
            data,attention_mask,label = get_batch(data_id,masks,labels,i,eval_batch_size)
            output=model(data,attention_mask=attention_mask)
            output = output["logits"]
            if num_labels==3:
                output[:,1] = float("-inf")
            loss = criterion(output,label)
            output=output.detach().to('cpu').numpy()
            label=label.to('cpu').numpy()
            eval_acc+=accuracy(output,label)
            total_loss+=loss
            eval_step+=1
    total_loss/=len(data_id)
    print("val_loss: {}".format(total_loss))
    print("val_acc: {}".format(eval_acc/eval_step))
    return eval_acc/eval_step

def prompt_preprocess(tokenizer,direct_prompt): #(plm,tokenizer)
    
    classes = ["contradiction","entailment"] #neutral
    
    promptTemplate = ManualTemplate(
        text = 'This is Semantic similarity detection task. {"placeholder":"text_a"}? {"mask"}. Yes, {"placeholder":"text_b"}',
        tokenizer = tokenizer,
    )
    promptVerbalizer = ManualVerbalizer(
        classes = classes,
        label_words = {
            "contradiction": ["Nonetheless"],
            # "neutral": ["Watch"],
            "entailment": ["Okay"]
        },
        tokenizer = tokenizer,
    )
    if direct_prompt==1:
        promptModel = PromptForClassification(
            template = promptTemplate,
            plm = plm,
            verbalizer = promptVerbalizer,
        )
        return promptTemplate,promptVerbalizer ,promptModel
    else:
        return promptTemplate,promptVerbalizer

def prompt_data(premise,hypothesis,labels):
    dataset = []
    for i in range(len(premise)):
        input_example = InputExample(text_a =premise[i],text_b=hypothesis[i],guid=i,label=labels[i])
        dataset.append(input_example)
    return dataset

def prompt_dataloader(dataset,tokenizer,promptTemplate,WrapperClass):
    data_loader = PromptDataLoader(
        dataset = dataset,
        tokenizer = tokenizer,
        template = promptTemplate,
        tokenizer_wrapper_class=WrapperClass,
        batch_size=10,
        shuffle=True
    )   
    return data_loader

def prompt_train(train_df,train_size,num_labels):
    if(train_size==0):
        return
    if train_size==-1: # -1 mean train on whole dataset 
        premise,hypothesis,labels,data_size = read_whole_data(train_df)
    else:
        premise,hypothesis,labels,data_size = extract_elem(train_df)
    train_dataset = prompt_data(premise,hypothesis,labels)
    train_loader = prompt_dataloader(train_dataset,prompt_tokenizer,promptTemplate,WrapperClass)
    
    for i in range(epochs):
        total_loss=0
        pbar = tqdm(train_loader, desc=f"Epoch{i}")
        for step, inputs in enumerate(pbar):
            promptModel.zero_grad()
            inputs = inputs.to(device)
            label = inputs["label"]
            output = promptModel(inputs)
            loss = criterion(output,label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(promptModel.parameters(),0.25)
            optimizer.step()
            total_loss+=loss.item()
        total_loss/=data_size
        print("epoch:{}  train_loss: {}".format(i,total_loss))

        val_acc = prompt_evaluate(num_labels)


def prompt_evaluate(num_labels):
    eval_batch_size =10
    promptModel.eval()
    total_loss=0.0
    eval_step=0
    eval_acc = 0.0
    premise,hypothesis,labels,data_size =read_whole_data(test_df)
    test_dataset = prompt_data(premise,hypothesis,labels)
    test_loader = prompt_dataloader(test_dataset,prompt_tokenizer,promptTemplate,WrapperClass)
    with torch.no_grad():
        pbar = tqdm(test_loader)
        for step, inputs in enumerate(pbar):
            promptModel.zero_grad()
            inputs = inputs.to(device)
            label = inputs["label"]
            output = promptModel(inputs)
            if num_labels==3:
                output[:,1] = float("-inf")
            loss = criterion(output,label)
            output=output.detach().to('cpu').numpy()
            label=label.to('cpu').numpy()
            eval_acc+=accuracy(output,label)
            total_loss+=loss
            eval_step+=1
    total_loss/=data_size
    print("val_loss: {}".format(total_loss))
    print("val_acc: {}".format(eval_acc/eval_step))
    scheduler.step(total_loss)
    return eval_acc/eval_step

parser = argparse.ArgumentParser()
parser.add_argument('--train_size',type=int, default=16) #use as k-shot k=train_size
parser.add_argument('--freeze',default='direct_train',choices=['direct_train','direct_freeze'])#,'delta_tuning'
parser.add_argument("--prompt",default=1,type=int) #whether to prompt
parser.add_argument("--dataset_name",default="annotate_data/processed_data_char.xlsx")#annotate_data/processed_data.xlsx
parser.add_argument("--direct_prompt",default=0,type=int)# direct_prompt=1 means use roberta_prompt 
parser.add_argument("--random_seed",default=123,type=int)
parser.add_argument("--dataset_type",default='qqp',choices = ['mnli','qqp'])

params = parser.parse_args()
train_size = params.train_size
to_freeze=params.freeze
prompt = params.prompt
dataset_name_or_path = params.dataset_name
direct_prompt = params.direct_prompt
random_seed = params.random_seed
dataset_type = params.dataset_type
print(random_seed)

if dataset_type=='mnli':
    num_labels=3 # use in select few_shot_sample
elif dataset_type=='qqp':
    num_labels=2
epochs=10
save_path="roberta_large_qqp_train_on_all.pt"
batch_size=10
learning_rate=3e-5
attack_type = {"annotate_data/processed_data.xlsx":"word", "annotate_data/processed_data_sen.xlsx":"sen", "annotate_data/processed_data_char.xlsx":"char"}
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

criterion = CrossEntropyLoss()


###base method
if dataset_name_or_path=='snli_1.0' or dataset_name_or_path=='multinli_1.0':
    train_df=preprocess_data_json('{}/{}/{}_{}.jsonl'.format(dataset_name_or_path,dataset_name_or_path,dataset_name_or_path,'train'))
    test_df=preprocess_data_json('{}/{}/{}_{}_matched.jsonl'.format(dataset_name_or_path,dataset_name_or_path,dataset_name_or_path,'dev'))
elif dataset_name_or_path=='QQP':
    train_df=preprocess_data_tsv('QQP/{}.tsv'.format('train'))
    test_df=preprocess_data_tsv('QQP/{}.tsv'.format('dev'))
else:
    df = preprocess_data_excel(dataset_name_or_path)
    # df2 = preprocess_data_excel("annotate_data/processed_data_sen.xlsx")
    if dataset_type=='qqp':
        df = process_label(df)
    train_df,test_df = train_test_split(df,train_size,num_labels)
    # train_df,test_df = df2,df

#### mnli
# tokenizer = AutoTokenizer.from_pretrained("roberta-large-mnli")
# model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli",num_labels=3)

#### qqp
# tokenizer = AutoTokenizer.from_pretrained("howey/roberta-large-qqp")
# model = AutoModelForSequenceClassification.from_pretrained("howey/roberta-large-qqp")

###freeze
# if(to_freeze=='direct_freeze'):
#     base_freeze()
# elif (to_freeze=="delta_tuning"):
#     delta_model =delta_freeze(model)


##prompt
plm, prompt_tokenizer, model_config,WrapperClass = load_plm("roberta", "howey/roberta-large-qqp")
# plm = AutoModelForSequenceClassification.from_pretrained("roberta-large")
if direct_prompt==0:
    with open(save_path,'rb') as f:
        promptModel = torch.load(f)

## base method
whether_prompt=False
if prompt==0:
    whether_prompt=False
elif prompt==1:
    whether_prompt=True

if whether_prompt==False:
    print(whether_prompt)
    val_acc=0
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    model.to(device)
    
    train(train_df,train_size,num_labels)
    val_acc=evaluate(num_labels)
    with open("acc_in_few_shot_{}_on_{}.csv".format(to_freeze,dataset_type),'a') as f:
        if(train_size==0): #firstly write into csv file
            f.writelines("train_size,val_acc\n")
            f.writelines("random_seed,{}\n".format(random_seed))
        f.writelines("{},{}\n".format(train_size,val_acc))

##prompt

elif whether_prompt==True:
    print(whether_prompt)
    val_acc=0
    if direct_prompt==0:
        promptTemplate,promptVerbalizer=prompt_preprocess(prompt_tokenizer,direct_prompt)
    else:
        promptTemplate,promptVerbalizer,promptModel=prompt_preprocess(prompt_tokenizer,direct_prompt)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, promptModel.parameters()), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
    factor=0.2, patience=2, threshold=0.01, threshold_mode='abs')
    promptModel.to(device)
    prompt_train(train_df,train_size,num_labels)
    val_acc = prompt_evaluate(num_labels)
    with open("acc_{}_prompt_roberta_large_qqp.csv".format(attack_type[dataset_name_or_path]),'a') as f:                # "qqp_result/acc_in_117_prompt_roberta_large_qqp.csv"
        if(train_size==0): #firstly write into csv file
            f.writelines("train_size,val_acc\n")
            f.writelines("random_seed,{}\n".format(random_seed))
        f.writelines("{},{}\n".format(train_size,val_acc))

# with open(save_path,'wb') as f:
#     torch.save(promptModel,f)


# test_loss =evaluate(test_data,test_masks,test_label,1)