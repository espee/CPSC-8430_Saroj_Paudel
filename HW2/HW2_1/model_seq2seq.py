## Homework HWK2
## Implementation of a sequence2sequence model for caption generation for a sequence of video
## Uses GRU for encoding and decoding
## Attention mechanism implemented
## Beam search method with width of 2/


import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import re
import random
import numpy as np
import json
import pandas as pd
import pickle
import math
import sys

if len(sys.argv) < 2 or len(sys.argv) > 3:
    print("ERROR: Check argument, python [model.py] [folder_name] [output.txt]")
    exit()

#input data folder and output data   
data_folder = sys.argv[1]
output_file = sys.argv[2]

#    DATA FOLDERS  : Please make changes here for the TA review data # 

# training_labels = data_folder + '/training_label.json'
testing_labels = data_folder + '/testing_label.json'
# training_features = data_folder+'/training_data/feat/'
testing_features = data_folder + '/testing_data/feat/'

#testing_features = data_folder + '/ta_review_data/feat/'
 
#model evaluation, List of filenames of video input .avi
test_id_file = data_folder + '/testing_data/id.txt'
#test_id_file = data_folder + '/ta_review_data/id.txt'


#model parameters
max_caption_words = 20
input_steps = 80
output_steps = max_caption_words
feature_size = 4096
hidden_size = 640
embedding_size = 512

# preferred pytorch device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using " + device + " device") 
#device = "cpu"

#caption preprocessing to get rid of unnecessary characters. 
def caption_preprocess(caption):
    caption = caption.lower().strip()  #converts all to lowercase, remove leading and trailing white spaces
    caption = re.sub(r"([?.!,¿])", r" \1 ", caption) #replace each character specified within [ and ] by space char space
    caption = re.sub(r'[" "]+', " ", caption) #replaces each characters specified within [ and ] by space
    caption = re.sub(r"[^a-zA-Z?.!,¿]+", " ", caption) #replace all except the characters specified by space
    caption = caption.strip() ##remove leading and trailing white spaces
    return caption

#Generate vocabulary from captions
class Vocabulary():
    def __init__(self, captions_dict, minimum_occurance = 3):
        #convert all to 1D list of captions
        captions = [caption for captions in list(captions_dict.values()) for caption in captions]
        
        #compute word counts and filter low frequency words
        word_counts = {}
        for caption in captions:
            words = caption.split()
            for word in words:
                if word not in word_counts:
                    word_counts[word] = 1
                else:
                    word_counts[word] += 1
                    
        #Sort and filter out with frequency less than 3
        self.vocabulary = sorted([word for word, count in word_counts.items() if count>=minimum_occurance])
        
        #Add custom tokens '<BOS>', '<EOS>', '<UNK>', '<PAD>'
        self.vocabulary += ['<BOS>', '<EOS>', '<UNK>', '<PAD>']
        
        #build index 
        vocab_size = len(self.vocabulary)
        self.itos = {i: self.vocabulary[i] for i in range(vocab_size)}
        self.stoi = {self.vocabulary[i]:i for i in range(vocab_size)}
        
        

#Class for building vocabulary object from the caption labels
class VideoToCaptionDataset(Dataset):
    def __init__(self, labels_file, features_file, max_caption_words, vocabulary = None):
        self.max_caption_words = max_caption_words
        self.video_id_caption_pairs = []
        self.video_id_to_features = {}
        
        with open(labels_file, 'r') as f:
            label_data = json.load(f)
        captions_dict = {}
        
        for i in range(len(label_data)):
            captions_dict[label_data[i]['id']] = [caption_preprocess(caption) for caption in label_data[i]['caption']]
        
        if vocabulary ==None:
            print("Building vocabulary for the first time...", end = ' ')
            self.vocab = Vocabulary(captions_dict)
            print("Vocabulary built with Vocab size" + str(len(self.vocab.vocabulary)))
        else:
            print("Vocabulary exists: Using existing vocabulary")
            self.vocab = vocabulary
            
        #print("Building the Datasets: Extract features and caption pairs from video ID")
        
        for video_id, captions in captions_dict.items():
            
            #Extract the features
            self.video_id_to_features[video_id] = torch.FloatTensor(np.load(features_file + video_id + ".npy"))
            
            #Extract the captions
            for caption in captions:
                processed_caption = ['<BOS>']  #add BOS at the beginning
                
                #Extract words from caption
                for word in caption.split():
                    if word in self.vocab.vocabulary:
                        processed_caption.append(word)
                    else:
                        processed_caption.append('<UNK>')
                        
                if len(processed_caption)+1 > self.max_caption_words:
                    processed_caption.append('<EOS>')
                    continue
                
                #pad to max caption size
                processed_caption += ['<PAD>'] * (self.max_caption_words-len(processed_caption))
                processed_caption = ' '.join(processed_caption) #joines all caption words with space in between
                
                #Video ID and Caption pair
                self.video_id_caption_pairs.append((video_id, processed_caption))
                
        print("Completed. Total number of examples: " +str(len(self.video_id_caption_pairs)))
        
    def __len__(self):
        return len(self.video_id_caption_pairs)
    
    def __getitem__(self, idx):
        # retrieve caption and features for given index
        video_id, caption = self.video_id_caption_pairs[idx]
        feature = self.video_id_to_features[video_id]
        
        # compute one-hot tensor for the caption 
        word_ids = torch.LongTensor([self.vocab.stoi[word] for word in caption.split(' ')])
        caption_one_hot = torch.LongTensor(self.max_caption_words, len(self.vocab.vocabulary))
        caption_one_hot.zero_()
        caption_one_hot.scatter_(1, word_ids.view(self.max_caption_words, 1), 1)
        
        return {'feature': feature, 'caption_one_hot': caption_one_hot, 'caption': caption}
           
        
#Creating training set
def create_trainset():
    batch_size = 64
    print("Generating training set")
    trainset = VideoToCaptionDataset(training_labels, training_features, max_caption_words)
    trainset_loader = DataLoader(trainset, batch_size = batch_size)
    
    return trainset, trainset_loader



#### Sequence to Sequence model class
class Seq2SeqModel(nn.Module):
    def __init__(self, vocabulary, input_steps, output_steps, feature_size, hidden_size, embedding_size, n_layers=1, dropout = 0.25):
        super(Seq2SeqModel, self).__init__()
        self.vocab = vocabulary
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.feature_size = feature_size
        self.fc_size = embedding_size 
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size 
        vocab_size = len(self.vocab.vocabulary)

        self.fc = nn.Linear(feature_size, self.fc_size)
        self.dropout = nn.Dropout(p = dropout)
        
        #Encoder implemented using GRU
        self.encoder = nn.GRU(self.fc_size, hidden_size, n_layers)

        #attention mechanism
        self.attention = Attention(hidden_size)

        #Decoder implemented using GRUs
        self.decoder = nn.GRU(hidden_size*2+embedding_size, hidden_size, n_layers)

        #Output layer. Output vector size is the size of the vocab. 
        self.out = nn.Linear(hidden_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.embedding = nn.Embedding(vocab_size, embedding_size)

    
    def forward(self, input_seq, output_seq, teacher_forcing_ratio = -1):
        #Initialize Loss
        loss = 0.0
        batch_size = input_seq.shape[1]
        
        input_seq = self.dropout(F.leaky_relu(self.fc(input_seq))) 
        encoder_padding = Variable(torch.zeros(self.output_steps, batch_size, self.fc_size)).to(device)
        #Concatenate
        encoder_input = torch.cat((input_seq, encoder_padding), 0)
        encoder_output, _ = self.encoder(encoder_input)     # GRU encoder
        
    
        decoder_padding = Variable(torch.zeros(self.input_steps, batch_size, self.hidden_size+self.embedding_size)).to(device) 
        #concatenate
        first_decoder_input = torch.cat((decoder_padding, encoder_output[:self.input_steps, :, :]), 2)
        first_decoder_output, z = self.decoder(first_decoder_input) #GRU decoder
        
        caption_embedded = self.embedding(output_seq)
        bos = [self.vocab.stoi['<BOS>']] * batch_size
        bos = Variable(torch.LongTensor([bos])).resize(batch_size, 1).to(device)
        
        
        #iterate prediction and update over output steps
        for output_step in range(self.output_steps):
            
            if output_step ==0:
                decoder_input = self.embedding(bos)
            elif random.random() <= teacher_forcing_ratio:
                decoder_input = caption_embedded[:, output_step-1, :].unsqueeze(1)
            else:
                decoder_input = self.embedding(decoder_output.max(1)[-1].resize(batch_size, 1))
                
            #Attention model
            #Calculation of attention weights
            attention_weights = self.attention(z, encoder_output[:self.input_steps])
            #Context vector generation
            context = torch.bmm(attention_weights.transpose(1,2), encoder_output[:self.input_steps].transpose(0,1))
            
            #decoder input concatenated with context vector
            decoder_input_with_a = torch.cat((decoder_input, encoder_output[self.input_steps+output_step].unsqueeze(1), context), 2). transpose(0,1)
            decoder_output, z = self.decoder(decoder_input_with_a, z)
            #Softmax output
            decoder_output = self.softmax(self.out(decoder_output[0]))
            
            #Loss update
            loss += F.nll_loss(decoder_output, output_seq[:, output_step]) / self.output_steps
            
        return loss
    
    
    
    
    def predict(self, input_seq, beam_width = 1):
        output_seq = []
        input_seq = F.leaky_relu(self.fc(input_seq))
        #padding and concatenate
        encoder_padding = Variable(torch.zeros(self.output_steps, 1, self.fc_size)).to(device)    
        encoder_input = torch.cat((input_seq, encoder_padding), 0)
        encoder_output, _ = self.encoder(encoder_input)
        
        #decoder padding and concatenate
        decoder_padding = Variable(torch.zeros(self.input_steps, 1, self.hidden_size+self.embedding_size)).to(device) 
        first_decoder_input = torch.cat((decoder_padding, encoder_output[:self.input_steps, :, :]), 2)
        first_decoder_output, z = self.decoder(first_decoder_input) 
        
        #define BOS as decoder first output
        bos = [self.vocab.stoi['<BOS>']]
        bos = Variable(torch.LongTensor([bos])).resize(1, 1).to(device)
        
        #Beam search 
        if beam_width >1:
            candidates = []
            #iterate output steps
            for output_step in range(self.output_steps):
                if output_step == 0:
                    decoder_input = self.embedding(bos)
                    attention_weights = self.attention(z, encoder_output[:self.input_steps])
                    context = torch.bmm(attention_weights.transpose(1,2), encoder_output[:self.input_steps].transpose(0,1))
                    
                    second_decoder_input = torch.cat((decoder_input, encoder_output[self.input_steps+output_step].unsqueeze(1), context),2).transpose(0,1)
                    decoder_output, z = self.decoder(second_decoder_input, z)
                    decoder_output = self.softmax(self.out(decoder_output[0]))
                    prob = math.e**decoder_output
                    
                    #select high probability
                    top_k_candidates , top_k_ids = prob.topk(beam_width)
                    top_k_scores = top_k_candidates.data[0].cpu().numpy().tolist()
                    candidates = top_k_ids.data[0].cpu().numpy().reshape(beam_width,1).tolist()
                    zs = [z] * beam_width
                else:
                    new_candidates = []
                    for i, candidate in enumerate(candidates): #iterate on old candidates
                        decoder_input = Variable(torch.LongTensor([candidate[-1]])).to(device).resize(1,1)
                        decoder_input = self.embedding(decoder_input)
                        attention_weights = self.attention(z, encoder_output[:self.input_steps])
                        context = torch.bmm(attention_weights.transpose(1,2), encoder_output[:self.input_steps].transpose(0,1))
                        second_decoder_input = torch.cat((decoder_input, encoder_output[self.input_steps+output_step].unsqueeze(1), context),2).transpose(0,1)
                        decoder_output, z = self.decoder(second_decoder_input, z)
                        decoder_output = self.softmax(self.out(decoder_output[0]))
                        prob = math.e**decoder_output 
                        
                        top_k_candidates, top_k_ids = prob.topk(beam_width)
                        for k in range(beam_width):
                            score = top_k_scores[i] * top_k_candidates.data[0, k]
                            new_candidate = candidates[i] + [top_k_ids[0, k].item()] 
                            new_candidates.append([score, new_candidate, zs[i]])
                            
                    #selet high probability
                    new_candidates = sorted(new_candidates, key = lambda x:x[0], reverse=True)[:beam_width]
                    top_k_scores = [candidate[0] for candidate in new_candidates]
                    candidates = [candidate[1] for candidate in new_candidates]
                    zs = [candidate[2] for candidate in new_candidates]
                    
            token_indices = [self.vocab.stoi[t] for t in ['<BOS>', '<EOS>', '<PAD>', '<UNK>']]
            output_seq = [self.vocab.itos[int(word_index)] for word_index in candidates[0] if int(word_index) not in token_indices]
            print(output_seq)
            return output_seq
            
        else:
            
            pred_seq = []
            for output_step in range(self.output_steps):
                if output_step == 0: 
                    decoder_input = self.embedding(bos)
                else:
                    decoder_input = self.embedding(decoder_output.max(1)[-1].resize(1, 1))

                attention_weights = self.attention(z, encoder_output[:self.input_steps])
                context = torch.bmm(attention_weights.transpose(1, 2),
                                   encoder_output[:self.input_steps].transpose(0, 1))

                second_decoder_input = torch.cat((decoder_input, encoder_output[self.input_steps+output_step].unsqueeze(1), context),2).transpose(0,1)

                decoder_output, z = self.decoder(second_decoder_input, z)
                decoder_output = self.softmax(self.out(decoder_output[0]))

                token_indices = [self.vocab.stoi[t] for t in ['<EOS>', '<PAD>', '<UNK>']]
                output_id = decoder_output.max(1)[-1].item()
                if output_id in token_indices:
                    break
                elif output_id != self.vocab.stoi['<BOS>']:
                    pred_seq.append(self.vocab.itos[int(output_id)])
            
            return pred_seq
        
        
        

## Attention mechanism: 
#Reference: https://github.com/PatrickSVM/Seq2Seq-with-Attention/blob/main/seq2seq_attention/preprocess.py
class Attention(nn.Module):
    def __init__(self, hidden_size, dropout = 0.25):
        super(Attention, self).__init__()
        self.Attention = nn.Linear(hidden_size*2, 1) 
        #Some examples use 
        #self.layer1 = nn.Linear(hidden_size_enc*2+ hidden_size_dec, hidden_size_dec)
        #self.Layer2 = nn.Linear(hidden_size_dec, 1)
        self.dropout = nn.Dropout(p=dropout)
        self.hidden_size = hidden_size
        
    def forward(self, hidden, encoder_outputs):
        attention_output = torch.bmm(encoder_outputs.transpose(0,1), hidden.transpose(0,1).transpose(1,2))
        attention_output = F.tanh(attention_output)
        attention_weights = F.softmax(attention_output, dim=1)
        return attention_weights
    

def train(trainset_loader, model, epochs = 3, lr=1e-3):
    
    optimizer = optim.Adam(model.parameters(), lr = lr)
    
    training_loss = []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
               
        for batch, data in enumerate(trainset_loader):
            X_train = data['feature'].transpose(0,1).to(device)
            Y_train = data['caption_one_hot'].to(device)
            optimizer.zero_grad() #reset gradient
            loss = model(X_train, Y_train.max(2)[-1], teacher_forcing_ratio = 0.1)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()/len(trainset_loader)
        
        training_loss.append(epoch_loss)
        
        #save the final model
        if(epoch == (epochs-1)):
            torch.save(model.state_dict(), "seq2seqmodel"+str(epoch+1))
        print("Epoch " + str(epoch) + ", Training loss: " + str(epoch_loss) )
    return training_loss
            

def train_model(epoch):
    trainset, trainset_loader = create_trainset()
    pickle.dump(trainset.vocab, open('vocabulary.pickle', 'wb'))
    seq2seqmodel = Seq2SeqModel(trainset.vocab, input_steps, output_steps, feature_size, hidden_size, embedding_size).to(device)
    print("Training Seq2Seq model Started...")
    training_loss = train(trainset_loader, seq2seqmodel, epoch)

    return training_loss
                        
                        
def postprocess_caption(caption):
    caption = caption.capitalize().replace(' .',"")  #Capitalize only the first character and replace any . by null
    return caption

def test(trained_model, vocabulary_file, testing_features, test_id_file, output_file):
    vocabulary = pickle.load(open(vocabulary_file, "rb"))
    seq2seqmodel = Seq2SeqModel(vocabulary, input_steps, output_steps, feature_size, hidden_size, embedding_size).to(device)
    seq2seqmodel.load_state_dict(torch.load(trained_model))
    seq2seqmodel.to(device)
    
    input_data = {}
    test_label = pd.read_fwf(test_id_file, header = None)
    for _, row in test_label.iterrows():
        feature_file = f"{testing_features}{row[0]}.npy"
        input_data[row[0]] = torch.FloatTensor(np.load(feature_file))
        
    seq2seqmodel.eval()
    predictions = []
    indices = []
    for _, row in test_label.iterrows():
        input_seq = Variable(input_data[row[0]].view(-1,1,feature_size)).to(device)
        pred = seq2seqmodel.predict(input_seq, beam_width = 2)
        pred = postprocess_caption(" ".join(pred))
        predictions.append(pred)
        indices.append(row[0])
        
    #save to txt file
    with open(output_file, 'w') as result_file:
        for i, _ in test_label.iterrows():
            result_file.write(indices[i] + "," + predictions[i] + "\n")
            

###   Train model  ###
#uncomment below 2 lines to train the model
epoch = 50
#train_model(epoch)



###     Testing     ###

#Load trained model (2 models)
#trained_model = "seq2seqmodel"+str(epoch)+"_BeamSearch2"
trained_model = "seq2seqmodel"+str(epoch)+"_Greedy"
#Load vocabulary data
vocab_file = "vocabulary.pickle"

#Test model
test(trained_model, vocab_file, testing_features, test_id_file, output_file)

                             
                                 


