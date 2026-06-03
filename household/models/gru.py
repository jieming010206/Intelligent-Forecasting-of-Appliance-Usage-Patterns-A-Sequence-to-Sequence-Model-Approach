import torch.nn as nn
import torch
import numpy as np

class Encoder_embedding(nn.Module):
    def __init__(self):
        super(Encoder_embedding, self).__init__()
        self.year = nn.Embedding(2,1)
        self.month = nn.Embedding(12, 2)
        self.day_month = nn.Embedding(31, 4)
        self.day_week = nn.Embedding(7, 2)
        self.time = nn.Embedding(144, 30)
        self.appliance = nn.Embedding(11, 9)
        
    def forward(self, inputs):
        year = self.year(inputs[:, :, 0])
        month = self.month(inputs[:, :, 1])
        day_month = self.day_month(inputs[:, :, 2])
        day_week = self.day_week(inputs[:, :, 3])
        time = self.time(inputs[:, :, 4])
        appliance = self.appliance(inputs[:, :, 5])
        outputs = torch.cat((year, month, day_month, day_week, time, appliance), dim=2)
        return outputs
    
class Decoder_embedding(nn.Module):
    def __init__(self):
        super(Decoder_embedding, self).__init__()
        self.time = nn.Embedding(144, 48)
        self.appliance = nn.Embedding(11, 48)
        
    def forward(self, inputs):
        time = self.time(inputs[:, :, 0])
        appliance = self.appliance(inputs[:, :, 1])
        outputs = torch.cat((time, appliance), dim=2)
        return outputs
        
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, inputs):
        outputs, h_n = self.gru(inputs)
        return outputs, h_n

class Decoder(nn.Module):
    def __init__(self, hidden_dim, time_output_dim, appliance_output_dim):
        super(Decoder, self).__init__()
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.time_linear = nn.Linear(hidden_dim, time_output_dim)
        self.appliance_linear = nn.Linear(hidden_dim, appliance_output_dim)

    def forward(self, inputs, h_0):
        outputs, h_n = self.gru(inputs, h_0)
        time_outputs = self.time_linear(outputs)
        appliance_outputs = self.appliance_linear(outputs)
        return time_outputs, appliance_outputs, h_n

class Model(nn.Module):
    def __init__(self, time_dim, appliance_dim, output_seq_len):
        super(Model, self).__init__()
        self.output_seq_len = output_seq_len
        
        input_dim = 48
        hidden_dim = appliance_dim + time_dim
        self.encoder_embedding = Encoder_embedding()
        self.decoder_embedding = Decoder_embedding()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, time_dim, appliance_dim)
        self.linear1 = nn.Linear(appliance_dim + time_dim, hidden_dim)
        self.linear2 = nn.Linear(input_dim * 2, hidden_dim)
        
    def forward(self, inputs, targets=None, ratio = 0.3):
        encoder_inputs = self.encoder_embedding(inputs)
        encoder_outputs, h_n = self.encoder(encoder_inputs)

        time_outputs = []
        appliance_outputs = []
        decoder_inputs = inputs[:, -1:, -2:].long()
        
        for t in range(self.output_seq_len):
            decoder_inputs = self.decoder_embedding(decoder_inputs)
            decoder_inputs = self.linear2(decoder_inputs)
            decoder_time_outputs, decoder_appliance_outputs, h_n = self.decoder(decoder_inputs, h_n)
            time_outputs.append(decoder_time_outputs)
            appliance_outputs.append(decoder_appliance_outputs)
            
            if targets is not None and np.random.rand() <= ratio:  # teacher forcing
                decoder_inputs = targets[:, t : t + 1, :]
            else:
                decoder_inputs = torch.cat(
                    (   
                        torch.argmax(decoder_time_outputs, dim=2, keepdim=True),
                        torch.argmax(decoder_appliance_outputs, dim=2, keepdim=True),
                    ),
                    dim=2,
                )
        time_outputs = torch.cat(time_outputs, dim=1)
        appliance_outputs = torch.cat(appliance_outputs, dim=1)
        return time_outputs, appliance_outputs

if __name__ == "__main__":
    batch_size = 32
    input_dim = 6
    input_seq_len = 60 
    time_dim = 144
    appliance_dim = 11
    output_seq_len = 12

    inputs = torch.rand(batch_size, input_seq_len, 6).long()
    targets = torch.rand(batch_size, input_seq_len, 2).long()

    model = Model(time_dim, appliance_dim, output_seq_len)
    # appliance_outputs, duration_outputs = model(inputs, targets)
    time_outputs, appliance_outputs = model(inputs)
    print(time_outputs.size(), appliance_outputs.size())