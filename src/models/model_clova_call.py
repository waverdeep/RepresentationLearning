# Thanks to https://github.com/clovaai/ClovaCall
import math
import torch
import torch.nn as nn
import random
import numpy as np
import torch.nn.functional as F

if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device


class MaskConv(nn.Module):
    def __init__(self, seq_module):
        super(MaskConv, self).__init__()
        self.seq_module = seq_module

    def forward(self, x, lengths):
        """
        :param x: The input of size BxCxDxT
        :param lengths: The actual length of each sequence in the batch
        :return: Masked output from the module
        """
        for module in self.seq_module:
            x = module(x)
            mask = torch.BoolTensor(x.size()).fill_(0)
            if x.is_cuda:
                mask = mask.cuda()
            for i, length in enumerate(lengths):
                length = length.item()
                if (mask[i].size(2) - length) > 0:
                    mask[i].narrow(2, length, mask[i].size(2) - length).fill_(1)
            x = x.masked_fill(mask, 0)
        return x, lengths


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1,
                 input_dropout_p=0, dropout_p=0,
                 bidirectional=False, rnn_cell='gru', variable_lengths=False):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.variable_lengths = variable_lengths

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        """
        Copied from https://github.com/SeanNaren/deepspeech.pytorch/blob/master/model.py
        Copyright (c) 2017 Sean Naren
        MIT License
        """
        outputs_channel = 32
        self.conv = MaskConv(nn.Sequential(
            nn.Conv2d(1, outputs_channel, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(outputs_channel),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(outputs_channel, outputs_channel, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(outputs_channel),
            nn.Hardtanh(0, 20, inplace=True)
        ))

        rnn_input_dims = int(math.floor(input_size + 2 * 20 - 41) / 2 + 1)
        rnn_input_dims = int(math.floor(rnn_input_dims + 2 * 10 - 21) / 2 + 1)
        rnn_input_dims *= outputs_channel

        self.rnn = self.rnn_cell(rnn_input_dims, self.hidden_size, self.n_layers, dropout=self.dropout_p,
                                 bidirectional=self.bidirectional)

    def forward(self, input_var, input_lengths=None):
        """
        param:input_var: Encoder inputs, Spectrogram, Shape=(B,1,D,T)
        param:input_lengths: inputs sequence length without zero-pad
        """

        output_lengths = self.get_seq_lens(input_lengths)

        x = input_var  # (B,1,D,T)
        x, _ = self.conv(x, output_lengths)  # (B, C, D, T)

        x_size = x.size()
        x = x.view(x_size[0], x_size[1] * x_size[2], x_size[3])  # (B, C * D, T)
        x = x.permute(0, 2, 1).contiguous()  # (B,T,D)

        total_length = x_size[3]
        x = nn.utils.rnn.pack_padded_sequence(x,
                                              output_lengths.cpu(),
                                              batch_first=True,
                                              enforce_sorted=False)
        x, h_state = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x,
                                                batch_first=True,
                                                total_length=total_length)

        return x, h_state

    def get_seq_lens(self, input_length):
        seq_len = input_length
        for m in self.conv.modules():
            if type(m) == nn.modules.conv.Conv2d:
                seq_len = ((seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) / m.stride[1] + 1)

        return seq_len.int()


class Attention(nn.Module):
    """
    Location-based
    """
    def __init__(self, dec_dim, enc_dim, conv_dim, attn_dim, smoothing=False):
        super(Attention, self).__init__()
        self.dec_dim = dec_dim
        self.enc_dim = enc_dim
        self.conv_dim = conv_dim
        self.attn_dim = attn_dim
        self.smoothing= smoothing
        self.conv = nn.Conv1d(in_channels=1, out_channels=self.attn_dim, kernel_size=3, padding=1)

        self.W = nn.Linear(self.dec_dim, self.attn_dim, bias=False)
        self.V = nn.Linear(self.enc_dim, self.attn_dim, bias=False)

        self.fc = nn.Linear(attn_dim, 1, bias=True)
        self.b = nn.Parameter(torch.rand(attn_dim))

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

        self.mask = None

    def set_mask(self, mask):
        """
        Sets indices to be masked
        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, queries, values, last_attn):
        """
        param:quries: Decoder hidden states, Shape=(B,1,dec_D)
        param:values: Encoder outputs, Shape=(B,enc_T,enc_D)
        param:last_attn: Attention weight of previous step, Shape=(batch, enc_T)
        """
        batch_size = queries.size(0)
        dec_feat_dim = queries.size(2)
        enc_feat_len = values.size(1)

        # conv_attn = (B, enc_T, conv_D)
        conv_attn = torch.transpose(self.conv(last_attn.unsqueeze(dim=1)), 1, 2)

        # (B, enc_T)
        score =  self.fc(self.tanh(
         self.W(queries) + self.V(values) + conv_attn + self.b
        )).squeeze(dim=-1)


        if self.mask is not None:
            score.data.masked_fill_(self.mask, -float('inf'))

        # attn_weight : (B, enc_T)
        if self.smoothing:
            score = torch.sigmoid(score)
            attn_weight = torch.div(score, score.sum(dim=-1).unsqueeze(dim=-1))
        else:
            attn_weight = self.softmax(score)

        # (B, 1, enc_T) * (B, enc_T, enc_D) -> (B, 1, enc_D)
        context = torch.bmm(attn_weight.unsqueeze(dim=1), values)

        return context, attn_weight


class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, max_len, hidden_size, encoder_size,
                 sos_id, eos_id,
                 n_layers=1, rnn_cell='gru',
                 bidirectional_encoder=False, bidirectional_decoder=False,
                 dropout_p=0, use_attention=True):
        super(DecoderRNN, self).__init__()

        self.output_size = vocab_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.bidirectional_encoder = bidirectional_encoder
        self.bidirectional_decoder = bidirectional_decoder
        self.encoder_output_size = encoder_size * 2 if self.bidirectional_encoder else encoder_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_len
        self.use_attention = use_attention
        self.eos_id = eos_id
        self.sos_id = sos_id

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        else:
            raise ValueError("Unsupported RNN Cell: {0}".format(rnn_cell))

        self.init_input = None
        self.rnn = self.rnn_cell(self.hidden_size + self.encoder_output_size, self.hidden_size, self.n_layers,
                                 batch_first=True, dropout=dropout_p, bidirectional=self.bidirectional_decoder)

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.input_dropout = nn.Dropout(self.dropout_p)

        self.attention = Attention(dec_dim=self.hidden_size, enc_dim=self.encoder_output_size, conv_dim=1,
                                   attn_dim=self.hidden_size)
        self.fc = nn.Linear(self.hidden_size + self.encoder_output_size, self.output_size)

    def forward_step(self, input_var, hidden, encoder_outputs, context, attn_w, function):
        batch_size = input_var.size(0)
        dec_len = input_var.size(1)
        enc_len = encoder_outputs.size(1)
        enc_dim = encoder_outputs.size(2)
        embedded = self.embedding(input_var)  # (B, dec_T, voc_D) -> (B, dec_T, dec_D)
        embedded = self.input_dropout(embedded)

        y_all = []
        attn_w_all = []
        for i in range(embedded.size(1)):
            embedded_inputs = embedded[:, i, :]  # (B, dec_D)

            rnn_input = torch.cat([embedded_inputs, context], dim=1)  # (B, dec_D + enc_D)
            rnn_input = rnn_input.unsqueeze(1)
            output, hidden = self.rnn(rnn_input, hidden)  # (B, 1, dec_D)

            context, attn_w = self.attention(output, encoder_outputs, attn_w)  # (B, 1, enc_D), (B, enc_T)
            attn_w_all.append(attn_w)

            context = context.squeeze(1)
            output = output.squeeze(1)  # (B, 1, dec_D) -> (B, dec_D)
            context = self.input_dropout(context)
            output = self.input_dropout(output)
            output = torch.cat((output, context), dim=1)  # (B, dec_D + enc_D)

            pred = function(self.fc(output), dim=-1)
            y_all.append(pred)

        if embedded.size(1) != 1:
            y_all = torch.stack(y_all, dim=1)  # (B, dec_T, out_D)
            attn_w_all = torch.stack(attn_w_all, dim=1)  # (B, dec_T, enc_T)
        else:
            y_all = y_all[0].unsqueeze(1)  # (B, 1, out_D)
            attn_w_all = attn_w_all[0]  # (B, 1, enc_T)

        return y_all, hidden, context, attn_w_all

    def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None,
                function=F.log_softmax, teacher_forcing_ratio=0):
        """
        param:inputs: Decoder inputs sequence, Shape=(B, dec_T)
        param:encoder_hidden: Encoder last hidden states, Default : None
        param:encoder_outputs: Encoder outputs, Shape=(B,enc_T,enc_D)
        """

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if teacher_forcing_ratio != 0:
            inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden, encoder_outputs,
                                                                 function, teacher_forcing_ratio)
        else:
            batch_size = encoder_outputs.size(0)
            inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            max_length = self.max_length

        decoder_hidden = None
        context = encoder_outputs.new_zeros(batch_size, encoder_outputs.size(2))  # (B, D)
        attn_w = encoder_outputs.new_zeros(batch_size, encoder_outputs.size(1))  # (B, T)

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)

        def decode(step, step_output):
            decoder_outputs.append(step_output)
            symbols = decoder_outputs[-1].topk(1)[1]
            sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols

        if use_teacher_forcing:
            decoder_input = inputs[:, :-1]
            decoder_output, decoder_hidden, context, attn_w = self.forward_step(decoder_input,
                                                                                decoder_hidden,
                                                                                encoder_outputs,
                                                                                context,
                                                                                attn_w,
                                                                                function=function)

            for di in range(decoder_output.size(1)):
                step_output = decoder_output[:, di, :]
                decode(di, step_output)
        else:
            decoder_input = inputs[:, 0].unsqueeze(1)
            for di in range(max_length):
                decoder_output, decoder_hidden, context, attn_w = self.forward_step(decoder_input,
                                                                                    decoder_hidden,
                                                                                    encoder_outputs,
                                                                                    context,
                                                                                    attn_w,
                                                                                    function=function)
                step_output = decoder_output.squeeze(1)
                symbols = decode(di, step_output)
                decoder_input = symbols

        return decoder_outputs

    def _validate_args(self, inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio):
        if self.use_attention:
            if encoder_outputs is None:
                raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        batch_size = encoder_outputs.size(0)

        if inputs is None:
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            max_length = self.max_length
        else:
            max_length = inputs.size(1) - 1  # minus the start of sequence symbol

        return inputs, batch_size, max_length


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, decode_function=F.log_softmax):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decode_function = decode_function

    def flatten_parameters(self):
        pass

    def forward(self, input_variable, input_lengths=None, target_variable=None,
                teacher_forcing_ratio=0):

        self.encoder.rnn.flatten_parameters()
        encoder_outputs, encoder_hidden = self.encoder(input_variable, input_lengths)

        self.decoder.rnn.flatten_parameters()
        decoder_output = self.decoder(inputs=target_variable,
                                      encoder_hidden=None,
                                      encoder_outputs=encoder_outputs,
                                      function=self.decode_function,
                                      teacher_forcing_ratio=teacher_forcing_ratio)

        return decoder_output

    @staticmethod
    def get_param_size(model):
        params = 0
        for p in model.parameters():
            tmp = 1
            for x in p.size():
                tmp *= x
            params += tmp
        return params