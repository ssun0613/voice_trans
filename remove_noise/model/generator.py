import sys, os
sys.path.append("..")
import torch
import torch.nn as nn


class generator(nn.Module):
    def __init__(self, opt):
        super(generator,self).__init__()
        if not opt.debugging:
            from ssun.Voice_trans.model.encoder import Er, Ec, InterpLnr
            from ssun.Voice_trans.model.decoder_s import Decoder_s as Ds
            from ssun.Voice_trans.model.pitch_predictor import pitch_predictor as P
            from ssun.Voice_trans.functions.etc_fcn import quantize_f0_torch
        else:
            from Voice_trans.model.encoder import Er, Ec, InterpLnr
            from Voice_trans.model.decoder_s import Decoder_s as Ds
            from Voice_trans.model.pitch_predictor import pitch_predictor as P
            from Voice_trans.functions.etc_fcn import quantize_f0_torch

        self.Er = Er()
        self.Ec = Ec()
        self.Ds = Ds()
        self.P = P()
        self.InterpLnr = InterpLnr()
        self.quantize_f0_torch = quantize_f0_torch

    def forward(self, mel_in, len_org, sp_id):
        mel_inter = self.InterpLnr(mel_in, len_org)

        rhythm = self.Er(mel_in.transpose(2,1))
        content = self.Ec(mel_inter.transpose(2,1))

        rhythm_repeat = rhythm.repeat_interleave(8, dim=1)
        content_repeat = content.repeat_interleave(8, dim=1)

        r_c_s = torch.cat((rhythm_repeat, content_repeat, sp_id.unsqueeze(1).expand(-1, mel_in.transpose(2,1).size(-1), -1)), dim=-1)
        pitch_p = self.P(r_c_s).repeat_interleave(8, dim=1)
        pitch_p_repeat_quan = self.quantize_f0_torch(pitch_p)[0]

        r_c_p = torch.cat((rhythm_repeat, content_repeat, pitch_p), dim=-1)
        mel_output = self.Ds(r_c_p)

        #------------------------------------ for calculate reconstruction loss ------------------------------------

        mel_out_inter = self.InterpLnr(mel_output, len_org)

        rhythm_r = self.Er(mel_output.transpose(2, 1))
        content_r = self.Ec(mel_out_inter.transpose(2, 1))

        rhythm_r_repeat = rhythm_r.repeat_interleave(8, dim=1)
        content_r_repeat = content_r.repeat_interleave(8, dim=1)

        r_c_p_r = torch.cat((rhythm_r_repeat, content_r_repeat, sp_id.unsqueeze(1).expand(-1, mel_in.transpose(2,1).size(-1), -1)), dim=-1)
        pitch_p_r = self.P(r_c_p_r).repeat_interleave(8, dim=1)
        pitch_p_r_quan = self.quantize_f0_torch(pitch_p_r)[0]

        return mel_output, pitch_p_repeat_quan, rhythm, content, rhythm_r, content_r, pitch_p_r_quan





