import os
import pickle
import torch
import math
import numpy as np
import matplotlib.pyplot as plt

from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch.nn.init as init
import torch.nn.functional as F

from edit_distance import SequenceMatcher
from neural_decoder.neural_decoder_trainer import loadModel as gruLoader
from neural_decoder.transformer_decoder_trainer import loadModel as transformerLoader
from neural_decoder.dataset import SpeechDataset
from neural_decoder.dataset import CompleteSpeechDataset

def complete_padding(batch):
    X, y, X_lens, y_lens, days, transcripts = zip(*batch)
    X_padded = pad_sequence(X, batch_first=True, padding_value=0)
    y_padded = pad_sequence(y, batch_first=True, padding_value=0)

    return (
        X_padded,
        y_padded,
        torch.stack(X_lens),
        torch.stack(y_lens),
        torch.stack(days),
        transcripts
    )

def padding(batch):
    X, y, X_lens, y_lens, days = zip(*batch)
    X_padded = pad_sequence(X, batch_first=True, padding_value=0)
    y_padded = pad_sequence(y, batch_first=True, padding_value=0)

    return (
        X_padded,
        y_padded,
        torch.stack(X_lens),
        torch.stack(y_lens),
        torch.stack(days)
    )

def plot_loss(paths):
    data = list()
    for path in paths:
        with open(path + "/trainingStats", "rb") as handle:
            data.append(pickle.load(handle))

    plt.title("Loss", fontsize=18)
    for i, path in enumerate(paths):
        plt.plot(data[i]['testLoss'], label=path)
    plt.legend()
    plt.show()
    plt.close()
    
    plt.title("CER", fontsize=18)
    for i, path in enumerate(paths):
        plt.plot(data[i]['testCER'], label=path)
    plt.legend()
    plt.show()

    return

def inference(model_path, model_type, data_path, ds, args):
    output = {}
    output['logits'] = []
    output['logitLengths'] = []
    output['decodedSeqs'] = []
    output['editDistances'] = []
    output['trueSeqLengths'] = []
    output['trueSeqs'] = []
    output['transcriptions'] = []
    output['seqErrorRate'] = []
    model = None
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    if model_type == "gru":
        model = gruLoader(model_path, args["n_days"], device)
    elif model_type == "transformer":
        model = transformerLoader(model_path, args["n_days"], device)

    with open(data_path, "rb") as handle:
        loaded_data = pickle.load(handle)

    dataset = CompleteSpeechDataset(loaded_data[ds])
    dataset_loader = DataLoader(
        dataset,
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=complete_padding,
    )

    with torch.no_grad():
        model.eval()
        allLoss = []
        total_edit_distance = 0
        total_seq_length = 0
        for X, y, X_len, y_len, testDayIdx, transcripts in dataset_loader:
            X, y, X_len, y_len, testDayIdx = (
                X.to(device),
                y.to(device),
                X_len.to(device),
                y_len.to(device),
                testDayIdx.to(device),
            )
        
            if model_type == "gru":
                pred = model.forward(X, testDayIdx)
                adjustedLens = ((X_len - model.kernelLen) / model.strideLen).to(torch.int32)
            elif model_type == "transformer":
                pred = model.forward(X, testDayIdx, X_len)
                adjustedLens = ((X_len - model.kernel_size) / model.stride_len).to(torch.int32)

            for iterIdx in range(pred.shape[0]):
                decodedSeq = torch.argmax(
                    torch.tensor(pred[iterIdx, 0 : adjustedLens[iterIdx], :]),
                    dim=-1,
                )  # [num_seq,]
                decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
                decodedSeq = decodedSeq.cpu().detach().numpy()
                decodedSeq = np.array([i for i in decodedSeq if i != 0])

                trueSeq = np.array(
                    y[iterIdx][0 : y_len[iterIdx]].cpu().detach()
                )

                matcher = SequenceMatcher(
                    a=trueSeq.tolist(), b=decodedSeq.tolist()
                )

                output['logits'].append(pred[iterIdx, :, :])
                output['logitLengths'].append(X_len[iterIdx].item())
                output['decodedSeqs'].append(decodedSeq)
                output['editDistances'].append(matcher.distance())
                output['trueSeqLengths'].append(y_len[iterIdx].item())
                output['trueSeqs'].append(y[iterIdx])
                output['transcriptions'].append(transcripts[iterIdx])

    max_logits = 0
    max_decode = 0
    for i in range(len(output['logits'])):
        max_logits = max(max_logits, output['logits'][i].shape[0])
        max_decode = max(max_decode, output['decodedSeqs'][i].shape[0])

    logits = torch.zeros((len(output['logits']), max_logits, output['logits'][0].shape[1]))
    decode = torch.zeros((len(output['logits']), output['trueSeqs'][0].shape[0]))
    true = torch.zeros((len(output['logits']), output['trueSeqs'][0].shape[0]))
    transcriptions = torch.zeros((len(output['logits']), output['trueSeqs'][0].shape[0]))

    for i in range(len(output['logits'])):
        logits[i,:output['logits'][i].shape[0],:] = output['logits'][i]
        decode[i,:len(output['decodedSeqs'][i])] = torch.from_numpy(output['decodedSeqs'][i])
        true[i,:] = output['trueSeqs'][i]
        transcriptions[i,:len(output['transcriptions'][i])] = torch.tensor([ord(c) for c in output['transcriptions'][i]], dtype=int)

    output['logitLengths'] = logits
    output['decodedSeqs'] = decode
    output['trueSeqs'] = true
    output['transcriptions'] = transcriptions

    with open("/".join(data_path.split("/")[:-1]) + "/" + model_path.split("/")[-1] + "_" + ds, "wb") as file:
        pickle.dump(output, file)

    return output

def phoneme_eval(model_path, model_type, data_path, ds, args):
    model = None
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    if model_type == "gru":
        model = gruLoader(model_path, args["n_days"], device)
    elif model_type == "transformer":
        model = transformerLoader(model_path, args["n_days"], device)

    with open(data_path, "rb") as handle:
        loaded_data = pickle.load(handle)

    dataset = SpeechDataset(loaded_data[ds])
    dataset_loader = DataLoader(
        dataset,
        batch_size=args["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=padding,
    )
    
    loss_ctc = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    print("Evalutating...")
    with torch.no_grad():
        model.eval()
        allLoss = []
        total_edit_distance = 0
        total_seq_length = 0
        for X, y, X_len, y_len, testDayIdx in dataset_loader:
            X, y, X_len, y_len, testDayIdx = (
                X.to(device),
                y.to(device),
                X_len.to(device),
                y_len.to(device),
                testDayIdx.to(device),
            )

            if model_type == "gru":
                pred = model.forward(X, testDayIdx)
                adjustedLens = ((X_len - model.kernelLen) / model.strideLen).to(torch.int32)
            elif model_type == "transformer":
                pred = model.forward(X, testDayIdx, X_len)
                adjustedLens = ((X_len - model.kernel_size) / model.stride_len).to(torch.int32)

            loss = loss_ctc(torch.permute(pred.log_softmax(2), [1, 0, 2]), y, adjustedLens, y_len)
            loss = torch.sum(loss)
            allLoss.append(loss.cpu().detach().numpy())

            for iterIdx in range(pred.shape[0]):
                decodedSeq = torch.argmax(
                    torch.tensor(pred[iterIdx, 0 : adjustedLens[iterIdx], :]),
                    dim=-1,
                )  # [num_seq,]
                decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
                decodedSeq = decodedSeq.cpu().detach().numpy()
                decodedSeq = np.array([i for i in decodedSeq if i != 0])

                trueSeq = np.array(
                    y[iterIdx][0 : y_len[iterIdx]].cpu().detach()
                )

                matcher = SequenceMatcher(
                    a=trueSeq.tolist(), b=decodedSeq.tolist()
                )
                total_edit_distance += matcher.distance()
                total_seq_length += len(trueSeq)

        avgDayLoss = np.sum(allLoss) / len(dataset_loader)
        cer = total_edit_distance / total_seq_length

    print("Average CTC Loss:", avgDayLoss)
    print("Phoneme Error Rate:", cer)
    return

# Note: cannot test on "competition" partition
def main():
    phoneme_eval("saved_models/pt_gru_baseline", "gru", "src/neural_decoder/ptDecoder_ctc", "train", {"n_days": 24, "batch_size": 64})
    phoneme_eval("saved_models/pt_transformer_baseline", "transformer", "src/neural_decoder/ptDecoder_ctc", "train", {"n_days": 24, "batch_size": 64})
    # plot_loss(["saved_models/pt_gru_baseline", "saved_models/pt_transformer_baseline", "saved_models/pt_transformer_baseline_unfold"])
    # with open("src/neural_decoder/ptDecoder_ctc", "rb") as handle:
    #     loaded_data = pickle.load(handle)

    # dataset = loaded_data["train"]
    # print(sum([len(x.split()) for x in dataset[0]['transcriptions']]) / (sum(dataset[0]['timeSeriesLens']) * 20 / 1000) * 60)
    # print((dataset[0]['timeSeriesLens'] * 20 / 1000))
    # inference("saved_models/pt_gru_baseline", "gru", "src/neural_decoder/ptDecoder_ctc", "test", {"n_days": 24, "batch_size": 64})
    return

if __name__ == "__main__":
    main()