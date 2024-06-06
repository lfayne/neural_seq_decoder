modelName = 'test'

args = {}
args['outputDir'] = '/Users/Woody/Stanford/CS 224N/Project/neural_seq_decoder/saved_models/' + modelName
args['datasetPath'] = '/Users/Woody/Stanford/CS 224N/Project/neural_seq_decoder/src/neural_decoder/ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 64
#args['lrStart'] = 1e-6
#args['lrEnd'] = 1e-4
args['lrStart'] = 0.02
args['lrEnd'] = 0.02

args['nBatch'] = 10000 #3000
args['nLayers'] = 5
args['seed'] = 0
args['n_classes'] = 40
args['neural_dim'] = 256
args['hidden_dim'] = 1024
args['convolve'] = False
args['kernel_size'] = 32
args['stride'] = 4
args['context'] = [50 for _ in range(3)]
args['n_heads'] = 8
args['dropout'] = 0.3
args['whiteNoiseSD'] = 0.8
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = True
args['l2_decay'] = 1e-5

from neural_decoder.eegnet_decoder_trainer import trainModel

trainModel(args)