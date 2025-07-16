# Elements to put in readme
- config files are fetched from configs -> can be changed in the code, but recommend modifying the config files directly
- run main.py from the root of the folder, which is `FLAIR-1`
- if you want to change the base model, put the weights in the `model` folder and update the model path in `pipeline_config;yaml` (the quantization methods may not work very well if you change though)

# TODO
## Cahier des charges quantization pipeline

- non bloquant (si une méthode marche pas, on passe à la suivante, keep error log)
- métriques : csv with method, precision, total latency, (number of patches?), memory used, carbon,success flag
- des logs simples (saved with output id = datetime up to second, 1 log / method)
	-> `method_date_time.log`
	- at the very beginning : GPU name, cuda, torch, onnx versions
	- tqdm sur les images traitées du dataset, pas les patches
	- méthode utilisée (de quantisation, quel inference engine, quel device)
	- error if something fails, then "skipping {method name}"
	- latency and memory used at the end
- choix d'intégrer le traitement cpu
- choix des méthodes : enabled, disabled, some parameters
- option for debugging : dry-run to log methods to apply and directory for saving 

### optionnel / future works:
- métriques de performance (miou....) si dataset + labels fournis (le csv comme il faut)
- save models that were tried and load them if available

## Tâches
- general structure of the run (= test all enabled methods)
- config file for switching on/off methods (+ parameters possibly) (yaml)
	pytorch_baseline :
		enable : true
		precision : fp32
	onnx_baseline :
		enable : true
		precision : fp32
	pruned : 
		enable : true
		sparsity : 0.05
		precision : fp32
	pruna_half :
	pruna_half_pruned :
	pytorch_bf16 : 
	ao_weights-only_int8 :
- prepare_model for every type of model (possibly adapt inference accordingly)
- save yaml config snapshot for reproducibility
- clean logger
## Methods to be tested (everything is done on the fly)
- baseline : pytorch model from checkpoints, Swin transformer
- onnx : baseline exported, optimized and simplified
- pruning : baseline pruned at 0.05 (pruna)
- quant + pruning : quantized at fp16, pruned at 0.05 (pruna)
- half : quantized at fp16 (pruna)
- int8 : torch dynamic from pruna
- bf16 : truncate weights (directly with pytorch ?)
- int8 : weight only PTDQ (torchao)
