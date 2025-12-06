import os
from architecture import *
from utils import *
from training import *
import architecture, utils, training
import importlib
importlib.reload(architecture)
importlib.reload(utils)
importlib.reload(training)

if os.path.exists('__pycache__'):
    shutil.rmtree('__pycache__')

params = get_params_settings(yaml_params_path=params_path)
start_logging(params)
device = prepare_system()
params['DEVICE'] = device

if params['TARGET_FEATURE_NAME'] == 'refractive_index':
    params['CONDITION_NAMES'] = ['temperature_K', 'wavelength_nm']

if params['TARGET_FEATURE_NAME'] == 'conductivity':
    params['CONDITION_NAMES'] = ['temperature_K']

if params['USE_SAVED_GRAPH_LIST']:
    data_list = GraphDataIO.load(os.path.join(DATA,f"{params['TARGET_FEATURE_NAME']}_train_graphs.pkl.gz"))

    loader = GraphDataIO.load_to_dataloader(os.path.join(DATA,f"{params['TARGET_FEATURE_NAME']}_train_graphs.pkl.gz"), batch_size=params['NUM_GRAPHS_PER_BATCH'], shuffle=True)
    val_loader = GraphDataIO.load_to_dataloader(os.path.join(DATA, f"{params['TARGET_FEATURE_NAME']}_valid_graphs.pkl.gz"), batch_size=params['NUM_GRAPHS_PER_BATCH'], shuffle=True)
    test_loader = GraphDataIO.load_to_dataloader(os.path.join(DATA, f"{params['TARGET_FEATURE_NAME']}_test_graphs.pkl.gz"), batch_size=params['NUM_GRAPHS_PER_BATCH'], shuffle=True)
    target_transformer = TargetTransformer(method=params['FEATURE_TRANSFORM_METHOD'], filename=os.path.join(scaler_paths, f"{params['TARGET_FEATURE_NAME']}_{params['FEATURE_TRANSFORM_METHOD']}.pkl"))
    target_transformer.load()
    glob_transformer = GlobTransformer(method=params['SCALE_CONDITIONS_TYPE'], filename=os.path.join(scaler_paths, f"{params['TARGET_FEATURE_NAME']}_global_features_transformer.pkl"))
    glob_transformer.load()
else:
    loader, val_loader, test_loader, data_list, target_transformer, feature_transformer, glob_transformer = set_training(params, device,scaler_paths)

if params['ACTION'] == 'train':
    '''To Train'''
    params['TRAIN_MODE'] = True
    params['SEARCH_MODE'] = False
    for pooling in ['mean', 'attention']: # ['mean_max','mean', 'max', 'add', 'attention']
        params['POOLING_METHOD'] = pooling
        params['ARCH'] = 'NNConv'
        logging.info('\n')
        logging.info(f"Architecture: {params['ARCH']}, POOLING_METHOD: {params['POOLING_METHOD']}")
        model = run_model(params, data_list,loader, val_loader, test_loader, target_transformer)
        torch.save(model.state_dict(), os.path.join(MODELS, f"{params['ARCH']}-{params['POOLING_METHOD']}-{params['TARGET_FEATURE_NAME']}-model_weights.pth"))

elif params['ACTION'] == 'search_and_train':
    '''To Serach for best parameters and ARCH'''
    params['TRAIN_MODE'] = False
    params['SEARCH_MODE'] = True
    study = run_search(objective, direction="minimize", n_trials=5, show_progress_bar=True)
    show_optimization(study)
    best_params = study.best_trial.params
    params['ARCH'] = best_params["ARCH"]
    params['LR'] = best_params["LR"]
    params['EMBEDDING_SIZE'] = sorted([best_params[i] for i in list(best_params.keys()) if i.strip().startswith('embedding_size')], reverse=False)
    params['LINEAR_SIZE'] = sorted([best_params[i] for i in list(best_params.keys()) if i.strip().startswith('linear_size')], reverse=True)
    
    params['TRAIN_MODE'] = True
    params['SEARCH_MODE'] = False
    model = run_model(params, data_list,loader, val_loader, test_loader, target_transformer)
    torch.save(model.state_dict(), os.path.join(MODELS, f"{params['ARCH']}-{params['POOLING_METHOD']}-{params['TARGET_FEATURE_NAME']}-model_weights.pth"))

if params['TRANSFER']:
    if params['USE_SAVED_GRAPH_LIST']:
        data_list = GraphDataIO.load(os.path.join(DATA,f"des_{params['TARGET_FEATURE_NAME']}_train_graphs.pkl.gz"))
        loader_des = GraphDataIO.load_to_dataloader(os.path.join(DATA,f"des_{params['TARGET_FEATURE_NAME']}_train_graphs.pkl.gz"), batch_size=params['NUM_GRAPHS_PER_BATCH'], shuffle=True)
        val_loader_des = GraphDataIO.load_to_dataloader(os.path.join(DATA, f"des_{params['TARGET_FEATURE_NAME']}_valid_graphs.pkl.gz"), batch_size=params['NUM_GRAPHS_PER_BATCH'], shuffle=True)
        test_loader_des = GraphDataIO.load_to_dataloader(os.path.join(DATA, f"des_{params['TARGET_FEATURE_NAME']}_test_graphs.pkl.gz"), batch_size=params['NUM_GRAPHS_PER_BATCH'], shuffle=True)
        target_transformer = TargetTransformer(method=params['FEATURE_TRANSFORM_METHOD'], filename=os.path.join(scaler_paths, f"des_{params['TARGET_FEATURE_NAME']}_{params['FEATURE_TRANSFORM_METHOD']}.pkl"))
        target_transformer.load()
        glob_transformer = GlobTransformer(method=params['SCALE_CONDITIONS_TYPE'], filename=os.path.join(scaler_paths, f"des_{params['TARGET_FEATURE_NAME']}_global_features_transformer.pkl"))
        glob_transformer.load()
    else:
            loader_des, val_loader_des, test_loader_des, data_list, target_transformer, feature_transformer, glob_transformer = set_training_des(params, device,scaler_paths)
    
    for pooling in ['mean', 'attention']: # ['mean_max','mean', 'max', 'add', 'attention']
        params['POOLING_METHOD'] = pooling
        encoder_model = instantiate_model(params, data_list)
        ckpt_path = os.path.join(MODELS, f"{params['ARCH']}-{params['POOLING_METHOD']}-{params['TARGET_FEATURE_NAME']}-model_weights.pth")
        if os.path.exists(ckpt_path):
            logging.info(f"Loading checkpoint: {ckpt_path}")
            load_checkpoint_into_encoder(encoder_model, ckpt_path, strict=False)

        transfer_model = TransferGNN(encoder_model,
                                    head_hidden=params['TRANSFER_LINEAR_SIZE'], 
                                    add_params_num=params['COND_DIM'],
                                    freeze_encoder=params['FREEZE']) 

        # optionally unfreeze last N conv layers
        if params['FREEZE'] and params['N_LAYERS_FINETUNE'] > 0:
            unfreeze_last_n_layers(transfer_model.encoder, params['N_LAYERS_FINETUNE'])
            logging.info(f"Unfroze last {params['N_LAYERS_FINETUNE']} Convolution Layers.")

        # If extract embeddings mode
        if params['MODE'] == 'extract':
            # user must supply a dataloader object 
            train_loader_des = GraphDataIO.load_to_dataloader(os.path.join(DATA,"des_train_graphs.pkl.gz"), batch_size=params['NUM_GRAPHS_PER_BATCH'], shuffle=True)
            embs, ids = extract_embeddings(transfer_model.encoder, train_loader_des, device=params['DEVICE'], adapter=adapter)
            np.save(os.path.join(DATA, 'des_embeddings.npy'), embs)
            logging.info(f"Saved embeddings to: {os.path.join(DATA, 'des_embeddings.npy')}")

        # Otherwise train: you must provide train and val loaders as pickled objects for convenience
        elif params['MODE'] == 'train':
            train_loader_des = GraphDataIO.load_to_dataloader(os.path.join(DATA, f"des_{params['TARGET_FEATURE_NAME']}_train_graphs.pkl.gz"), batch_size=params['NUM_GRAPHS_PER_BATCH'], shuffle=True)
            val_loader_des = GraphDataIO.load_to_dataloader(os.path.join(DATA,f"des_{params['TARGET_FEATURE_NAME']}_valid_graphs.pkl.gz"), batch_size=params['NUM_GRAPHS_PER_BATCH'], shuffle=True)
            test_loader_des = GraphDataIO.load_to_dataloader(os.path.join(DATA,f"des_{params['TARGET_FEATURE_NAME']}_test_graphs.pkl.gz"), batch_size=params['NUM_GRAPHS_PER_BATCH'], shuffle=True)

            logging.info('\n')
            logging.info(f"Architecture: {params['ARCH']}, POOLING_METHOD: {params['POOLING_METHOD']}")

            #transfer_model = train_transfer(
            #    transfer_model,
            #    train_loader_des,
            #    val_loader_des,
            #    device=params['DEVICE'],
            #    epochs=params['EPOCH'],
            #    lr_head=params['LR_HEAD'],
            #    lr_encoder=params['LR_ENCODER'],
            #    finetune=params['FINE_TUNE'],
            #    grad_clip=1.0,
            #    save_path=os.path.join(MODELS, f"Transfer-{params['ARCH']}-{params['POOLING_METHOD']}-{params['TARGET_FEATURE_NAME']}-model_weights.pth")
            #)

            model = run_transfer(transfer_model, params, train_loader_des, val_loader_des, test_loader_des,
                                   EPOCHS=params['EPOCH'], device=params['DEVICE'], transformer=target_transformer, lr_head=params['LR_HEAD'],
                                    lr_encoder=params['LR_ENCODER'], finetune=params['FINE_TUNE'])
            torch.save(model.state_dict(), os.path.join(MODELS, f"transfer_{params['ARCH']}-{params['POOLING_METHOD']}-{params['TARGET_FEATURE_NAME']}-model_weights.pth"))

        elif params['MODE'] == 'none':
            logging.info(f'\n{transfer_model}')
            print(f'\n{transfer_model}')

logging.info('Settings used for this run')
for key, val in params.items():
    logging.info(f"{key} ------------> {val}")
logging.info('\n')
