{
    "dataset_reader": {
        "type": "pico_sents",
        "token_indexers": {
            "elmo": {
                "type": "elmo_characters"
            }
        }
    },
    "train_data_path": std.extVar("TRAIN_DATA_PATH"),
    "validation_data_path": std.extVar("DEV_DATA_PATH"),
    "model": {
        "type": "pico_sentence_attn",
        "heads": 4,
        "gamma": std.extVar("GAMMA"), 
        "dropout": 0,
        "text_field_embedder": {
            "token_embedders": {
                "elmo": {                                                       
                      "type": "elmo_token_embedder",                              
                      "options_file": "/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json",
                      "weight_file": "/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights_PubMed_only.hdf5",
                      "do_layer_norm": false,                                     
                      "dropout": 0.0                                              
                  }                                                               
              }    
        },
        "encoder": {
            "type": "lstm",
            "input_size": 1024, 
            "hidden_size": 200,
            "num_layers": 1,
            "bidirectional": true
        },
        "feedforward": {
            "input_dim": 400,
            "num_layers": 2,
            "hidden_dims": [200, 1],
            "activations": ["relu", "linear"]
        }
    },
    "iterator": {
        "type": "basic",
        "batch_size": std.parseInt(std.extVar("BATCH_SIZE"))
    },
    "trainer": {
        "optimizer": {
            "type": "adam",
            // will be reset with --overrides flag
            "lr": 0.01,
            // will be reset with --overrides flag
            "weight_decay": 1e-7
        },
    "validation_metric": "+avg_f1",
    "num_serialized_models_to_keep": 1,
    "num_epochs": std.parseInt(std.extVar("EPOCHS")),
    "patience": 5,
    "cuda_device": 0 
  }
}
