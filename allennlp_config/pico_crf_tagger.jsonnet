local maybepath = std.extVar("PRETRAINED_MODEL_PATH");
local pretrained_path = if maybepath == "" then null else maybepath;

{
    "dataset_reader": {
        "type": "ebm_nlp",
        "element": "all",
        "single_sent": true,
        "token_indexers": {
            "elmo": {
                "type": "elmo_characters"
            }
        }
    },
    "train_data_path": std.extVar("TRAIN_DATA_PATH"),
    "validation_data_path": std.extVar("DEV_DATA_PATH"),
    // "test_data_path": "/io_tags/test",
    // "evaluate_on_test": true,
    "model": {
        "type": "pretrained_crf_tagger",
        "label_encoding": "IO",
        "pretrained_file": pretrained_path, 
        "constrain_crf_decoding": false,
        "calculate_span_f1": false,
        "dropout": 0.5, 
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
            "dropout": 0,
            "bidirectional": true
        }
    },
    "iterator": {
        "type": "basic",
        "batch_size": std.parseInt(std.extVar("BATCH_SIZE")) 
    },
    //"vocabulary": {
    //    "directory_path": "/io_vocab/"
    //},
    "trainer": {
        "optimizer": {
            "type": "adam",
            // will be reset with --overrides flag
            "lr": 0.001,
            "weight_decay": 1e-7
        },
        "validation_metric": "+avg_f1",
        "num_serialized_models_to_keep": 1,
        "num_epochs": std.parseInt(std.extVar("EPOCHS")), 
        "patience": 5, 
        "cuda_device": 0,
//        "learning_rate_scheduler": {
//            "type": "exponential",
//            "gamma": 0.9
//        }
  }
}
