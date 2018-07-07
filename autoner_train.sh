MODEL_NAME="BC5CDR"
GPU_ID=0
RAW_TEXT="data/BC5CDR/raw_text.txt"
DICT_CORE="data/BC5CDR/dict_core.txt"
DICT_FULL="data/BC5CDR/dict_full.txt"
EMBEDDING_TXT_FILE="embedding/bio_embedding.txt"
FIRST_RUN=1

green=`tput setaf 2`
reset=`tput sgr0`

DEV_SET="data/BC5CDR/truth_dev.ck"
TEST_SET="data/BC5CDR/truth_test.ck"

MODEL_ROOT=./models/$MODEL_NAME
TRAINING_SET=$MODEL_ROOT/annotations.ck

mkdir -p $MODEL_ROOT

echo ${green}=== Compilation ===${reset}
make

echo ${green}=== Generating Distant Supervision ===${reset}
bin/generate $RAW_TEXT $DICT_CORE $DICT_FULL $TRAINING_SET

if [ $FIRST_RUN == 1 ] && [ ! -e $MODEL_ROOT/embedding.pk ]; then
    echo ${green}=== Encoding Embeddings ===${reset}
    python preprocess_partial_ner/save_emb.py --input_embedding $EMBEDDING_TXT_FILE --output_embedding $MODEL_ROOT/embedding.pk
fi

if [ DEV_SET == "" ]; then
    DEV_SET=$TRAINING_SET
fi

if [ TEST_SET == "" ]; then
    TEST_SET=$TRAINING_SET
fi

mkdir -p $MODEL_ROOT/encoded_data

if [ $FIRST_RUN == 1 ] && [ ! -e $MODEL_ROOT/encoded_data/test.pk ]; then
    echo ${green}=== Encoding Dataset ===${reset}
    python preprocess_partial_ner/encode_folder.py --input_train $TRAINING_SET --input_testa $DEV_SET --input_testb $TEST_SET --pre_word_emb $MODEL_ROOT/embedding.pk --output_folder $MODEL_ROOT/encoded_data/
fi

LOG_DIR=$MODEL_ROOT/tensorboard_logs
CHECKPOINT=$MODEL_ROOT/autoner.model

# clear the previous logs if there are any
rm -rf $LOG_DIR

echo ${green}=== Training AutoNER Model ===${reset}
python train_partial_ner.py --dataset_folder $MODEL_ROOT/encoded_data/ --gpu $GPU_ID --update SGD --lr 0.05 --log_dir $LOG_DIR --hid_dim 300 --droprate 0.5 --sample_ratio 1.0 --word_dim 200 --checkpoint $CHECKPOINT --epoch 20

echo ${green}Done.${reset}
