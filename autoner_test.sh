MODEL_NAME="BC5CDR"
GPU_ID=0
RAW_TEXT="data/BC5CDR/raw_text.txt"

green=`tput setaf 2`
reset=`tput sgr0`

MODEL_ROOT=./models/$MODEL_NAME
CHECKPOINT=$MODEL_ROOT/checkpoint/autoner/

python preprocess_partial_ner/encode_test.py --input_data $RAW_TEXT --checkpoint_folder $CHECKPOINT --output_file $MODEL_ROOT/encoded_test.pk

python test_partial_ner.py --input_corpus $MODEL_ROOT/encoded_test.pk --checkpoint_folder $CHECKPOINT --output_text $MODEL_ROOT/decoded.txt --hid_dim 300 --droprate 0.5 --word_dim 200 