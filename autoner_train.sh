MODEL_NAME="BC5CDR"

RAW_TEXT="data/BC5CDR/raw_text.txt"
DICT_CORE="data/BC5CDR/dict_core.txt"
DICT_FULL="data/BC5CDR/dict_full.txt"

mkdir -p models/$MODEL_NAME

bin/generate $RAW_TEXT $DICT_CORE $DICT_FULL models/$MODEL_NAME/annotations.ck
