#include "utils.h"
#include "annotation.h"
#include "string_match.h"

using namespace Utils;
using Annotation::Token;
using Annotation::AnnotatedData;

const string STOPWORDS_FILENAME = "data/stopwords.txt";

int main(int argc, char* argv[])
{
    if (argc != 5) {
        fprintf(stderr, "[usage] <tokenized_raw_txt> <core KB file> <full KB file> <output ck filename>\n");
        return -1;
    }

    string RAW_FILENAME = argv[1];
    string CORE_FILENAME = argv[2];
    string FULL_FILENAME = argv[3];
    string OUTPUT_FILENAME = argv[4];

    FILE* in = tryOpen(RAW_FILENAME);
    vector<string> rawTokens;
    while (getLine(in)) {
        if (strlen(line) == 0) {
            rawTokens.push_back("\n");
        } else {
            rawTokens.push_back(line);
        }
    }
    fclose(in);

    Annotation::initialize(CORE_FILENAME, FULL_FILENAME, STOPWORDS_FILENAME);
    
    AnnotatedData stringMatchingLabels = StringMatching::getDistantSupervision(rawTokens);
    FILE* out = tryOpen(OUTPUT_FILENAME, "w");
    fprintf(out, "%s\n", stringMatchingLabels.toCk().c_str());
    fclose(out);

    return 0;
}
