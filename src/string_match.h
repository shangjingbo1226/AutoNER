#ifndef __STRING_MATCH_H__
#define __STRING_MATCH_H__

#include "utils.h"
#include "annotation.h"

namespace StringMatching
{

using namespace Utils;
using namespace Annotation;

// start the dp process

struct Traceback
{
    int i;
    set<string> types;
};

inline vector<Token> segmentDP(const vector<string>& tokens, double weight = 1.0)
{
    int n = tokens.size();
    vector<double> f(n + 1, -1);
    f[0] = 0;
    vector<Traceback> traceback(n + 1);
    for (int i = 0; i < n; ++ i) {
        if (f[i] > f[i + 1]) {
            f[i + 1] = f[i];
            traceback[i + 1].i = i;
            traceback[i + 1].types = {};
        }
        int u = 0, delta = 0;
        for (int j = i; j < n; ++ j) {
            if (tokens[j] == "\n") {
                break;
            }
            ++ delta;
            const string& token = tokens[j];
            u = TrieForKB::getChild(u, token);
            if (u == -1) {
                break;
            }
            if (TrieForKB::isEntity(u)) { // matches some entity in the core dict
                if (f[j + 1] < f[i] + delta * delta) {
                    f[j + 1] = f[i] + delta * delta;
                    traceback[j + 1].i = i;
                    traceback[j + 1].types = TrieForKB::getTypes(u);
                }
            } else if (TrieForKB::isFiltered(u)) { // matches some entity in the full dict
                if (f[j + 1] < f[i] + weight * delta * delta) {
                    f[j + 1] = f[i] + weight * delta * delta;
                    traceback[j + 1].i = i;
                    traceback[j + 1].types = TrieForKB::getTypes(u);
                }
            }
        }
    }
    vector<Token> ret;
    int j = n;
    while (j > 0) {
        int i = traceback[j].i;
        set<string> types = traceback[j].types;
        
        // [i, j)
        Token token;
        token.l = i;
        token.r = j;

        token.type = "";
        for (const string& type : types) {
            if (token.type != "") {
                token.type += ",";
            }
            token.type += type;
        }

        token.token = "";
        for (int k = i; k < j; ++ k) {
            token.token += tokens[k];
        }

        ret.push_back(token);
        j = i;
    }
    reverse(ret.begin(), ret.end());

    return ret;
}

AnnotatedData getDistantSupervision(const vector<string>& tokens)
{
    vector<Token> matchedTokens = StringMatching::segmentDP(tokens);

    AnnotatedData ret;
    ret.rawTokens = tokens;
    ret.annotatedTokens = matchedTokens;
    return ret;
}


} // end namespace StringMatch

#endif