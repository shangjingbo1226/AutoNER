#ifndef __ANNOTATION_H__
#define __ANNOTATION_H__

#include "utils.h"

namespace Annotation
{

const string FILTERED_TYPE = "__FILTERED__";
const int HALF_WINDOW_SIZE = 7;

using namespace Utils;

unordered_set<string> stopwordSet;

    namespace TrieForKB {
        struct Node {
            unordered_map<string, int> children;
            set<string> types;
        };

        vector<Node> trie;

        inline set<string> getTypes(int u) {
            assert(0 <= u && u < trie.size());
            return trie[u].types;
        }

        inline int getChild(int u, const string& token) {
            if (u < 0 || u >= trie.size()) {
                return -1;
            }
            if (!trie[u].children.count(token)) {
                return -1;
            }
            return trie[u].children[token];
        }

        inline bool isEntity(int u) {
            return u >= 0 && u < trie.size() && trie[u].types.size() > 0 && !trie[u].types.count(FILTERED_TYPE);
        }

        inline bool isFiltered(int u) {
            return u >= 0 && u < trie.size() && trie[u].types.size() == 1 && trie[u].types.count(FILTERED_TYPE);
        }

        inline void initialize() {
            trie.clear();
            trie.push_back(Node());
        }

        inline void markAsFiltered(const vector<string>& tokens, bool noLowercases, bool mustExactlySame = false) {
            // add the raw form
            if (true) {
                int u = 0;
                for (const string& ch : tokens) {
                    if (!trie[u].children.count(ch)) {
                        trie[u].children[ch] = trie.size();
                        trie.push_back(Node());
                    }
                    u = trie[u].children[ch];
                }
                if (trie[u].types.size() == 0) {
                    trie[u].types.insert(FILTERED_TYPE);
                }
            }
            // add the all upper form
            if (!mustExactlySame) {
                int u = 0;
                for (const string& token : tokens) {
                    const string& ch = toUpper(token);
                    if (!trie[u].children.count(ch)) {
                        trie[u].children[ch] = trie.size();
                        trie.push_back(Node());
                    }
                    u = trie[u].children[ch];
                }
                if (trie[u].types.size() == 0) {
                    trie[u].types.insert(FILTERED_TYPE);
                }
            }
            // add the all lower form
            if (!noLowercases) {
                int u = 0;
                for (const string& token : tokens) {
                    const string& ch = toLower(token);
                    if (!trie[u].children.count(ch)) {
                        trie[u].children[ch] = trie.size();
                        trie.push_back(Node());
                    }
                    u = trie[u].children[ch];
                }
                if (trie[u].types.size() == 0) {
                    trie[u].types.insert(FILTERED_TYPE);
                }
            }
        }

        inline void insert(const vector<string>& tokens, const vector<string>& types, bool noLowercases, bool mustExactlySame = false) {
            // add the raw form
            if (true) {
                int u = 0;
                for (const string& ch : tokens) {
                    if (!trie[u].children.count(ch)) {
                        trie[u].children[ch] = trie.size();
                        trie.push_back(Node());
                    }
                    u = trie[u].children[ch];
                }
                trie[u].types.insert(types.begin(), types.end());
            }
            // add the all upper form
            if (!mustExactlySame) {
                int u = 0;
                for (const string& token : tokens) {
                    const string& ch = toUpper(token);
                    if (!trie[u].children.count(ch)) {
                        trie[u].children[ch] = trie.size();
                        trie.push_back(Node());
                    }
                    u = trie[u].children[ch];
                }
                trie[u].types.insert(types.begin(), types.end());
            }
            // add the all lower form
            if (!noLowercases) {
                int u = 0;
                for (const string& token : tokens) {
                    const string& ch = toLower(token);
                    if (!trie[u].children.count(ch)) {
                        trie[u].children[ch] = trie.size();
                        trie.push_back(Node());
                    }
                    u = trie[u].children[ch];
                }
                trie[u].types.insert(types.begin(), types.end());
            }
        }

        inline void remove(const vector<string>& tokens) {
            int u = 0;
            for (const string& ch: tokens) {
                if (!trie[u].children.count(ch)) {
                    return;
                }
                u = trie[u].children[ch];
            }
            trie[u].types.clear();
        }

        inline bool inKB(const vector<string>& tokens) {
            int u = 0;
            for (const string& ch : tokens) {
                if (!trie[u].children.count(ch)) {
                    return false;
                }
                u = trie[u].children[ch];
            }
            return trie[u].types.size() > 0;
        }

        inline string getTypeFromKB(const vector<string>& tokens) {
            int u = 0;
            for (const string& ch : tokens) {
                if (!trie[u].children.count(ch)) {
                    return "";
                }
                u = trie[u].children[ch];
            }
            string ret = "";
            for (const string& type : trie[u].types) {
                if (ret.size() > 0) {
                    ret += ",";
                }
                ret += type;
            }
            return ret;
        }
    } // end namespace TrieForKB

void loadKBForMatching(const string& coreKBFilename, const string& fullKBFilename)
{
    FILE* in = tryOpen(coreKBFilename, "r");
    bool noLowercasesForThisKB = false;
    while (getLine(in)) {
        vector<string> tokens = splitBy(line, '\t');
        assert(tokens.size() == 2);
        vector<string> entity_types = splitBy(tokens[0], ',');
        assert(entity_types.size() >= 1);

        const string& surface = tokens[1];
        const string strip_surface = strip(surface);
        vector<string> surfaceTokens = splitBy(strip_surface, ' ');

        bool noLowercases = tokens[0].find("PER") != -1 || tokens[0].find("ORG") != -1  || tokens[0].find("LOC") != -1;
        noLowercasesForThisKB = noLowercases;
        if (!noLowercases) {
            for (const string& token : surfaceTokens) {
                if (stopwordSet.count(toLower(token))) {
                    noLowercases = true;
                    break;
                }
            }
        }
        TrieForKB::insert(surfaceTokens, entity_types, noLowercases);
    }
    fclose(in);
    cerr << "core dict inserted" << endl;

    in = tryOpen(fullKBFilename, "r");
    while (getLine(in)) {
        const string strip_surface = strip(line);
        vector<string> surfaceTokens = splitBy(strip_surface, ' ');
        TrieForKB::markAsFiltered(surfaceTokens, noLowercasesForThisKB);
    }
    fclose(in);
    cerr << "full dict marked" << endl;
}

void cleanStopwords(const string& filename)
{
    // remove unigram stopwords, one per line
    FILE* in = tryOpen(filename, "r");
    while (getLine(in)) {
        string token = strip(line);
        stopwordSet.insert(toLower(token));
        TrieForKB::remove({toLower(token)});
        TrieForKB::remove({toUpper(token)});
        token[0] = toupper(token[0]);
        TrieForKB::remove({token});
    }
    fclose(in);
}

void initialize(const string& coreKBFilename, const string& fullKBFilename, const string& stopwordFilename)
{
    TrieForKB::initialize();
    fprintf(stderr, "loading KB...\n");
    loadKBForMatching(coreKBFilename, fullKBFilename);
    fprintf(stderr, "cleaning stopwords...\n");
    cleanStopwords(stopwordFilename);
    fprintf(stderr, "initialized! # of trie nodes = %d\n", TrieForKB::trie.size());
}

// start the dp process

struct Traceback
{
    int i;
    set<string> types;
};

struct Token
{
    int l, r;
    string token, type;
    Token() { token = type = ""; }
    Token(string token) : token(token) { type = ""; }
    Token(string token, string type) : token(token), type(type) {}
};

struct AnnotatedData
{
    vector<string> rawTokens;
    vector<Token> annotatedTokens;

    vector<int> getBoundary() const { // 1 means Break, -1 means Connect, 0 means Unknown
        vector<int> ret(rawTokens.size(), 1);
        string lastType = "";
        for (int i = 0; i < annotatedTokens.size(); ++ i) {
            const Token& token = annotatedTokens[i];
            if (isSeparator(token.token) && !isRealSeparator(token.token)) {
                continue; // whitespace
            }
            if (token.type == FILTERED_TYPE) {
                ret[token.l] = 0;
                // assert(token.l + 1 == token.r);
                for (int j = token.l + 1; j <= token.r; ++ j) {
                    ret[j] = 0; // 0 means "not sure"
                }
                lastType = FILTERED_TYPE;
            } else if (token.type == "") {
                // O
                // by default, it's already a Break
                lastType = "O";
            } else {
                // entity
                ret[token.l] = 1;
                for (int j = token.l + 1; j < token.r; ++ j) {
                    ret[j] = -1;
                }
                lastType = "ENTITY";
            }
        }
        return ret;
    }

    vector<string> getTypes() const {
        vector<string> ret(rawTokens.size(), "None");
        for (int i = 0; i < annotatedTokens.size(); ++ i) {
            const Token& token = annotatedTokens[i];
            if (isSeparator(token.token) && !isRealSeparator(token.token)) {
                continue; // whitespace
            }
            if (token.type == FILTERED_TYPE) {
                // Filtered
            } else if (token.type == "") {
                // O
            } else {
                // entity
                for (int j = token.l; j < token.r; ++ j) {
                    ret[j] = token.type;
                }
            }
        }
        return ret;
    }

    string toCk() const {
        vector<int> boundary = getBoundary();
        vector<string> types = getTypes();

        stringstream sout;
        int status = 0; // outside
        for (int i = 0; i < boundary.size(); ++ i) {
            if (rawTokens[i] == "-DOCSTART-" || rawTokens[i] == "\n") {
                if (status == 1) {
                    sout << "<eof> I None S" << endl;
                    status = 0; // outside
                }
                sout << rawTokens[i] << endl;
                continue;
            }
            if (status != 1) {
                sout << "<s> O None S" << endl;
                status = 1; // inside
            }

            sout << rawTokens[i] << " ";
            if (boundary[i] == 0) {
                assert(types[i] == "None");
                sout << "O None D" << endl; // Unknown boundary
            } else if (boundary[i] == 1) {
                sout << "I " << types[i] << " S" << endl;
            } else {
                assert(boundary[i] == -1);
                sout << "O " << types[i] << " S" << endl;
            }
        }
        return sout.str();
    }

    string toBIOES() const {
        vector<int> boundary = getBoundary();
        vector<string> types = getTypes();

        stringstream sout;
        int status = 0; // outside
        for (int i = 0; i < boundary.size(); ++ i) {
            if (rawTokens[i] == "-DOCSTART-" || rawTokens[i] == "\n") {
                sout << endl;
                continue;
            }
            sout << rawTokens[i] << " ";
            if (boundary[i] == 0) {
                assert(types[i] == "None");
                // Unknown
                sout << "B-Chemical,I-Chemical,E-Chemical,S-Chemical,B-Disease,I-Disease,E-Disease,S-Disease,O" << endl;
            } else if (boundary[i] == 1) {
                if (types[i] == "None") {
                    // O
                    sout << " O" << endl;
                } else {
                    if (i + 1 < boundary.size() && boundary[i + 1] == -1) {
                        // B
                        sout << "B-" << types[i] << endl;
                    } else {
                        // S
                        sout << "S-" << types[i] << endl;
                    }
                }
            } else {
                assert(boundary[i] == -1);
                if (i + 1 < boundary.size() && boundary[i + 1] == -1) {
                    // I
                    sout << "I-" << types[i] << endl;
                } else {
                    // E
                    sout << "E-" << types[i] << endl;
                }
            }
        }
        return sout.str();
    }
};

AnnotatedData getDistantSupervision(const string& filename)
{
    FILE* in = tryOpen(filename, "r");
    vector<Token> matchedTokens;
    vector<string> rawTokens;
    while (getLine(in)) {
        vector<string> tokens = splitBy(line, ' ');
        if (tokens.size() == 0) {
            rawTokens.push_back("\n");
            continue;
        }
        bool inPhrase = false;
        vector<string> phrase;
        for (string token : tokens) {
            bool start = false, end = false;
            while (token.find("<phrase>") != -1) {
                int ptr = token.find("<phrase>");
                token = token.substr(0, ptr) + token.substr(ptr + 8);
                start = true;
            }
            while (token.find("</phrase>") != -1) {
                int ptr = token.find("</phrase>");
                token = token.substr(0, ptr) + token.substr(ptr + 9);
                end = true;
            }

            if (start) {
                // assert(!inPhrase);
                inPhrase = true;
                phrase.clear();
            }

            rawTokens.push_back(token);
            if (inPhrase) {
                phrase.push_back(token);
            } else {
                Token cur = Token(token, "");
                cur.l = rawTokens.size() - 1;
                cur.r = rawTokens.size();
                matchedTokens.push_back(cur);
            }

            if (end) {
                // assert(inPhrase);
                inPhrase = false;
                string type = TrieForKB::getTypeFromKB(phrase);
                if (type != "") {
                    string surface = "";
                    for (const string& token : phrase) {
                        if (surface.size() > 0) {
                            surface += " ";
                        }
                        surface += token;
                    }

                    Token cur = Token(surface, type);
                    cur.l = rawTokens.size() - phrase.size();
                    cur.r = rawTokens.size();
                    matchedTokens.push_back(cur);
                } else {
                    for (int i = 0; i < phrase.size(); ++ i) {
                        Token token(phrase[i], FILTERED_TYPE);
                        token.l = rawTokens.size() - phrase.size() + i;
                        token.r = token.l + 1;
                        matchedTokens.push_back(token);
                    }
                }
                
            }
        }
        assert(!inPhrase);
        rawTokens.push_back("\n");
    }
    fclose(in);

    AnnotatedData ret;
    ret.rawTokens = rawTokens;
    ret.annotatedTokens = matchedTokens;
    return ret;
}


} // end namespace StringMatch

#endif