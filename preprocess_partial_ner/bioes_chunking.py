import argparse
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Summarizing output')
    parser.add_argument('--input', default='./data/ner/eng.testb.iobes')
    parser.add_argument('--output', default='./data/ner/eng.testb.ck')
    parser.add_argument('--ignore_misc', action='store_true')
    args = parser.parse_args()

    with open(args.input, 'r') as fin, open(args.output, 'w') as fout:
        start = False
        alread_ends = False

        for line in fin:
            if line.isspace() or line.startswith('-DOCSTART-'):
                if start:
                    fout.write('<eof> I None\n\n')
                else:
                    fout.write('\n')
                start = False
            else:

                tups = line.split()
                label = tups[-1]
                
                if not start:

                    if 'O' == label or (args.ignore_misc and label.endswith('MISC')):
                    
                        fout.write('<s> O None\n' + tups[0] +' I None\n')
                    
                    else:
                    
                        fout.write('<s> O None\n' + tups[0] + ' I ' + label.split('-')[-1] + '\n' )

                        if label.startswith('S-') or label.startswith('E-'):
                            alread_ends = True
                    
                    start = True

                else:
                    
                    fout.write(tups[0])

                    if args.ignore_misc and label.endswith('MISC'):

                        fout.write(' I None\n')

                    elif label.startswith('B-'):

                        fout.write(' I ' + label.split('-')[-1] + '\n')
                        alread_ends = False

                    elif label.startswith('S-'):

                        fout.write(' I ' + label.split('-')[-1] + '\n')
                        alread_ends = True

                    elif label.startswith('E-'):
                        
                        fout.write(' O ' + label.split('-')[-1] + '\n')
                        alread_ends = True

                    elif label.startswith('I-'):
                        
                        fout.write(' O ' + label.split('-')[-1] + '\n')
                        alread_ends = False

                    elif alread_ends:
                        fout.write(' I None\n')
                        alread_ends = False
                    else:
                        fout.write(' I None\n')
                        alread_ends = False


        if start:
            fout.write('<eof> I None\n')
        else:
            fout.write('\n')
        