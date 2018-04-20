import rouge
import sumbasic
from sumbasic import *

stories = []
targets = []

valid_stories = []
valid_targets = []

rouge = rouge.Rouge()

with open("../data/processed/test.bin", 'r') as f:
    line = f.readline()
    while line:
        stories.append(line)
        targets.append(f.readline())
        line = f.readline()

with open("../data/processed/val.bin", 'r') as f:
    line = f.readline()
    while line:
        valid_stories.append(line)
        valid_targets.append(f.readline())
        line = f.readline()

N = len(stories)

summary_lengths = [80, 90, 100]
n_grams = [1,2,3]
lambdas = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

sf = open("scores_sumbasic.csv", 'w')

best_result = {'n': 0, 'length': 0, 'lambda': 0, 'score': 0}

num_experiments = len(summary_lengths) * len(n_grams) * len(lambdas)
experiment = 0

T = len(valid_stories)
sf.write("Summary Length, n-gram, lambda, score\n")
for lam in lambdas:
    for length in summary_lengths:
        for n in n_grams:
            experiment += 1
            summaries = []
            print("\033[32;1mWriting " + str(length) + "-word summaries by " + str(n) + "-grams with lambda = " \
                    + str(lam) + "\t[" + str(experiment) + "/" + str(num_experiments) + "]\033[0m")
            for story in valid_stories:
                summary = make_summary(story, length, n=n, lam=lam)
                summaries.append(summary)
                print("\tWrote summary " + str(len(summaries)) + " of " + str(T),end='\r')
            print('')
            print("\tCalculating scores...")
            scores = rouge.get_scores(summaries, valid_targets, avg=True)
            sf.write(str(length) + ",")
            sf.write(str(n) + ",")
            sf.write(str(lam) + ",")
            sf.write(str(scores['rouge-1']['f']) + ",")
            sf.write(str(scores['rouge-2']['f']) + ",")
            sf.write(str(scores['rouge-l']['f']) + "\n")
            if scores['rouge-1']['f'] > best_result['score']:
                best_result['score'] = scores['rouge-1']['f']
                best_result['lambda'] = lam
                best_result['n'] = n
                best_result['length'] = length

sf.close()

print("Best Rouge-1 score on validation set of " + str(100*best_result['score']) + " with " + \
        str(best_result['length']) + "-word summaries over " + str(best_result['n']) + \
        "-grams and lambda = " + str(best_result['lambda']))
summaries = []
print("\033[33;1mCreating test-set summaries...\033[0m")
f = open("summaries_sumbasic.txt", 'w')
for story in stories:
    summary = make_summary(story, best_result['length'], n=best_result['n'], lam=best_result['lambda'])
    summaries.append(summary)
    f.write(summary + '\n')
    print("\033[32mWrote summary " + str(len(summaries)) + " of " + str(N) + "\033[0m",end='\r')
f.close()
print('')
print("Calculating scores...")
scores = rouge.get_scores(summaries, targets, avg=True)
print("Achieved the following average Rouge F1 scores on the test set:")
print("\tRouge-1: " + str(100 * scores["rouge-1"]['f']))
print("\tRouge-2: " + str(100 * scores["rouge-2"]['f']))
print("\tRouge-L: " + str(100 * scores["rouge-l"]['f']))
