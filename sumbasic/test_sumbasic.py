import rouge
import sumbasic
from sumbasic import *

stories = []
targets = []

valid_stories = []
valid_targets = []

rouge = rouge.Rouge()

with open("data/processed/test.bin", 'r') as f:
    line = f.readline()
    while line:
        stories.append(line)
        targets.append(f.readline())
        line = f.readline()

with open("data/processed/val.bin", 'r') as f:
    line = f.readline()
    while line:
        valid_stories.append(line)
        valid_targets.append(f.readline())
        line = f.readline()

N = len(stories)

summary_lengths = [80, 90, 100, 110]
n_grams = [1,2,3]

sf = open("scores_sumbasic.csv", 'w')

best_length = 0
best_n = 0
best_f1 = 0

T = len(valid_stories)
for length in summary_lengths:
    for n in n_grams:
        summaries = []
        print("\033[32;1mWriting " + str(length) + "-word summaries by " + str(n) + "-grams\033[0m")
        for story in valid_stories:
            summary = make_summary(story, length,n=n)
            summaries.append(summary)
            print("\tWrote summary " + str(len(summaries)) + " of " + str(T),end='\r')
        print('')
        print("\tCalculating scores...")
        scores = rouge.get_scores(summaries, valid_targets, avg=True)
        sf.write(str(length) + ",")
        sf.write(str(n) + ",")
        sf.write(str(scores['rouge-1']['f']) + ",")
        sf.write(str(scores['rouge-2']['f']) + ",")
        sf.write(str(scores['rouge-l']['f']) + "\n")
        if scores['rouge-1']['f'] > best_f1:
            best_f1 = scores['rouge-1']['f']
            best_length = length
            best_n = n

sf.close()

print("Best Rouge-1 score on validation set of " + str(100*best_f1) + " with " + str(best_length) + "-word summaries over " + str(best_n) + "-grams")
summaries = []
print("\033[33;1mCreating test-set summaries...\033[0m")
f = open("summaries_sumbasic.txt", 'w')
for story in stories:
    summary = make_summary(story, best_length, best_n)
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
