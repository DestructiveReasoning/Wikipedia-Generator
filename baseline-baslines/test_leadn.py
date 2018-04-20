from leadn import *
import rouge

rouge = rouge.Rouge()

stories = []
targets = []
with open('../data/processed/val.bin', 'r') as f:
    line = f.readline()
    while(line):
        stories.append(line)
        targets.append(f.readline())
        line = f.readline()

test_stories = []
test_targets = []
with open('../data/processed/test.bin', 'r') as f:
    line = f.readline()
    while(line):
        test_stories.append(line)
        test_targets.append(f.readline())
        line = f.readline()

f = open("scores_leadn.csv", 'w')

best_result = {'score': 0, 'n': 0}

for n in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
    i = 0
    summaries = []
    for story in stories:
        i += 1
        print(str(n) + ": Writing summary " + str(i) + " of " + str(len(stories)), end='\r')
        summaries.append(make_summary(story, n))
    print('')
    scores = rouge.get_scores(summaries, targets, avg=True)
    if scores['rouge-1']['f'] > best_result['score']:
        best_result = {'score': scores['rouge-1']['f'], 'n': n}
    f.write(str(n) + "," + str(scores['rouge-1']['f']) + "," + str(scores['rouge-2']['f']) + "," + str(scores['rouge-l']['f']) + "\n")

print("Best result achieved: " + str(best_result['score']) + " at n=" + str(best_result['n']))

summaries = []
i = 0
for story in test_stories:
    i += 1
    print(str(best_result['n']) + ": Writing summary " + str(i) + " of " + str(len(test_stories)), end='\r')
    summaries.append(make_summary(story, best_result['n']))
print('')
scores = rouge.get_scores(summaries, targets, avg=True)
print(scores)
