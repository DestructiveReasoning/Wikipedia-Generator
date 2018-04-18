import sumbasic
from sumbasic import *

stories = []
targets = []

with open("data/processed/shorttest.bin", 'r') as f:
    line = f.readline()
    while line:
        stories.append(line)
        targets.append(f.readline())
        line = f.readline()

summaries = []

for story in stories:
    summaries.append(make_summary(story, 100))
