"""
figure out how reproducible a score is with the same parameter values
"""
# n = 150, repeat = 20, max-min = 0.10074650012974429
# n = 100, repeat 20, max-min = 0.142707468495
# n = 200, repeat 30, max-min = 0.0983575496398

# set verbose to False to prevent terminal from getting filled up
from roboskeeter import experiments
import matplotlib.pyplot as plt

scores = []
for i in range(30):
    experiment = experiments.start_simulation(200, None, None)
    score, _ = experiment.calc_score()
    scores.append(score)
    print "iteration ", i, " complete"

print max(scores) - min(scores)

fig = plt.figure()
plt.hist(scores)
plt.show()
