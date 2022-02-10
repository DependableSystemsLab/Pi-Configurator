import pandas as pd
from z3 import *
import time
import tracemalloc
ts = time.time()
tracemalloc.start()

# Put your dataset here. It'll need an index column
data = pd.read_csv('', index_col=0)

index = data.index.values

#################### List of avaiable features need to be optimized ###############################

availableSet = data['Param'].values  # configuration parameters

received_latency = data['WallClock'].values

received_cpu = data['CPUCycle'].values

received_size = data['Memory'].values

received_accuracy = data['Accuracy'].values

#################### List of avaiable features need to be optimized ###############################

# create a new Z3 solver instance.
s = Optimize()
# to reproduce its result each time the same
set_param("smt.random_seed", 1234)
z3.set_param('auto_config', False,
             'smt.phase_selection', 5,
             'smt.arith.random_initial_value', True,
             'smt.random_seed', 1234,
             'sat.phase', 'random',
             'sat.random_seed', 1234)


class Job:
    def __init__(self, name, latencies, accuracies, cpus, sizes):
        self.name = name

        self.latencies = dict()

        self.lowest_possible_job_time = min(latencies.values())

        for node, runtime in latencies.items():
            self.latencies[node] = IntVal(runtime)

        self.accuracies = dict()
        for node, latency in accuracies.items():
            self.accuracies[node] = IntVal(latency)

        self.cpus = dict()
        for node, cpu in cpus.items():
            self.cpus[node] = IntVal(cpu)

        self.sizes = dict()
        for node, size in sizes.items():
            self.sizes[node] = IntVal(size)


class Setting:
    def __init__(self, name):
        self.name = name


nodes = []
for i in range(len(index)):
    settings = availableSet.item(i)
    k = index.item(i)
    nodes.append(Setting("%d_%s" % (k, settings)))

jobs = []
# for j in range(4):
latency_dict = dict()
accuracy_dict = dict()
cpu_dict = dict()
size_dict = dict()
count_n = 0
for n in nodes:
    lat = received_latency.item(count_n)
    latency_dict[n] = lat
    acu = received_accuracy.item(count_n)
    accuracy_dict[n] = acu
    cpu = received_cpu.item(count_n)
    cpu_dict[n] = cpu
    size = received_size.item(count_n)
    size_dict[n] = size
    count_n += 1

jobs.append(Job("j%d" % (0), latency_dict, accuracy_dict, cpu_dict, size_dict))


job_placements = dict()
for j in jobs:
    job_placements[j] = dict()


# assert that, each job has to be placed on exactly one node
for j in jobs:
    node_choices = []
    for n in nodes:
        p = Bool("pipe_%s_on_%s" % (j.name, n.name))
        node_choices.append(p)
        job_placements[j][n] = p
    # Assert that each job is placed on _exactly_ one node
    # for the current job j assert that exactly one of p is true
    s.add(Sum([If(b, 1, 0) for b in node_choices]) == 1)


#################### Interchangeable Objectives #######################################


# minimize the runtime
runtimes = []
for j in jobs:
    job_runtime = IntVal(0)
    for n in nodes:
        p = job_placements[j][n]
        job_runtime = If(p, j.latencies[n], job_runtime)
    # runtimes.append(job_runtime)
# total_runtime = Sum(runtimes)
# help Z3 by calculating, in advance, a lower bound on the possible sum total runtime
# known_minimum_time = sum(j.lowest_possible_job_time for j in jobs)
# s.add(job_runtime >= IntVal(known_minimum_time))
# The objective function should be an integer (or real) that Z3 will minimize.
s.minimize(job_runtime)
print("minimzed job_runtime")

# # maximize the sum total accuracy
for j in jobs:
    job_accuracy = IntVal(0)
    for n in nodes:
        p = job_placements[j][n]
        job_accuracy = If(p, j.accuracies[n], job_accuracy)
s.maximize(job_accuracy)
print("maximized job_accuracy")

# # # minimize the cpu time
for j in jobs:
    job_cpu = IntVal(0)
    for n in nodes:
        p = job_placements[j][n]
        job_cpu = If(p, j.cpus[n], job_cpu)
s.minimize(job_cpu)
print("minimzed job_cpu")

# minimize transmission volume
for j in jobs:
    job_size = IntVal(0)
    for n in nodes:
        p = job_placements[j][n]
        job_size = If(p, j.sizes[n], job_size)
s.minimize(job_size)
print("minimzed job_size")


#################### Interchangeable Objectives #######################################

print("Solving...")

if s.check():  # attempt to solve the instance, and return True if it could be solved
    m = s.model()  # the model contains the actual assignments found by Z3 for each variable
    for j in jobs:
        placements = job_placements[j]
        found = False
        for n, p in placements.items():  # p is if job j is placed on node n
            if is_true(m[p]):
                assert(not found)
                found = True
                print("For job %s select Setting %s" % (j.name, n.name))

        assert(found)
else:
    print("Could not find a valid placement")

print("Total number of nodes %d" % len(index))
tf = time.time()
td = tf-ts
_, first_peak = tracemalloc.get_traced_memory()
memoryUSed = round((first_peak/10**6), 2)
print("Time took to schedule is %d" % td+" seconds")
print("Memory Taken: ", memoryUSed)
