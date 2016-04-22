import sys
import matplotlib.pyplot as plt
import random
import copy
import math
from library import *


exponents_lambda = [-4, -3, -2, -1, 0, 1]
base_lambda = 10
exponents_stepsize = [-3, -2, -1, 0, 1, 2, 3, 4]
base_stepsize = 2

def scale_w(lambda_, w):
    # if lambda = .001 , then 1/.001 causes overflow when we do e^ in derivative
    mag = float(1)/ float(lambda_)
    if linalg.norm(w) > mag:
        w = (1.0/linalg.norm(w)) * (1.0/float(lambda_))  * numpy.array(w)
    return w

def stepsize_fn(c, lambda_, t):
    config = 2
    if config == 2:
        return float(c) / float(lambda_ * t)
    else:
        return float(c) / numpy.sqrt(t)

def l2_norm(vec):
    accum = 0.0
    for val in vec:
        accum += val ** 2
    return math.sqrt(accum)

def get_sample(training_set, policy, w):
    if policy == "random":
        return random.choice(training_set)
    elif policy == "hard":
        # hard is the smallest l-2 norm of gradient
        curr_hard_l2 = 100.0 # some arbitrary big val
        curr_hard_sample = None
        for sample in training_set:
            if l2_norm(derivative(sample, w)) < curr_hard_l2:
                curr_hard_l2 = l2_norm(derivative(sample, w))
                curr_hard_sample = sample
        return curr_hard_sample
    elif policy == "easy":
        curr_easy_l2 = 0.0
        curr_easy_sample = None
        for sample in training_set:
            if l2_norm(derivative(sample, w)) > curr_easy_l2:
                curr_easy_l2 = l2_norm(derivative(sample, w))
                curr_easy_sample = sample
        return curr_easy_sample


def SGD(training_set, stepsize_constant, lambda_, error_log, test_set, num_iterations, policy, init_w=None):
    #print "Starting SGD algorithm with stepsize_constant: " + str(stepsize_constant) + " lambda: " + str(lambda_)
    init = rn.normal(size=(1, len(training_set[0][0])))[0]

    if not(init_w is None):
        init = init_w
        print "Initialized w for SGD run"

    w = scale_w(lambda_, init)
    errors = []
    error_rate_trace = []
    error_rate_trace.append(calc_error_rate(w, test_set))
    errors.append(total_error(w, lambda_, test_set))

    training_set = copy.deepcopy(training_set)
    
    for i in range(0, num_iterations):
        datum = get_sample(training_set, policy, w)
        training_set.remove(datum)
        # Stepsize is current c / sqrtroot(t)
        stepsize = stepsize_fn(stepsize_constant, lambda_, i + 1)
        delta = derivative(datum, w) + lambda_ * numpy.array(w)
        delta = stepsize * numpy.array(delta)
        w = w - delta
        w = scale_w(lambda_, w)
        if error_log:
            #Bottle neck right now
            errors.append(total_error(w, lambda_, test_set))
            error_rate_trace.append(calc_error_rate(w, test_set))
    #print "Final w from SGD is " + str(w) + "\n"
    return (w, (errors, error_rate_trace))

def tune_l1_norm_stepsize(training_set, validation_set, num_iterations, policy):
    global exponents_lambda
    global base_lambda

    global exponents_stepsize
    global base_stepsize
    
    best_mistakes = len(validation_set)
    best_lambda = 1.0
    best_stepsize = 1.0
    for exp_lambda in exponents_lambda:
        for exp_stepsize in exponents_stepsize:
            curr_lambda = float(math.pow(base_lambda, exp_lambda))
            curr_stepsize = float(math.pow(base_stepsize, exp_stepsize))
            (w, errors) = SGD(training_set, curr_stepsize, curr_lambda, False, validation_set, num_iterations, policy)
            mistakes = count_errors(w, validation_set)
            if mistakes < best_mistakes:
                best_mistakes = mistakes
                best_lambda = curr_lambda
                best_stepsize = curr_stepsize
    return (best_lambda, best_stepsize)

def avg(l):
    return float(sum(l))/len(l)

def curriculum_learning(sort_by, times_to_run, training_set, lambda_, stepsize_constant, test_set, num_iterations):
    avg_error_rate_trace = []
    avg_trace_objective_func_val = []
    final_objective_func_val = []
    error_rate = []
    print "Stepsize for", sort_by, "is", stepsize_constant
    print "Lambda for", sort_by, "is", lambda_
    for i in range(0, times_to_run):
        (w, (errors, error_rate_trace)) = SGD(training_set, stepsize_constant, lambda_, True, test_set, num_iterations, sort_by)
        
        if len(avg_error_rate_trace) == 0:
            avg_error_rate_trace = error_rate_trace
        elif len(avg_error_rate_trace) == len(error_rate_trace):
            tmp = [avg_error_rate_trace[i] + error_rate_trace[i] for i in range(0, len(avg_error_rate_trace))]
            avg_error_rate_trace = tmp
        else:
            print "Visit this problem"

        if len(avg_trace_objective_func_val) == 0:
            avg_trace_objective_func_val = errors
        else:
            length = 0.0
            if len(avg_trace_objective_func_val) < len(errors):
                length = len(avg_trace_objective_func_val)
            else:
                length = len(errors)
            for i in range(0, length):
                if avg_trace_objective_func_val[i] == float("inf"):
                    avg_trace_objective_func_val[i] = errors[i]
                elif errors[i] == float("inf"):
                    avg_trace_objective_func_val[i] += avg_trace_objective_func_val[i]/i
                else:
                    avg_trace_objective_func_val[i] += errors[i]
        error_rate.append(calc_error_rate(w, test_set))
        final_objective_func_val.append(total_error(w, lambda_, test_set))

    for i in range(0, len(avg_error_rate_trace)):
        avg_error_rate_trace[i] = avg_error_rate_trace[i]/times_to_run
    for i in range(0, len(avg_trace_objective_func_val)):
        avg_trace_objective_func_val[i] = avg_trace_objective_func_val[i]/times_to_run

    return (error_rate, final_objective_func_val, avg_trace_objective_func_val, avg_error_rate_trace)

def main():
    if len(sys.argv) < 2:
        print "Wrong way to use me!"
        return
    elif len(sys.argv) == 3:
        filename = sys.argv[1]    
        num_iterations = int(sys.argv[2])    
        (wstar, data) = load_data(filename)
        random.shuffle(data)

        test_set = data[2*len(data)/3 : ]

        data = data[ : 2*len(data)/3]
        training_set = data[len(data)/2 : ]
        validation_set = data[ : len(data)/2]
        print "Length of Test Set:", len(test_set)
        print "Length of Validation Set:", len(validation_set)
        print "Length of Training Set:", (len(training_set))

        default_policy = "random"
        (lambda_, stepsize_constant) = tune_l1_norm_stepsize(training_set, validation_set, num_iterations, default_policy)
        print "Lambda:", lambda_, "Stepsize Constant:", stepsize_constant

        (result, (errors, error_rate_trace)) = SGD(training_set, stepsize_constant, lambda_, True, test_set, num_iterations, default_policy)
        print "Test Set Error Rate: " + str(calc_error_rate(result, test_set))
        
        output_final_w(result)
        
        title = filename + " lambda: " + str(lambda_) + " stepsize constant: " + str(stepsize_constant) + " " + default_policy
        f_0 = plt.figure(0)
        f_0.canvas.set_window_title(title)
        plt.plot(errors)
        plt.ylabel('Objective Function')
        
        f_1 = plt.figure(1)
        f_1.canvas.set_window_title(title)
        plt.plot(error_rate_trace)
        plt.ylabel('Error Rate Trace')
        plt.show()
    elif len(sys.argv) == 4:
        filename = sys.argv[1]    
        num_iterations = int(sys.argv[2])   
        num_rounds = int(sys.argv[3])

        (wstar, data) = load_data(filename)
        random.shuffle(data)

        test_set = data[2*len(data)/3 : ]

        data = data[ : 2*len(data)/3]
        training_set = data[len(data)/2 : ]
        validation_set = data[ : len(data)/2]

        print "Number of Iterations:", num_iterations
        print "Number of Rounds for Curriculum Learning:", num_rounds
        print "Length of Test Set:", len(test_set)
        print "Length of Validation Set:", len(validation_set)
        print "Length of Training Set:", (len(training_set))

        default_policy = "random"
        (lambda_, stepsize_constant) = tune_l1_norm_stepsize(training_set, validation_set, num_iterations, default_policy)

        print "Running Hard"
        (hard_error_rate, hard_objective_func_val, obj_plot_hard, hard_err_trace) = curriculum_learning("hard", num_rounds, training_set, lambda_, stepsize_constant, test_set, num_iterations)
        print "Running Easy"
        (easy_error_rate, easy_objective_func_val, obj_plot_easy, easy_err_trace) = curriculum_learning("easy", num_rounds, training_set, lambda_, stepsize_constant, test_set, num_iterations)
        print "Running Random"
        (random_error_rate, random_objective_func_val, obj_plot_random, random_err_trace) = curriculum_learning("random", num_rounds, training_set, lambda_, stepsize_constant, test_set, num_iterations)


        print "Error Rate of w*:", calc_error_rate(wstar, data)

        print "Avg Error Rate Hard->Easy:", avg(hard_error_rate)
        print "Avg Error Rate Easy->Hard:", avg(easy_error_rate)
        print "Avg Error Rate random:", avg(random_error_rate)

        f_0 = plt.figure(0)
        f_0.canvas.set_window_title(filename)
        plt.plot(hard_error_rate)
        plt.plot(easy_error_rate)
        plt.plot(random_error_rate)
        plt.legend(['Hard Examples First', 'Easy Examples First', 'Normal/Random Examples'], loc='upper left')
        plt.ylabel('Error Rate')

        f_1 = plt.figure(1)
        f_1.canvas.set_window_title(filename)
        plt.plot(hard_objective_func_val)
        plt.plot(easy_objective_func_val)
        plt.plot(random_objective_func_val)
        plt.legend(['Hard Examples First', 'Easy Examples First', 'Normal/Random Examples'], loc='upper left')
        plt.ylabel("Final Objective Function Value")

        plt.figure(2)
        f_2 = plt.figure(2)
        f_2.canvas.set_window_title(filename)
        plt.plot(obj_plot_hard)
        plt.plot(obj_plot_easy)
        plt.plot(obj_plot_random)
        plt.legend(['Hard Examples First', 'Easy Examples First', 'Normal/Random Examples'], loc='upper right')
        plt.ylabel("Objective Function Value Trace")


        plt.figure(3)
        f_3 = plt.figure(3)
        f_3.canvas.set_window_title(filename)
        plt.plot(hard_err_trace)
        plt.plot(easy_err_trace)
        plt.plot(random_err_trace)
        plt.legend(['Hard Examples First', 'Easy Examples First', 'Normal/Random Examples'], loc='upper right')
        plt.ylabel("Error Rate Trace")



        plt.show()

if __name__ == '__main__':
    main()