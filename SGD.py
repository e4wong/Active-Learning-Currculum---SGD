import sys
import matplotlib.pyplot as plt
import random
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

def get_sample(training_set, policy):
    if policy == "random":
        return random.choice(training_set)


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
    for i in range(0, num_iterations):
        datum = get_sample(training_set, policy)
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
        print len(errors)
        output_final_w(result)
        f_0 = plt.figure(0)
        f_0.canvas.set_window_title(filename)
        plt.plot(errors)
        plt.ylabel('Objective Function')
        
        f_1 = plt.figure(1)
        f_1.canvas.set_window_title(filename)
        plt.plot(error_rate_trace)
        plt.ylabel('Error Rate Trace')
        plt.show()


if __name__ == '__main__':
    main()