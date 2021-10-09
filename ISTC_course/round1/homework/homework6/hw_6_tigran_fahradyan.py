import numpy as np
from scipy.stats import multivariate_normal

def counter(func):

    def wrapper(self, *args, **kwargs):
        self.func_call_count += 1
        ret = func(self, *args, **kwargs)
        if type(ret) is any((int, float)): self.returned_values_list.append(ret)
        return ret
    return wrapper


class Sandbox:

    def __init__(self, name):
        """
        :param name: name of the sandbox
        """
        self.name = name
        self.func_call_count = 0
        self.returned_values_list = []

    @counter
    def print_name(self):
        # print your name from self
        print(self.name)

    @staticmethod
    def normal_distribution_pdf(x, mean, sigma):
        # return normal distribution probability density
        return multivariate_normal.pdf(x, mean, sigma)

    @counter
    def print_all_calculated_numbers(self):
        # print all number ever returned by this class
        print(self.returned_values_list)

    @counter
    def return_sum_of_all_calculated_numbers(self):
        # return sum of all number ever returned by this class
        return sum(self.returned_values_list) if len(self.returned_values_list) != 0 else 0

    @staticmethod
    def sum(*args):
        # return sum of args
        return sum(args)

    @counter
    def revert_name(self):
        # return reverted name of the sandbox
        return self.name[::-1] # or self.name.revert()

    @counter
    def number_of_calls(self):
        # return number of times any method was called
        return self.func_call_count


if __name__ == "__main__":
    sandbox = Sandbox("tigran")
    sandbox.print_name()
    print(sandbox.revert_name())
    sandbox.print_all_calculated_numbers()
    print('Normal distribution probability density :', Sandbox.normal_distribution_pdf(np.linspace(0, 5, 10), mean=2.5, sigma=0.5))
    print('Sum of all numbers ever returned by this class :', sandbox.return_sum_of_all_calculated_numbers())
    print(sandbox.number_of_calls())
