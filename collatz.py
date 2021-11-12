"""
The Collatz conjecture is a conjecture in mathematics that concerns
sequences defined as follows: start with any positive integer n. Then
each term is obtained from the previous term as follows: if the
previous term is even, the next term is one half of the previous term.
If the previous term is odd, the next term is 3 times the previous term
plus 1. The conjecture is that no matter what value of n, the sequence
will always reach 1.

f(n) = \frac{n}{2} if n \equiv 0
f(n) = 3n + 1 if n \equiv 1
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import warnings

from typing import (List,
                    Literal,
                    Optional)


class Collatz(object):
    def __init__(self, n: List[int]):
        """
        The Collatz object only needs a list of positive integers (if
        one integer is passed it must be wrapped in a list). Then, you
        can use either the recursive or the iterative method for
        solving the problem.

        The object also has a method to determine the best and the
        worst initial numbers in terms of iterations.
        """
        self.n_array = self.__delete_invalid(n)

    def __delete_invalid(self, n: List[int]):
        """
        Private method for delete the invalid initial numbers, which
        are redundant for this problem. Thanks to this method, the user
        can pass a np.arange(x) without worries.
        """
        invalid = [0, 1, 2, 4]
        for i in invalid:
            try:
                n.remove(i)
                warnings.warn('Initial n = ' + str(i) + ' is not a valid '
                              'initial number. It will be dropped from the '
                              'list.')
            except:
                pass

        return n

    def __eureka(self, results: pd.DataFrame):
        """
        In case some of the initial numbers break the Collatz
        conjecture, a print will follow congratulating the lucky
        bastard that used my code to find it.

        Parameters
        ----------
        results: pd.DataFrame
            A DataFrame with the results provided by one of the two
            methods.
        """
        unique_results = results.loc['Last numbers'].drop_duplicates()
        if len(unique_results) > 1:
            for result in unique_results:
                if result != [4, 2, 1]:
                    print('EUREKA! A million bucks for the result', result)

    def iterative_collatz(self) -> pd.DataFrame:
        """
        Solving the problem in an iterative way.

        Returns
        -------
        df: pd.DataFrame
            A DataFrame with the results: number of iterations and the
            last 3 numbers of the series.
        """
        start = time.time()
        global_results = []
        for n in self.n_array:
            n_results = []
            while True:
                n_results.append(n)
                if n == 1:
                    break
                elif n % 2 == 0:
                    n = int(n / 2)
                else:
                    n = 3*n + 1
            global_results.append(n_results)

        n_iters = [len(n_iter) - 1 for n_iter in global_results]
        last_results = [last[-3:] for last in global_results]
        df = pd.DataFrame((n_iters, last_results), columns=[self.n_array],
                          index=['Number of iterations', 'Last numbers'])
        self.__eureka(df)
        time_spent = round(((time.time() - start) / 60), 2)
        print('Time spent for the iterative method:', time_spent, 'minutes')
        return df

    def recursive_collatz(self) -> pd.DataFrame:
        """
        Solving the problem in a recursive way. Actually, this is an
        iterative method who calls a recursive method.

        Returns
        -------
        df: pd.DataFrame
            A DataFrame with the results: number of iterations and the
            last 3 numbers of the series.
        """
        start = time.time()
        n_iters = []
        last_results = []
        for n in self.n_array:
            results = self.__aux_recursive_collatz(n, [])
            n_iters.append(len(results) - 1)
            last_results.append(results[-3:])

        df = pd.DataFrame((n_iters, last_results), columns=[self.n_array],
                          index=['Number of iterations', 'Last numbers'])
        self.__eureka(df)
        time_spent = round(((time.time() - start) / 60), 2)
        print('Time spent for the recursive method:', time_spent, 'minutes')
        return df

    def __aux_recursive_collatz(self, n: int, results: List[int] = []):
        """
        This private method is the true recursive method that only
        works with an integer.
        """
        results.append(n)
        if n == 1:
            return results
        elif n % 2 == 0:
            return self.__aux_recursive_collatz(int(n / 2), results)
        else:
            return self.__aux_recursive_collatz(3*n + 1, results)

    @staticmethod
    def best_worst_n(results: pd.DataFrame) -> tuple:
        """
        This method determines the initial number that has the fewer
        iterations and the one who has the most.

        Parameters
        ----------
        results: pd.DataFrame
            A DataFrame with the results provided by one of the two
            methods.

        Returns
        -------
        min_iter, n_inital_min, max_iter, n_inital_max: tuple
            A tuple with 4 integers.
        """
        min_iter = min(results.loc['Number of iterations'].array)
        n_inital_min = results.loc['Number of iterations'].index[
            results.loc['Number of iterations'] == min_iter].to_numpy()
        n_inital_min = int(n_inital_min[0][0])

        max_iter = max(results.loc['Number of iterations'].array)
        n_inital_max = results.loc['Number of iterations'].index[
            results.loc['Number of iterations'] == max_iter].to_numpy()
        n_inital_max = int(n_inital_max[0][0])

        return min_iter, n_inital_min, max_iter, n_inital_max


n_array = np.arange(2e6).tolist()
collatz = Collatz(n_array)
iterative = collatz.iterative_collatz()
recursive = collatz.recursive_collatz()
min_iter, n_inital_min, max_iter, n_inital_max = collatz.best_worst_n(iterative)

# Setting the plot variables
y = iterative.loc['Number of iterations'].array
x = iterative.columns.values
x = [np.asarray(i) for i in x]
# Trendline for the plot
x_poly = np.asarray(x).reshape(len(x))
y_poly = np.asarray([np.asarray(i) for i in y])
z = np.polyfit(x_poly, y_poly, 1)
p = np.poly1d(z)
# Plotting
plt.plot(x, y)
plt.plot(x, p(x), "r--")
# Showing the best and the worst initial n in the plot
plt.text(n_inital_min, min_iter, str((n_inital_min, min_iter)))
plt.text(n_inital_max, max_iter, str((n_inital_max, max_iter)))
plt.ylabel('Number of iterations')
plt.xlabel('Initial n')
plt.show()
