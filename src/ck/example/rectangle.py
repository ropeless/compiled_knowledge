import random as _random
from typing import Callable

import numpy as _np

from ck.pgm import PGM


class Rectangle(PGM):
    """
    This PGM is an interpretation of the 'rectangle' Bayesian network as discussed
    in "An Advance on Variable Elimination with Applications to Tensor-Based Computation",
    Adnan Darwiche, ECAI-2020.
    """

    def __init__(
            self,
            n: int = 8,
            duplicate_functions: int = 1,
            extend_label: bool = False,
            random_seed: int = 129837697,
    ):
        """
        Args:
            n: is the image size (n * n pixels).
            extend_label: if true extends the rectangle label to 'tall', 'wide', 'square', 'none'.
            duplicate_functions: is the multiplier for duplicates of the functional CPTs.
                Its minimum is 1 (no duplication) and maximum is n (fully duplicated as per Darwiche 2020).
                If 'duplicate_functions' is outside this range, it will be clipped to the extreme (1 or n).
                If 'duplicate_functions' is 'full' then it will be set to the maximum, n.
        """
        params = (n, duplicate_functions, extend_label)
        super().__init__(f'{self.__class__.__name__}({",".join(str(param) for param in params)})')

        duplicate_functions = n if duplicate_functions == 'full' else duplicate_functions
        duplicate_functions = max(1, min(duplicate_functions, n))  # clip value
        start_duplicate_functions = n - duplicate_functions + 1

        name = f'rectangle_{n}x{n}_dup({duplicate_functions})'
        if extend_label:
            name += '(extend_label)'

        random_stream = _random.Random(random_seed).random
        rand_iter = iter(random_stream, None)

        rect_row = self.new_rv('rect_row', n)
        rect_col = self.new_rv('rect_col', n)
        rect_height = self.new_rv('rect_height', n)
        rect_width = self.new_rv('rect_width', n)
        if extend_label:
            rect_label = self.new_rv('label', ['tall', 'wide', 'square', 'none'])
        else:
            rect_label = self.new_rv('label', ['tall', 'wide'])
        rows = [self.new_rv(f'row_{i}', 2) for i in range(n)]
        cols = [self.new_rv(f'col_{i}', 2) for i in range(n)]
        dup_rows = [[]] * n
        dup_cols = [[]] * n
        for k in range(n):
            if k < start_duplicate_functions:
                dup_rows[k] = rows
                dup_cols[k] = cols
            else:
                # Each rows[i] and cols[i] RV will be the child of a functional CPT.
                # and there children will be all pixs[i,j]
                # In Darwiche 2020, functional CPTs get duplicated for each child
                dup_rows[k] = [self.new_rv(f'row_{i}_dup_{k}', 2) for i in range(n)]
                dup_cols[k] = [self.new_rv(f'col_{i}_dup_{k}', 2) for i in range(n)]

        pixs = [[self.new_rv(f'pix_{i},{j}', 2) for j in range(n)] for i in range(n)]

        self.new_factor(rect_row).set_dense().set_iter(rand_iter)
        self.new_factor(rect_col).set_dense().set_iter(rand_iter)

        f_height = self.new_factor(rect_height, rect_row).set_sparse()
        for row in range(len(rect_row)):
            max_height = n - row
            if max_height > 0:
                cpd = _random_cpd(max_height - 1, 0, random_stream)  # -1 as height cannot be zero
                for height in range(1, max_height):
                    f_height[height, row] = cpd.item(height - 1)

        f_width = self.new_factor(rect_width, rect_col).set_sparse()
        for col in range(len(rect_col)):
            max_width = n - col
            if max_width > 0:
                cpd = _random_cpd(max_width - 1, 0, random_stream)  # -1 as width cannot be zero
                for width in range(1, max_width):
                    f_width[width, col] = cpd.item(width - 1)

        # functional relationship: rect_label <- rect_height, rect_width
        f_label = self.new_factor(rect_label, rect_height, rect_width).set_sparse()
        if extend_label:
            for height in range(len(rect_height)):
                for width in range(len(rect_width)):
                    if height == 0 or width == 0:
                        label = 3  # none
                    elif height > width:
                        label = 0  # tall
                    elif height < width:
                        label = 1  # wide
                    else:
                        label = 2  # square
                    f_label[label, height, width] = 1
        else:
            for height in range(len(rect_height)):
                for width in range(len(rect_width)):
                    label = 0 if height > width else 1
                    f_label[label, height, width] = 1

        # functional relationship: rows[i] <- rect_row, rect_height
        f_row = []
        for i in range(n):
            f_row_i = self.new_factor(rows[i], rect_row, rect_height).set_sparse()
            f_row.append(f_row_i)
            for row in range(len(rect_row)):
                for height in range(len(rect_height)):
                    if row <= i < row + height:
                        f_row_i[1, row, height] = 1
                    else:
                        f_row_i[0, row, height] = 1

        # functional relationship: cols[i] <- rect_col, rect_width
        f_col = []
        for i in range(n):
            f_col_i = self.new_factor(cols[i], rect_col, rect_width).set_sparse()
            f_col.append(f_col_i)
            for col in range(len(rect_col)):
                for width in range(len(rect_width)):
                    if col <= i < col + width:
                        f_col_i[1, col, width] = 1
                    else:
                        f_col_i[0, col, width] = 1

        # patch in the potential functions for the duplicate factors
        for k in range(start_duplicate_functions, n):
            for i in range(n):
                self.new_factor(dup_rows[k][i], rect_row, rect_height).function = f_row[i]
                self.new_factor(dup_cols[k][i], rect_col, rect_width).function = f_col[i]

        # connect rows and cols to pixels
        for i in range(n):
            for j in range(n):
                self.new_factor(pixs[i][j], dup_rows[j][i], dup_cols[i][j]).set_dense().set_iter(rand_iter)


def _random_cpd(size: int, sparsity: int, random_stream: Callable[[], float]):
    cpd = _np.zeros(size)
    if sparsity <= 0:
        for i in range(len(cpd)):
            cpd[i] = 0.0000001 + random_stream()
        cpd /= _np.sum(cpd)
    else:
        for i in range(len(cpd)):
            if random_stream() <= sparsity:
                cpd[i] = 0
            else:
                cpd[i] = 0.0000001 + random_stream()
        sum_value = _np.sum(cpd)
        if sum_value > 0:
            cpd /= sum_value
    return cpd
