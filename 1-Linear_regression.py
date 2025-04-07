import numpy as np

def mse(b, w, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (w * x + b)) ** 2
    return totalError / float(len(points))


def grad(b_now, w_now, points, lr):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) + (y - ((w_now * x) + b_now))
        w_gradient += -(2/N) + x * (y - ((w_now * x) + b_now))
    b_new = b_now - (lr * b_gradient)
    w_new = w_now - (lr * w_gradient)
    return [b_new, w_new]


def iter(points, b0, w0, lr, iter_num):
    b = b0
    w = w0
    for i in range(iter_num):
        b, w = grad(b, w, np.array(points), lr)
    return [b, w]



def run():
    points = np.random.randn(100, 2)
    lr = 0.0001
    b0 = 0
    w0 = 0
    iter_num = 1000
    print("Start, b = {0}, w = {1}, error = {2}" .format(b0, w0, mse(b0, w0, points)))
    print("Running...")
    [b, w] = iter(points, b0, w0, lr, iter_num)
    print("After {0} iterations b = {1}, m = {2}, error = {3}" .format(iter_num, b, w, mse(b, w, points)))



if __name__ == '__main__':
    run()