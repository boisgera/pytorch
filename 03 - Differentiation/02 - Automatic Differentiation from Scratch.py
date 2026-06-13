import marimo

__generated_with = "0.23.5"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Automatic Differentiation from Scratch

    [Sébastien Boisgérault], Mines Paris - PSL University

    [Sébastien Boisgérault]: mailto:Sebastien.Boisgerault@minesparis.psl.eu
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Introduction

    Automatic differentiation refers to a family of numerical methods used to compute derivatives, gradients, and Jacobian matrices of numerical functions. These methods have the major advantage of eliminating a significant portion of rounding errors typically associated with classical methods such as finite differences. The error is in fact as low as in a "manual" symbolic differentiation of functions without the need for delicate parameter tuning.

    In practice, if you were to implement manually the derivative of the function

    $$
    f: x \in \R \mapsto \frac{1-e^{-2x}}{1+e^{-2x}} \in \R,
    $$

    you would probably start with the implementation of the function $f$, for example as:
    """)
    return


@app.cell
def _(exp):
    def f(x):
        y = exp(-2.0 * x)
        u = 1.0 - y
        v = 1.0 + y
        w = u / v
        return w

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    and then proceed to implement its derivative $g:= f'$, typically as
    """)
    return


@app.cell
def _(exp):
    def g(x):
        y = exp(-2.0 * x)
        u = 1.0 - y
        v = 1.0 + y
        w = u / v
        dx = 1.0
        dy = -2.0 * exp(-2.0 * x) * dx
        du = 0.0 - dy
        dv = 0.0 + dy
        dw = du / v + u * (- dv) / (v * v)  
        return dw

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here, automatic differentiation would typically automate the generation of the function $g$ given the function $f$, giving you exactly the same results but avoiding all the tedious work (and the corresponding risks of manual computation errors).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Computation Graph
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Python being dynamically typed does not assign types to function arguments during their definition. Thus, the addition function,
    """)
    return


@app.function
def add_0(x, y):
    return x + y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    allows of course adding floating-point numbers:
    """)
    return


@app.cell
def _():
    add_0(1.0, 2.0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    but it will also work perfectly fine with integers or NumPy arrays or even non-numeric types like strings:
    """)
    return


@app.cell
def _():
    add_0(1, 2)
    return


@app.cell
def _():
    add_0("one", "two")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The key is that during execution, the objects `x` and `y` support the addition operation — otherwise, an exception will be raised. Thus, our add function is implicitly defined for addable objects.

    This is also confirmed by examining the bytecode of the add function, which makes no reference to the type of the arguments `x` and `y`:
    """)
    return


@app.cell
def _():
    from dis import dis
    dis(add_0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the case of addition, the operation `x + y` is delegated to the `__add__` method of the `x` object. To intercept this call, it is necessary to modify the type of floating-point numbers we are going to use and override the definition of this method:
    """)
    return


@app.class_definition
class Float_0(float):
    def __add__(self, other):
        print(f"trace: {self} + {other}")
        return super().__add__(other)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As our class inherits from the standard type float, operations we haven't explicitly redefined will be handled as usual. So, we've just modified addition for instances of `Float`, and in a very limited way since we've delegated the result calculation to the parent class float.

    Once this effort is made, we can indeed trace the additions made:
    """)
    return


@app.cell
def _():
    x = Float_0(2.0) + 1.0
    x
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    provided, of course, that we work with instances of `Float` and not `float`!
    To start generalizing this usage,
    we will ensure that operations on our floats return our own float type as much as possible:
    """)
    return


@app.class_definition
class Float(float):

    def __add__(self, other):
        print(f'trace: {self} + {other}')
        return Float(super().__add__(other))


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    But this is not enough: Python's math library functions will return regular floats,
    so we need to adapt them again; first, let's import the math module:
    """)
    return


@app.cell
def _():
    import math

    return (math,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    then define our own `cos` function:
    """)
    return


@app.cell
def _(math):
    def cos_0(x):
        print(f'trace: cos({x})')
        return Float(math.cos(x))

    return (cos_0,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's check the result:
    """)
    return


@app.cell
def _(cos_0):
    from math import pi
    cos_0(pi) + 1.0
    return (pi,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Unfortunately, we still cannot correctly trace the very similar expression `1.0 + cos(pi)`:
    """)
    return


@app.cell
def _(cos_0, pi):
    1.0 + cos_0(pi)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Indeed, it is the `__add__` method of `1.0`, an instance of `float`, that is called; thus, this call is not traced.
    To successfully handle this type of call, we need to ... make it fail!
    The method called to perform the sum so far entrusts the operation to the `__add__` method of 1.0
    because this object knows how to handle the operation, as it's
    about adding itself with another instance (deriving) of float.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If we ensure that the left operand is unable to handle this operation, it will be delegated to the right operand and the `__radd__` method; for this purpose, we simply replace `Float`, a numeric type, with `Node`, a class that contains (encapsulates) a numerical value:
    """)
    return


@app.class_definition
class Node_0:
    def __init__(self, value):
        self.value = value


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We won't dwell on this first version of `Node`. It's named like that because it will represent a node in a computation graph. Instead of displaying the operations performed on the standard output, we will record the operations each variable undergoes and how they are organized; each node resulting from an operation must remember which operation was applied and what the operation's arguments were (themselves nodes). To support this approach, `Node` becomes:
    """)
    return


@app.class_definition
class Node:
    def __init__(self, value, function=None, *args):
        self.value = value
        self.function = function
        self.args = args


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We then need to make the usual operations compatible with node creation;
    by examining the function's arguments, we must decide if it's in "normal" mode
    (receiving numerical values, producing numerical values) or tracing calculations.
    For example:
    """)
    return


@app.cell
def _(math):
    def cos(x):
        if isinstance(x, Node):
            cos_x_value = math.cos(x.value)
            cos_x = Node(cos_x_value, cos, x)
            return cos_x
        else:
            return math.cos(x)

    return (cos,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    or
    """)
    return


@app.function
def add(x, y):
    if isinstance(x, Node) or isinstance(y, Node):
        if not isinstance(x, Node):
            x = Node(x)
        if not isinstance(y, Node):
            y = Node(y)
        add_x_y_value = x.value + y.value
        return Node(add_x_y_value, add, x, y)
    else:
        return x + y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The add function probably won't be used directly but called using the + operator;
    therefore, it must allow us to define the `__add__` and `__radd__` methods:
    """)
    return


@app.cell
def _():
    Node.__add__ = add
    Node.__radd__ = add
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We notice many similarities between the two codes;
    rather than continuing this process for all the functions we will need,
    at the expense of abstraction effort,
    it would be possible to define a function that automatically performs this transformation.
    This function is a higher-order function because it takes a function (the original numerical function)
    as an argument and returns a new function compatible with node management.
    We can ignore its implementation on first reading.
    """)
    return


@app.function
def autodiff(function):

    def autodiff_function(*args):
        if any([isinstance(arg, Node) for arg in args]):
            node_args = []
            values = []
            for arg in args:
                if isinstance(arg, Node):
                    node_args.append(arg)
                    values.append(arg.value)
                else:
                    node_args.append(Node(arg))
                    values.append(arg)
            output_value = function(*values)
            output_node = Node(output_value, autodiff_function, *node_args)
            return output_node
        else:
            return function(*args)
    autodiff_function.__qualname__ = function.__qualname__
    return autodiff_function


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Despite its apparent complexity, using this function is simple;
    thus, to make the `sin` function and the `*` operator compatible with node management,
    all we need to do is:
    """)
    return


@app.cell
def _(math):
    sin = autodiff(math.sin)
    return (sin,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    and
    """)
    return


@app.cell
def _():
    def multiply(x, y):
        return x * y
    multiply = autodiff(multiply)
    Node.__mul__ = Node.__rmul__ = multiply
    return (multiply,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    which is considerably faster and more readable than the approach taken for `cos` and `+`;
    but once again, the result is the same.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It's now possible to implement the tracer.
    This function encapsulates the arguments of the function to trace into nodes,
    then calls the function and returns the node associated with the value returned by the function:
    """)
    return


@app.function
def trace(f, args):
    args = [Node(arg) for arg in args]
    end_node = f(*args)
    return end_node


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To verify that everything is working as expected, let's ensure we display a readable and friendly representation of the node contents as a string:
    """)
    return


@app.function
def node_str(node):
    if node.function is None:
        return str(node.value)
    else:
        function_name = node.function.__qualname__
        args_str = ", ".join(str(arg) for arg in node.args)
        return f"{function_name}({args_str})"


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Then, let's make sure it's used when invoking the `print` function, rather than the standard display:
    """)
    return


@app.cell
def _():
    Node.__str__ = node_str
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We complete this description with a second representation, more explicit but also more verbose:
    """)
    return


@app.cell
def _():
    def node_repr(node):
        reprs = [repr(node.value)]
        if node.function is not None:
            reprs.append(node.function.__qualname__)
        if node.args:
            reprs.extend([repr(arg) for arg in node.args])
        args_repr = ', '.join(reprs)
        return f'Node({args_repr})'
    Node.__repr__ = node_repr
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We're ready to do our verification:
    """)
    return


@app.cell
def _(cos, pi):
    def h(x):
        return 1.0 + cos(x)
    trace(h, [pi])
    return (h,)


@app.cell
def _(h, pi):
    print(trace(h, [pi]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The result is read as follows: the computation of `h(pi)` produces the value `0.0`,
    resulting from the addition of `-1.0`, calculated as `cos(3.141592653589793)`,
    and the constant `1.0`. So, this seems correct!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Another example, with two arguments, just for good measure:
    """)
    return


@app.cell
def _():
    def i(x, y):
        return x * (x + y)
    trace(i, [1.0, 2.0])
    return (i,)


@app.cell
def _(i):
    print(trace(i, [1.0, 2.0]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualization
    """)
    return


@app.cell
def _():
    import graphviz  

    def graph(end_node):
        dot = graphviz.Digraph()
        todo = [end_node]
        while todo:
            node = todo.pop()
            id_ = str(id(node))
            label = node.function.__qualname__ if node.function else str(node.value) 
            dot.node(id_, label)
            for arg in node.args:
                dot.node(str(id(arg)), node.function.__qualname__ or str(node.value))
                dot.edge(id_, str(id(arg)))
            todo.extend(node.args)
        return dot

    return (graph,)


@app.cell
def _(graph):
    def graph_function(f, *args):
        start_nodes = [Node(arg) for arg in args]
        end_node = f(*start_nodes)
        return graph(end_node)

    return (graph_function,)


@app.cell
def _(graph_function, mo):
    def j(x, y):
        return x * x + 2.0 * x * y + 1.0
    dot = graph_function(j, 3.0, 4.0)
    _filename = dot.render("tmp/graph", format="png")
    mo.center(mo.image(_filename))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Differential of Elementary Functions

    To leverage the computation graph we can now determine, we need to declare the differentials
    of primitive operations and functions in a "registry" of differentials,
    indexed by the function to differentiate.
    """)
    return


@app.cell
def _():
    differential = {}
    return (differential,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For addition and multiplication, we utilize the identities

    $$
    d(x+y) = dx + dy
    $$

    and

    $$
    d(x \times y) = x \times dy + dx \times y
    $$
    """)
    return


@app.cell
def _(differential, multiply):
    def d_add(x, y):
        return add
    differential[add] = d_add

    def d_multiply(x, y):

        def d_multiply_xy(dx, dy):
            return x * dy + dx * y
        return d_multiply_xy
    differential[multiply] = d_multiply
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For a function such as `cos`, we use the identity

    $$
    d (\cos(x)) = -\sin(x) dx
    $$
    """)
    return


@app.cell
def _(cos, differential, sin):
    def d_cos(x):
        def d_cos_x(dx):
            return -sin(x) * dx
        return d_cos_x
    differential[cos] = d_cos
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    But this is just a particular case of the identity $d (f(x)) = f'(x) dx$.
    We can equip ourselves with a function that will compute the differential $df$ from the derivative $f'$:
    """)
    return


@app.function
def d_from_deriv(g):
    def d_f(x):
        def d_f_x(dx):
            return g(x) * dx
        return d_f_x
    return d_f


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The declaration of differentials is thus simplified; for instance, from the identity $(\sin x)' = \cos x$, we have:
    """)
    return


@app.cell
def _(cos, differential, sin):
    differential[sin] = d_from_deriv(cos)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Differential of Composite Functions

    To leverage the tracing of a function, we need to extract from the final node produced by this process all the upstream nodes, which represent the arguments used in the calculation of the final value. Then, to prepare for the calculation of the differential, we will order the nodes so that the arguments of a function always appear before the value it produces. The following implementation, relatively naive, accomplishes this operation:
    """)
    return


@app.function
def find_and_sort_nodes(end_node):
    todo = [end_node]
    nodes = []
    while todo:
        node = todo.pop()
        nodes.append(node)
        for parent in node.args:
            if parent not in nodes + todo:
                todo.append(parent) 
    done = []
    while nodes:
        for node in nodes[:]:
            if all([parent in done for parent in node.args]):
                done.append(node)
                nodes.remove(node)
    return done


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The calculation of the differential itself then consists of propagating the variation of the node arguments
    from node to node, based on the chain rule of differentiation;
    these intermediate variations are stored in the `d_value` attribute of the graph nodes.
    """)
    return


@app.cell
def _(differential):
    def d(f):
        def df(*args):
            start_nodes = [Node(arg) for arg in args]
            end_node = f(*start_nodes)
            if not isinstance(end_node, Node):
                end_node = Node(end_node)
            nodes = find_and_sort_nodes(end_node).copy()

            def df_x(*d_args):
                for node in nodes:
                    if node in start_nodes:
                        i = start_nodes.index(node)
                        node.d_value = d_args[i]
                    elif node.function is None:
                        node.d_value = 0.0
                    else:
                        _d_f = differential[node.function]
                        _args = node.args
                        _args_values = [_node.value for _node in _args]
                        _d_args = [_node.d_value for _node in _args]
                        node.d_value = _d_f(*_args_values)(*_d_args)
                return end_node.d_value
            return df_x
        return df

    return (d,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Usage

    To simply leverage our differential calculation, we can deduce the derivative in the case of a function of a real variable; recall that we have $f'(x) = df(x) \cdot 1$.
    """)
    return


@app.cell
def _(d):
    def deriv(f):
        df = d(f)
        def deriv_f(x):
            return df(x)(1.0)
        return deriv_f

    return (deriv,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's verify that the behavior of these differentiation operators meets our expectations for single-variable functions; first, in the case of a constant function:
    """)
    return


@app.cell
def _(deriv, pi):
    def k(x):
        return pi
    k_1 = deriv(k)
    return (k_1,)


@app.cell
def _(k_1):
    k_1(0.0)
    return


@app.cell
def _(k_1):
    k_1(1.0)
    return


@app.cell
def _(k_1):
    k_1(2.0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Then, in the case of an affine function:
    """)
    return


@app.cell
def _(deriv):
    def l(x):
        return 2.0 * x + 1.0
    l_1 = deriv(l)
    return (l_1,)


@app.cell
def _(l_1):
    l_1(0.0)
    return


@app.cell
def _(l_1):
    l_1(1.0)
    return


@app.cell
def _(l_1):
    l_1(2.0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And finally, in the case of a quadratic function:
    """)
    return


@app.cell
def _(deriv):
    def m(x):
        return x * x + 2 * x + 1
    m_1 = deriv(m)
    return (m_1,)


@app.cell
def _(m_1):
    m_1(0.0)
    return


@app.cell
def _(m_1):
    m_1(1.0)
    return


@app.cell
def _(m_1):
    m_1(2.0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To conclude within this framework,
    let's test two functions using trigonometric functions `sin` and `cos`:
    """)
    return


@app.cell
def _(cos, deriv, sin):
    def n(x):
        return cos(x) * cos(x) + sin(x) * sin(x)
    n_1 = deriv(n)
    return (n_1,)


@app.cell
def _(n_1):
    n_1(0.0)
    return


@app.cell
def _(n_1):
    n_1(1.0)
    return


@app.cell
def _(n_1):
    n_1(2.0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the general case, since our functions always yield real values, we can deduce the gradient of the differential:
    """)
    return


@app.cell
def _(d):
    def grad(f):
        df = d(f)
        def grad_f(*args):
            n = len(args)
            grad_f_x = n * [0.0]
            df_x = df(*args)
            for i in range(0, n):
                e_i = n * [0.0]; e_i[i] = 1.0
                grad_f_x[i] = df_x(*e_i)
            return grad_f_x  
        return grad_f

    return (grad,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Similarly, constant, affine, and quadratic functions allow for elementary tests:
    """)
    return


@app.function
def o(x, y):
    return 1.0


@app.cell
def _(grad):
    grad(o)(0.0, 0.0)
    return


@app.cell
def _(grad):
    grad(o)(1.0, 2.0)
    return


@app.function
def p(x, y):
    return x + 2.0 * y + 1.0


@app.cell
def _(grad):
    grad(p)(0.0, 0.0)
    return


@app.cell
def _(grad):
    grad(p)(1.0, 2.0)
    return


@app.function
def f_10(x, y):
    return x * x + y * y


@app.cell
def _(grad):
    grad(p)(0.0, 0.0)
    return


@app.cell
def _(grad):
    grad(p)(1.0, 2.0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Backward Differentiation

    If we only intend to compute gradients of scalar-valued functions, we can use  **backward differentiation** instead of forwards differentiation which is more efficient in this case.

    Note that in Machine Learning, most of the time we are only interested in the gradient of the loss function with respect to the model parameters, so we are exactly in this scenario. For this reason, backward differentiation is very popular.
    """)
    return


@app.cell
def _():
    gradient = {}
    return (gradient,)


@app.cell
def _(cos, gradient, multiply, sin):
    def grad_sin(x):
        return [cos(x)]
    gradient[sin] = grad_sin

    def grad_cos(x):
        return [-sin(x)]
    gradient[cos] = grad_cos

    def grad_add(x, y):
        return [1.0, 1.0]
    gradient[add] = grad_add

    def grad_multiply(x, y):
        return [y, x]
    gradient[multiply] = grad_multiply
    return


@app.cell
def _(gradient):
    def backward_grad(f):
        def grad_f(*args):
            start_nodes = [Node(arg) for arg in args]
            end_node = f(*start_nodes)
            if not isinstance(end_node, Node):
                end_node = Node(end_node)
            nodes = find_and_sort_nodes(end_node)
            nodes.reverse()
            end_node_args_values = [arg.value for arg in end_node.args]
            end_node.grad = gradient[end_node.function](*end_node_args_values)
            for node in nodes:
                if node.function is not None:
                    assert len(node.grad) == len(node.args)
                    for i, node_i in enumerate(node.args):
                        g_i = node.grad[i]
                        if not hasattr(node_i, 'grad'):
                            node_i.grad = max(1, len(node_i.args)) * [0.0]
                        if node_i.function is None:
                            gradient_ = [1.0]
                        else:
                            node_args_values = [arg.value for arg in node_i.args]
                            gradient_ = gradient[node_i.function](*node_args_values)
                        extra_grad = [g_i * g for g in gradient_]
                        node_i.grad = [g + xg for g, xg in zip(node_i.grad, extra_grad)]
            return [node.grad[0] for node in start_nodes]
        return grad_f

    return (backward_grad,)


@app.cell
def _(backward_grad):
    def q(x):
        return 0.0 * x * x
    grad_q = backward_grad(q)
    return (grad_q,)


@app.cell
def _(grad_q):
    grad_q(1.0)
    return


@app.cell
def _(grad_q):
    grad_q(2.0)
    return


@app.cell
def _(backward_grad):
    def r(x, y):
        return x * x + 2.0 * x * y + 3.0 * y * y + 1.0
    grad_r = backward_grad(r)
    return (grad_r,)


@app.cell
def _(grad_r):
    grad_r(0.0, 0.0)
    return


@app.cell
def _(grad_r):
    grad_r(1.0, 0.0)
    return


@app.cell
def _(grad_r):
    grad_r(0.0, 1.0)
    return


@app.cell
def _(grad_r):
    grad_r(2.0, 3.0)
    return


if __name__ == "__main__":
    app.run()
